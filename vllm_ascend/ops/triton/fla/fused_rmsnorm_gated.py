#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""
Fused RMSNorm + Gating Triton kernel for Qwen3.5 GatedDeltaNet.

This kernel fuses two operations into one:
1. RMS Normalization: x_normed = x / sqrt(mean(x^2) + eps) * weight
2. Element-wise gating: output = x_normed * z

Benefits over separate operations:
- Single kernel launch instead of two
- No intermediate tensor allocation
- Better memory locality
"""

import torch
from vllm.triton_utils import tl, triton
from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def _fused_rmsnorm_gated_kernel(
    output_ptr,
    input_ptr,
    z_ptr,
    weight_ptr,
    input_row_stride,
    z_row_stride,
    output_row_stride,
    n_rows,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused RMSNorm + Gating kernel.

    Each program processes multiple rows.
    For each row:
      1. Compute RMS: rms = sqrt(mean(x^2) + eps)
      2. Normalize: x_normed = x / rms * weight
      3. Gate: output = x_normed * z

    Args:
        output_ptr: Output tensor pointer (n_rows, n_cols)
        input_ptr: Input tensor x pointer (n_rows, n_cols)
        z_ptr: Gating tensor z pointer (n_rows, n_cols)
        weight_ptr: RMSNorm weight pointer (n_cols,)
        input_row_stride: Stride for input rows
        z_row_stride: Stride for z rows
        output_row_stride: Stride for output rows
        n_rows: Number of rows to process
        n_cols: Number of columns (hidden dimension)
        eps: Epsilon for numerical stability
        BLOCK_SIZE: Block size for column processing
    """
    pid = tl.program_id(0)
    n_programs = tl.num_programs(0)

    # Each program processes multiple rows
    rows_per_program = (n_rows + n_programs - 1) // n_programs
    start_row = pid * rows_per_program
    end_row = tl.minimum(start_row + rows_per_program, n_rows)

    for row_idx in range(start_row, end_row):
        # Calculate row base pointers
        input_row_ptr = input_ptr + row_idx * input_row_stride
        z_row_ptr = z_ptr + row_idx * z_row_stride
        output_row_ptr = output_ptr + row_idx * output_row_stride

        # ==============================================================
        # Step 1: Compute sum of squares (in float32 for precision)
        # ==============================================================
        sum_sq = tl.zeros([1], dtype=tl.float32)
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols

            x = tl.load(input_row_ptr + col_idx, mask=mask, other=0.0)
            x_f32 = x.to(tl.float32)
            sq_x = x_f32 * x_f32
            sum_sq += tl.sum(tl.where(mask, sq_x, 0.0))

        # ==============================================================
        # Step 2: Compute RMS (root mean square)
        # ==============================================================
        mean_sq = sum_sq / n_cols
        rms = tl.sqrt(mean_sq + eps)
        inv_rms = 1.0 / rms

        # ==============================================================
        # Step 3: Normalize, apply weight, and gate (fused)
        # ==============================================================
        for col_offset in range(0, n_cols, BLOCK_SIZE):
            col_idx = col_offset + tl.arange(0, BLOCK_SIZE)
            mask = col_idx < n_cols

            # Load inputs
            x = tl.load(input_row_ptr + col_idx, mask=mask, other=0.0)
            z = tl.load(z_row_ptr + col_idx, mask=mask, other=0.0)
            w = tl.load(weight_ptr + col_idx, mask=mask, other=1.0)

            # Convert to float32 for computation
            x_f32 = x.to(tl.float32)
            z_f32 = z.to(tl.float32)
            w_f32 = w.to(tl.float32)

            # Fused: normalize + weight + gate
            x_normed = x_f32 * inv_rms * w_f32
            output_f32 = x_normed * z_f32

            # Convert back to original dtype
            output = output_f32.to(x.dtype)

            # Store output
            tl.store(output_row_ptr + col_idx, output, mask=mask)


def fused_rmsnorm_gated(
    x: torch.Tensor,
    z: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    Fused RMSNorm + Gating operation using Triton.

    Computes: output = (x / sqrt(mean(x^2) + eps) * weight) * z

    This is equivalent to:
        x_normed = RMSNorm(x, weight, eps)
        output = x_normed * z
    but fused into a single kernel for better performance.

    Args:
        x: Input tensor of shape (..., hidden_size)
        z: Gating tensor of shape (..., hidden_size)
        weight: RMSNorm weight tensor of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        Output tensor of shape (..., hidden_size)

    Example:
        >>> x = torch.randn(16, 64, 128, device='npu')
        >>> z = torch.randn(16, 64, 128, device='npu')
        >>> weight = torch.ones(128, device='npu')
        >>> out = fused_rmsnorm_gated(x, z, weight, eps=1e-6)
        >>> out.shape
        torch.Size([16, 64, 128])
    """
    # Input validation
    assert x.shape == z.shape, (
        f"Input x and gating z must have same shape, "
        f"got x.shape={x.shape}, z.shape={z.shape}"
    )
    assert weight.dim() == 1, f"Weight must be 1-dimensional, got {weight.dim()}"
    assert x.shape[-1] == weight.shape[0], (
        f"Input last dimension ({x.shape[-1]}) must match "
        f"weight dimension ({weight.shape[0]})"
    )

    # Flatten all dimensions except the last one
    original_shape = x.shape
    x_2d = x.reshape(-1, x.shape[-1]).contiguous()
    z_2d = z.reshape(-1, z.shape[-1]).contiguous()
    weight = weight.contiguous()

    n_rows, n_cols = x_2d.shape

    # Allocate output
    output = torch.empty_like(x_2d, dtype=x.dtype)

    # Kernel configuration
    BLOCK_SIZE = 1024
    max_grid_size = get_vectorcore_num()
    grid = (min(n_rows, max_grid_size),)

    # Launch kernel
    _fused_rmsnorm_gated_kernel[grid](
        output,
        x_2d,
        z_2d,
        weight,
        x_2d.stride(0),
        z_2d.stride(0),
        output.stride(0),
        n_rows,
        n_cols,
        eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output.reshape(original_shape)


def test_fused_rmsnorm_gated():
    """
    Simple test to verify correctness against PyTorch implementation.
    """
    import torch.nn.functional as F

    # Test configuration
    batch, n_heads, head_dim = 16, 64, 64
    eps = 1e-6
    device = 'npu' if torch.npu.is_available() else 'cpu'

    # Generate random inputs
    torch.manual_seed(0)
    x = torch.randn(batch, n_heads, head_dim, device=device, dtype=torch.float32)
    z = torch.randn(batch, n_heads, head_dim, device=device, dtype=torch.float32)
    weight = torch.ones(head_dim, device=device, dtype=torch.float32)

    # PyTorch reference implementation
    x_flat = x.reshape(-1, head_dim)
    variance = x_flat.pow(2).mean(dim=-1, keepdim=True)
    x_normed = x_flat * torch.rsqrt(variance + eps) * weight
    pytorch_out = (x_normed * z.reshape(-1, head_dim)).reshape(batch, n_heads, head_dim)

    # Triton fused implementation
    if device == 'npu':
        triton_out = fused_rmsnorm_gated(x, z, weight, eps=eps)

        # Check correctness
        max_diff = (pytorch_out - triton_out).abs().max().item()
        mean_diff = (pytorch_out - triton_out).abs().mean().item()

        print(f"Max difference: {max_diff:.6e}")
        print(f"Mean difference: {mean_diff:.6e}")

        assert max_diff < 1e-3, f"Output mismatch: max diff = {max_diff}"
        print("✓ Test passed!")
    else:
        print("⚠ NPU not available, skipping Triton test")


if __name__ == "__main__":
    test_fused_rmsnorm_gated()
