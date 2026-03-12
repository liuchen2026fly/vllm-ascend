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
Fused split/reshape kernel for Qwen3.5 GatedDeltaNet input projections.

This kernel replaces 5-6 Python tensor operations:
  1. qkvz_out[:, :qkv_dim] → mixed_qkv
  2. qkvz_out[:, qkv_dim:] → z (temp)
  3. z.reshape(...) → z (reshaped)
  4. ba_out[:, :b_dim] → b
  5. ba_out[:, b_dim:] → a
  + multiple .contiguous() calls

Benefits:
  - Single kernel launch instead of 5-6 operations
  - Eliminates intermediate tensor allocations
  - Coalesced memory access
  - Expected 5-10x speedup for this stage
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _fused_qwen3_5_split_kernel(
    # Input tensors
    qkvz_out_ptr,
    ba_out_ptr,
    # Output tensors
    mixed_qkv_ptr,
    z_ptr,
    b_ptr,
    a_ptr,
    # Split dimensions
    qkv_out_dim: tl.constexpr,
    b_out_dim: tl.constexpr,
    # Reshape parameters for z
    num_v_heads: tl.constexpr,
    head_v_dim: tl.constexpr,
    # Tensor dimensions
    num_tokens: tl.constexpr,
    qkvz_total_dim: tl.constexpr,
    ba_total_dim: tl.constexpr,
    # Block size
    BLOCK_SIZE: tl.constexpr,
):
    """
    Fused kernel for Qwen3.5 projection splitting and reshaping.

    Each program processes one or more tokens (rows).

    Processing per token:
      1. Load qkvz_out row
      2. Split and store: mixed_qkv = qkvz_out[:qkv_out_dim]
      3. Split and reshape: z = qkvz_out[qkv_out_dim:].reshape(num_v_heads, head_v_dim)
      4. Load ba_out row
      5. Split and store: b = ba_out[:b_out_dim]
      6. Split and store: a = ba_out[b_out_dim:]
    """
    pid = tl.program_id(0)

    # Each program processes one token (row)
    token_idx = pid

    if token_idx >= num_tokens:
        return

    # ============================================================
    # Part 1: Process QKVZ output
    # ============================================================

    # Calculate row pointers
    qkvz_row_ptr = qkvz_out_ptr + token_idx * qkvz_total_dim
    mixed_qkv_row_ptr = mixed_qkv_ptr + token_idx * qkv_out_dim

    # Copy mixed_qkv part (first qkv_out_dim elements)
    for offset in range(0, qkv_out_dim, BLOCK_SIZE):
        col_idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < qkv_out_dim

        data = tl.load(qkvz_row_ptr + col_idx, mask=mask, other=0.0)
        tl.store(mixed_qkv_row_ptr + col_idx, data, mask=mask)

    # Copy and reshape z part (remaining elements)
    # z starts at qkv_out_dim, has shape (num_v_heads, head_v_dim)
    z_start_offset = qkv_out_dim
    z_total_dim = num_v_heads * head_v_dim
    z_row_ptr = z_ptr + token_idx * z_total_dim

    for offset in range(0, z_total_dim, BLOCK_SIZE):
        col_idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < z_total_dim

        # Load from qkvz_out (flat layout)
        data = tl.load(qkvz_row_ptr + z_start_offset + col_idx, mask=mask, other=0.0)
        # Store to z (reshaped layout - but actually same layout, just different view)
        tl.store(z_row_ptr + col_idx, data, mask=mask)

    # ============================================================
    # Part 2: Process BA output
    # ============================================================

    ba_row_ptr = ba_out_ptr + token_idx * ba_total_dim
    b_row_ptr = b_ptr + token_idx * b_out_dim
    a_out_dim = ba_total_dim - b_out_dim
    a_row_ptr = a_ptr + token_idx * a_out_dim

    # Copy b part (first b_out_dim elements)
    for offset in range(0, b_out_dim, BLOCK_SIZE):
        col_idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < b_out_dim

        data = tl.load(ba_row_ptr + col_idx, mask=mask, other=0.0)
        tl.store(b_row_ptr + col_idx, data, mask=mask)

    # Copy a part (remaining elements)
    for offset in range(0, a_out_dim, BLOCK_SIZE):
        col_idx = offset + tl.arange(0, BLOCK_SIZE)
        mask = col_idx < a_out_dim

        data = tl.load(ba_row_ptr + b_out_dim + col_idx, mask=mask, other=0.0)
        tl.store(a_row_ptr + col_idx, data, mask=mask)


def fused_qwen3_5_split_reshape(
    qkvz_out: torch.Tensor,
    ba_out: torch.Tensor,
    qkv_out_dim: int,
    b_out_dim: int,
    num_v_heads: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Fused split and reshape operation for Qwen3.5 projections.

    Replaces the following Python operations:
        mixed_qkv = qkvz_out[:, :qkv_out_dim].contiguous()
        z = qkvz_out[:, qkv_out_dim:].contiguous()
        z = z.reshape(z.size(0), num_v_heads, head_v_dim)
        b = ba_out[:, :b_out_dim].contiguous()
        a = ba_out[:, b_out_dim:].contiguous()

    Args:
        qkvz_out: Output of fused qkvz projection, shape (num_tokens, qkv_dim + z_dim)
        ba_out: Output of fused ba projection, shape (num_tokens, b_dim + a_dim)
        qkv_out_dim: Dimension of qkv output (split point for qkvz_out)
        b_out_dim: Dimension of b output (split point for ba_out)
        num_v_heads: Number of value heads (for z reshape)
        head_v_dim: Dimension per value head (for z reshape)

    Returns:
        Tuple of (mixed_qkv, z, b, a):
            mixed_qkv: shape (num_tokens, qkv_out_dim)
            z: shape (num_tokens, num_v_heads, head_v_dim)
            b: shape (num_tokens, b_out_dim)
            a: shape (num_tokens, a_out_dim)
    """
    # Validate inputs
    num_tokens = qkvz_out.shape[0]
    assert qkvz_out.shape[0] == ba_out.shape[0], "Batch size mismatch"
    assert qkvz_out.shape[1] >= qkv_out_dim, "qkv_out_dim exceeds qkvz_out dimension"
    assert ba_out.shape[1] >= b_out_dim, "b_out_dim exceeds ba_out dimension"

    qkvz_total_dim = qkvz_out.shape[1]
    ba_total_dim = ba_out.shape[1]
    z_out_dim = qkvz_total_dim - qkv_out_dim
    a_out_dim = ba_total_dim - b_out_dim

    # Validate z reshape dimensions
    assert z_out_dim == num_v_heads * head_v_dim, (
        f"z dimension mismatch: z_out_dim={z_out_dim}, "
        f"num_v_heads * head_v_dim={num_v_heads * head_v_dim}"
    )

    # Allocate output tensors
    mixed_qkv = torch.empty(
        (num_tokens, qkv_out_dim),
        dtype=qkvz_out.dtype,
        device=qkvz_out.device,
    )
    z = torch.empty(
        (num_tokens, num_v_heads, head_v_dim),
        dtype=qkvz_out.dtype,
        device=qkvz_out.device,
    )
    b = torch.empty(
        (num_tokens, b_out_dim),
        dtype=ba_out.dtype,
        device=ba_out.device,
    )
    a = torch.empty(
        (num_tokens, a_out_dim),
        dtype=ba_out.dtype,
        device=ba_out.device,
    )

    # Kernel configuration
    BLOCK_SIZE = 256  # Good balance for NPU memory hierarchy
    grid = (num_tokens,)

    # Launch kernel
    _fused_qwen3_5_split_kernel[grid](
        qkvz_out,
        ba_out,
        mixed_qkv,
        z.view(num_tokens, -1),  # Flatten z for kernel
        b,
        a,
        qkv_out_dim,
        b_out_dim,
        num_v_heads,
        head_v_dim,
        num_tokens,
        qkvz_total_dim,
        ba_total_dim,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return mixed_qkv, z, b, a


def test_fused_qwen3_5_split():
    """
    Test correctness against Python reference implementation.
    """
    # Test configuration
    num_tokens = 32
    qkv_out_dim = 4096
    z_out_dim = 2048
    b_out_dim = 64
    a_out_dim = 64
    num_v_heads = 32
    head_v_dim = z_out_dim // num_v_heads

    device = 'npu' if torch.npu.is_available() else 'cpu'
    dtype = torch.float16

    # Generate random inputs
    torch.manual_seed(42)
    qkvz_out = torch.randn(num_tokens, qkv_out_dim + z_out_dim, device=device, dtype=dtype)
    ba_out = torch.randn(num_tokens, b_out_dim + a_out_dim, device=device, dtype=dtype)

    # Python reference implementation
    mixed_qkv_ref = qkvz_out[:, :qkv_out_dim].contiguous()
    z_ref = qkvz_out[:, qkv_out_dim:].contiguous()
    z_ref = z_ref.reshape(num_tokens, num_v_heads, head_v_dim)
    b_ref = ba_out[:, :b_out_dim].contiguous()
    a_ref = ba_out[:, b_out_dim:].contiguous()

    # Triton fused implementation
    if device == 'npu':
        mixed_qkv_tri, z_tri, b_tri, a_tri = fused_qwen3_5_split_reshape(
            qkvz_out, ba_out, qkv_out_dim, b_out_dim, num_v_heads, head_v_dim
        )

        # Verify correctness
        torch.testing.assert_close(mixed_qkv_tri, mixed_qkv_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(z_tri, z_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(b_tri, b_ref, rtol=1e-5, atol=1e-5)
        torch.testing.assert_close(a_tri, a_ref, rtol=1e-5, atol=1e-5)

        print("✓ All outputs match reference implementation!")
        print(f"  mixed_qkv: {mixed_qkv_tri.shape}")
        print(f"  z: {z_tri.shape}")
        print(f"  b: {b_tri.shape}")
        print(f"  a: {a_tri.shape}")
    else:
        print("⚠ NPU not available, skipping Triton test")


if __name__ == "__main__":
    test_fused_qwen3_5_split()
