#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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
# This file is a part of the vllm-ascend project.
#

"""
Fused MLA partial RoPE + Q/K assembly Triton kernel.

In MLA (Multi-head Latent Attention), after q_b_proj, the q tensor has shape
[num_tokens, num_heads, qk_head_dim] where qk_head_dim = nope_dim + rope_dim.
Only the rope portion needs RoPE rotation. The k tensor is assembled from
k_nope (per-head) and k_pe (shared MQA, after RoPE).

This kernel fuses:
  1. Split q into nope/rope parts
  2. Apply RoPE to q_rope
  3. Reassemble q_out = [q_nope | RoPE(q_rope)]
  4. Apply RoPE to k_pe (shared across heads)
  5. Assemble k_out = [k_nope | RoPE(k_pe)]

Reducing 5-7 kernel launches to 1.
"""

import torch
from vllm.triton_utils import tl, triton

from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num


@triton.jit
def mla_partial_rope_kernel(
    # Input pointers
    q_ptr,              # [num_tokens, num_heads, qk_head_dim]
    k_nope_ptr,         # [num_tokens, num_heads, nope_dim]
    k_pe_ptr,           # [num_tokens, 1, rope_dim]
    # Output pointers
    q_out_ptr,          # [num_tokens, num_heads, qk_head_dim]
    k_out_ptr,          # [num_tokens, num_heads, qk_head_dim]
    # RoPE cache
    cos_sin_cache_ptr,  # [max_pos, rope_dim]
    positions_ptr,      # [num_tokens]
    # Dimensions
    num_tokens,
    num_heads,
    # Constexpr dimensions
    nope_dim: tl.constexpr,
    rope_dim: tl.constexpr,
    qk_head_dim: tl.constexpr,
    HALF_ROPE_DIM: tl.constexpr,
    PAD_HALF_ROPE_DIM: tl.constexpr,
    PAD_NOPE_DIM: tl.constexpr,
    IS_NEOX_STYLE: tl.constexpr,
):
    """
    Fused MLA partial RoPE kernel.

    Each program processes multiple tokens (strided by grid size).
    For each token:
      - Load cos/sin from cache (once per token)
      - Apply RoPE to k_pe (once per token, shared across heads)
      - For each head: copy q_nope, apply RoPE to q_rope, copy k_nope,
        store RoPE(k_pe)

    cos_sin_cache layout: [max_pos, rope_dim] where
      cache[pos, :rope_dim//2] = cos values
      cache[pos, rope_dim//2:] = sin values

    IS_NEOX_STYLE behavior:
      True:  Read/write in neox layout (first-half/second-half split)
      False: Read from interleaved positions (even/odd indices), write to
             neox positions (first-half/second-half). This matches DeepSeek's
             convention of shuffle-to-neox + neox-style RoPE.
    """
    pid = tl.program_id(0)
    num_programs = tl.num_programs(0)

    # Precompute offset ranges
    cos_offsets = tl.arange(0, PAD_HALF_ROPE_DIM)
    cos_mask = cos_offsets < HALF_ROPE_DIM
    nope_offsets = tl.arange(0, PAD_NOPE_DIM)
    nope_mask = nope_offsets < nope_dim

    for token_idx in tl.range(pid, num_tokens, num_programs):
        # Load position index and fetch cos/sin from cache
        pos = tl.load(positions_ptr + token_idx).to(tl.int64)
        cos_sin_base = pos * rope_dim

        cos = tl.load(
            cos_sin_cache_ptr + cos_sin_base + cos_offsets,
            mask=cos_mask, other=0.0
        ).to(tl.float32)
        sin = tl.load(
            cos_sin_cache_ptr + cos_sin_base + HALF_ROPE_DIM + cos_offsets,
            mask=cos_mask, other=0.0
        ).to(tl.float32)

        # Apply RoPE to k_pe (shared across all heads, compute once)
        k_pe_base = token_idx * rope_dim
        if IS_NEOX_STYLE:
            k_pe_1 = tl.load(
                k_pe_ptr + k_pe_base + cos_offsets,
                mask=cos_mask, other=0.0
            ).to(tl.float32)
            k_pe_2 = tl.load(
                k_pe_ptr + k_pe_base + HALF_ROPE_DIM + cos_offsets,
                mask=cos_mask, other=0.0
            ).to(tl.float32)
            roped_k_pe_1 = k_pe_1 * cos - k_pe_2 * sin
            roped_k_pe_2 = k_pe_2 * cos + k_pe_1 * sin
        else:
            # Non-neox style: interleaved layout
            # x1 = x[::2], x2 = x[1::2]
            even_offsets = 2 * cos_offsets
            odd_offsets = even_offsets + 1
            even_mask = even_offsets < rope_dim
            k_pe_1 = tl.load(
                k_pe_ptr + k_pe_base + even_offsets,
                mask=even_mask, other=0.0
            ).to(tl.float32)
            k_pe_2 = tl.load(
                k_pe_ptr + k_pe_base + odd_offsets,
                mask=even_mask, other=0.0
            ).to(tl.float32)
            roped_k_pe_1 = k_pe_1 * cos - k_pe_2 * sin
            roped_k_pe_2 = k_pe_2 * cos + k_pe_1 * sin

        # Process each head
        for head_idx in tl.range(0, num_heads):
            # Base offsets for this token/head in q and output
            q_base = (token_idx * num_heads + head_idx) * qk_head_dim
            out_base = (token_idx * num_heads + head_idx) * qk_head_dim

            # --- Q processing ---
            # Copy q_nope (unchanged)
            q_nope = tl.load(
                q_ptr + q_base + nope_offsets,
                mask=nope_mask, other=0.0
            )
            tl.store(
                q_out_ptr + out_base + nope_offsets,
                q_nope, mask=nope_mask
            )

            # Apply RoPE to q_rope
            if IS_NEOX_STYLE:
                q_rope_1 = tl.load(
                    q_ptr + q_base + nope_dim + cos_offsets,
                    mask=cos_mask, other=0.0
                ).to(tl.float32)
                q_rope_2 = tl.load(
                    q_ptr + q_base + nope_dim + HALF_ROPE_DIM + cos_offsets,
                    mask=cos_mask, other=0.0
                ).to(tl.float32)
                new_q_1 = q_rope_1 * cos - q_rope_2 * sin
                new_q_2 = q_rope_2 * cos + q_rope_1 * sin
                tl.store(
                    q_out_ptr + out_base + nope_dim + cos_offsets,
                    new_q_1.to(q_out_ptr.dtype.element_ty),
                    mask=cos_mask
                )
                tl.store(
                    q_out_ptr + out_base + nope_dim + HALF_ROPE_DIM + cos_offsets,
                    new_q_2.to(q_out_ptr.dtype.element_ty),
                    mask=cos_mask
                )
            else:
                # Non-neox (DeepSeek style): read interleaved, write neox
                even_offsets_q = 2 * cos_offsets
                odd_offsets_q = even_offsets_q + 1
                even_mask_q = even_offsets_q < rope_dim
                q_rope_1 = tl.load(
                    q_ptr + q_base + nope_dim + even_offsets_q,
                    mask=even_mask_q, other=0.0
                ).to(tl.float32)
                q_rope_2 = tl.load(
                    q_ptr + q_base + nope_dim + odd_offsets_q,
                    mask=even_mask_q, other=0.0
                ).to(tl.float32)
                new_q_1 = q_rope_1 * cos - q_rope_2 * sin
                new_q_2 = q_rope_2 * cos + q_rope_1 * sin
                # Write to neox positions (first-half / second-half)
                tl.store(
                    q_out_ptr + out_base + nope_dim + cos_offsets,
                    new_q_1.to(q_out_ptr.dtype.element_ty),
                    mask=cos_mask
                )
                tl.store(
                    q_out_ptr + out_base + nope_dim + HALF_ROPE_DIM + cos_offsets,
                    new_q_2.to(q_out_ptr.dtype.element_ty),
                    mask=cos_mask
                )

            # --- K processing ---
            # Copy k_nope
            k_nope_base = (token_idx * num_heads + head_idx) * nope_dim
            k_nope = tl.load(
                k_nope_ptr + k_nope_base + nope_offsets,
                mask=nope_mask, other=0.0
            )
            tl.store(
                k_out_ptr + out_base + nope_offsets,
                k_nope, mask=nope_mask
            )

            # Store RoPE(k_pe) — same rotated values for all heads
            # Both neox and non-neox write to neox positions (first-half /
            # second-half), since non-neox follows DeepSeek convention of
            # producing neox-layout output.
            tl.store(
                k_out_ptr + out_base + nope_dim + cos_offsets,
                roped_k_pe_1.to(k_out_ptr.dtype.element_ty),
                mask=cos_mask
            )
            tl.store(
                k_out_ptr + out_base + nope_dim + HALF_ROPE_DIM + cos_offsets,
                roped_k_pe_2.to(k_out_ptr.dtype.element_ty),
                mask=cos_mask
            )


def mla_partial_rope_impl(
    q: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    nope_dim: int,
    rope_dim: int,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Fused MLA partial RoPE + Q/K assembly.

    Args:
        q: [num_tokens, num_heads, qk_head_dim] — q_b_proj output reshaped
        k_nope: [num_tokens, num_heads, nope_dim] — from kv_b_proj split
        k_pe: [num_tokens, 1, rope_dim] — from kv_a_proj split
        cos_sin_cache: [max_pos, rope_dim] — rotary embedding cache
        positions: [num_tokens] — position indices
        nope_dim: dimension of the non-rotary part
        rope_dim: dimension of the rotary part
        is_neox_style: whether to use neox-style RoPE

    Returns:
        q_out: [num_tokens, num_heads, qk_head_dim] — q with RoPE on rope part
        k_out: [num_tokens, num_heads, qk_head_dim] — assembled k with RoPE
    """
    # Ensure contiguity for simple offset computation
    if not q.is_contiguous():
        q = q.contiguous()
    if not k_nope.is_contiguous():
        k_nope = k_nope.contiguous()
    if not k_pe.is_contiguous():
        k_pe = k_pe.contiguous()

    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    qk_head_dim = q.shape[2]
    half_rope_dim = rope_dim // 2

    # Allocate outputs
    q_out = torch.empty_like(q)
    k_out = torch.empty(
        num_tokens, num_heads, qk_head_dim,
        device=q.device, dtype=q.dtype
    )

    # Flatten k_pe from [num_tokens, 1, rope_dim] to [num_tokens, rope_dim]
    k_pe_flat = k_pe.view(num_tokens, rope_dim)

    # Determine grid
    num_vectorcores = get_vectorcore_num()
    n_programs = min(num_tokens, num_vectorcores)

    # Pad dimensions to next power of 2 for Triton
    pad_half_rope_dim = triton.next_power_of_2(half_rope_dim)
    pad_nope_dim = triton.next_power_of_2(nope_dim)

    mla_partial_rope_kernel[(n_programs,)](
        q,
        k_nope,
        k_pe_flat,
        q_out,
        k_out,
        cos_sin_cache,
        positions,
        num_tokens,
        num_heads,
        nope_dim,
        rope_dim,
        qk_head_dim,
        half_rope_dim,
        pad_half_rope_dim,
        pad_nope_dim,
        is_neox_style,
    )

    return q_out, k_out


def mla_partial_rope_fake(
    q: torch.Tensor,
    k_nope: torch.Tensor,
    k_pe: torch.Tensor,
    cos_sin_cache: torch.Tensor,
    positions: torch.Tensor,
    nope_dim: int,
    rope_dim: int,
    is_neox_style: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fake implementation for shape inference during Dynamo/AOT tracing."""
    num_tokens = q.shape[0]
    num_heads = q.shape[1]
    qk_head_dim = q.shape[2]
    q_out = torch.empty(
        num_tokens, num_heads, qk_head_dim,
        device=q.device, dtype=q.dtype
    )
    k_out = torch.empty(
        num_tokens, num_heads, qk_head_dim,
        device=q.device, dtype=q.dtype
    )
    return q_out, k_out
