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
Model patch for MLA partial RoPE + Q/K assembly fusion.

Replaces the 5-7 kernel launch sequence in DeepseekV2Attention.forward
(split → RoPE → slice_scatter → empty → slice_scatter × 2) with a single
fused Triton kernel call via torch.ops.vllm.mla_partial_rope_assemble.

This is the primary enablement path; the fusion pass in
mla_rope_fusion_pass.py is a complementary approach that works at the
FX graph level.
"""

import torch
from torch import nn
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Attention


class AscendDeepseekV2Attention(nn.Module):

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        llama_4_scaling: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.q_lora_rank is not None:
            q = self.q_a_proj(hidden_states)[0]
            q = self.q_a_layernorm(q)
            q = self.q_b_proj(q)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )
        else:
            q = self.q_proj(hidden_states)[0].view(
                -1, self.num_local_heads, self.qk_head_dim
            )

        latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
        kv_a, _ = latent_cache.split(
            [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
        )
        latent_cache = latent_cache.unsqueeze(1)
        kv_a = self.kv_a_layernorm(kv_a)
        kv = self.kv_b_proj(kv_a)[0]
        kv = kv.view(
            -1, self.num_local_heads,
            self.qk_nope_head_dim + self.v_head_dim
        )
        k_nope, v = kv.split(
            [self.qk_nope_head_dim, self.v_head_dim], dim=-1
        )
        k_pe = latent_cache[:, :, self.kv_lora_rank:]

        # --- Fused MLA RoPE + Q/K assembly ---
        # Replaces:
        #   q_nope, q_pe = q.split([nope, rope], dim=-1)
        #   q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        #   q[..., nope:] = q_pe
        #   k = torch.empty_like(q)
        #   k[..., :nope] = k_nope
        #   k[..., nope:] = k_pe
        #
        # Note: DeepSeek uses is_neox_style=False in get_rope(), meaning
        # the data is in interleaved layout. The kernel reads interleaved
        # and writes in neox layout, matching the original rotary_emb's
        # shuffle+neox behavior.
        q, k = torch.ops.vllm.mla_partial_rope_assemble(
            q, k_nope, k_pe,
            self.rotary_emb.cos_sin_cache,
            positions,
            self.qk_nope_head_dim,
            self.qk_rope_head_dim,
            False,  # is_neox_style=False (DeepSeek style: read interleaved, write neox)
        )

        # Apply llama 4 scaling if provided
        if llama_4_scaling is not None:
            q = q * llama_4_scaling

        # padding value to qk_head_dim for alignment
        v = torch.nn.functional.pad(
            v, [0, self.qk_head_dim - self.v_head_dim], value=0
        ).view(-1, self.num_local_heads * self.qk_head_dim)
        attn_output = self.attn(q, k, v)
        attn_output = attn_output.view(
            -1, self.num_local_heads, self.qk_head_dim
        )[..., :self.v_head_dim].reshape(
            -1, self.num_local_heads * self.v_head_dim
        )
        output, _ = self.o_proj(attn_output)
        return output


DeepseekV2Attention.forward = AscendDeepseekV2Attention.forward  # type: ignore[assignment]
