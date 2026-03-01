#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
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
Fusion pass for MLA partial RoPE + Q/K assembly.

Matches the FX subgraph produced by DeepseekV2Attention.forward:
  q_nope, q_pe = q.split([nope_dim, rope_dim], dim=-1)
  q_pe, k_pe = rotary_emb(positions, q_pe, k_pe)
  q[..., nope_dim:] = q_pe
  k = torch.empty_like(q)
  k[..., :nope_dim] = k_nope
  k[..., nope_dim:] = k_pe

And replaces it with a single fused kernel call:
  q_out, k_out = torch.ops.vllm.mla_partial_rope_assemble(
      q, k_nope, k_pe, cos_sin_cache, positions, nope_dim, rope_dim, True)

Note: FX graph pattern matching for in-place slice_scatter operations is
fragile. If this pass fails to match, the model patch in
patch_mla_rope.py serves as a reliable fallback.
"""

import torch
from torch._inductor.pattern_matcher import PatternMatcherPass
from vllm.config import VllmConfig
from vllm.config.compilation import Range
from vllm.logger import logger

from vllm_ascend.compilation.passes.base_pattern import BasePattern, _registered_patterns
from vllm_ascend.utils import vllm_version_is

if vllm_version_is("0.15.0"):
    from vllm.compilation.vllm_inductor_pass import VllmInductorPass  # type: ignore
else:
    from vllm.compilation.passes.vllm_inductor_pass import VllmInductorPass


class MlaRopeFusionPattern(BasePattern):
    """
    Pattern that matches the MLA split -> RoPE -> assemble subgraph
    and replaces it with a single fused kernel.
    """

    def __init__(self, vllm_config: VllmConfig, nope_dim: int,
                 rope_dim: int, num_heads: int):
        super().__init__(vllm_config)
        self.nope_dim = nope_dim
        self.rope_dim = rope_dim
        self.num_heads = num_heads
        self.qk_head_dim = nope_dim + rope_dim

    def get_inputs(self) -> list[torch.Tensor]:
        T = 5
        max_position_embeddings = 16384
        q = torch.empty(
            T, self.num_heads, self.qk_head_dim,
            dtype=self.dtype, device="npu"
        )
        k_nope = torch.empty(
            T, self.num_heads, self.nope_dim,
            dtype=self.dtype, device="npu"
        )
        k_pe = torch.empty(
            T, 1, self.rope_dim,
            dtype=self.dtype, device="npu"
        )
        cos_sin_cache = torch.empty(
            max_position_embeddings, self.rope_dim,
            dtype=self.dtype, device="npu"
        )
        positions = torch.ones(T, dtype=torch.int64, device="npu")
        return [q, k_nope, k_pe, cos_sin_cache, positions]

    def get_pattern(self):
        def pattern(
            q: torch.Tensor,
            k_nope: torch.Tensor,
            k_pe: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            positions: torch.Tensor,
        ):
            # Split q into nope and rope parts
            q_nope, q_pe = q.split(
                [self.nope_dim, self.rope_dim], dim=-1
            )

            # Apply RoPE to q_pe and k_pe
            q_pe_r, k_pe_r = torch.ops.vllm.npu_rotary_embedding(
                positions, q_pe, k_pe,
                cos_sin_cache,
                self.rope_dim, self.rope_dim, True
            )

            # Assemble q and k using cat (functional equivalent of
            # slice_scatter in the FX graph)
            q_out = torch.cat([q_nope, q_pe_r], dim=-1)
            k_out = torch.cat(
                [k_nope, k_pe_r.expand(-1, k_nope.shape[1], -1)], dim=-1
            )

            return q_out, k_out

        return pattern

    def get_replacement(self):
        def replacement(
            q: torch.Tensor,
            k_nope: torch.Tensor,
            k_pe: torch.Tensor,
            cos_sin_cache: torch.Tensor,
            positions: torch.Tensor,
        ):
            return torch.ops.vllm.mla_partial_rope_assemble(
                q, k_nope, k_pe, cos_sin_cache, positions,
                self.nope_dim, self.rope_dim, False
            )

        return replacement

    def register(self, pm_pass):
        """Override register to use MLA-specific pattern_id."""
        import torch._inductor.pattern_matcher as pm
        import torchair

        pattern_id = (
            f"{self.__class__.__name__}"
            f"_nope{self.nope_dim}_rope{self.rope_dim}"
            f"_heads{self.num_heads}"
        )
        if pattern_id in _registered_patterns:
            return

        pattern_fn = self.get_pattern()
        replacement_fn = self.get_replacement()
        example_inputs = self.get_inputs()

        pm.register_replacement(
            pattern_fn, replacement_fn, example_inputs,
            pm.fwd_only, pm_pass
        )

        torchair.register_replacement(
            search_fn=pattern_fn,
            replace_fn=replacement_fn,
            example_inputs=example_inputs,
            extra_check=self.get_extra_stream_scope_check(),
        )

        _registered_patterns.add(pattern_id)


class MlaRopeFusionPass(VllmInductorPass):
    """
    A fusion pass for MLA partial RoPE + Q/K assembly.

    This pass matches the split -> RoPE -> assemble pattern in
    DeepseekV2Attention (MLA mode) and replaces it with a single
    fused Triton kernel call.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(
            pass_name="mla_rope_fusion_pass"
        )

        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.bfloat16,):
            logger.debug(
                "MLA RoPE fusion not enabled: unsupported dtype %s", dtype
            )
            return

        # Check if model uses MLA
        if not getattr(vllm_config.model_config, 'use_mla', False):
            logger.debug("MLA RoPE fusion not enabled: model does not use MLA")
            return

        hf_config = vllm_config.model_config.hf_text_config
        nope_dim = getattr(hf_config, 'qk_nope_head_dim', None)
        rope_dim = getattr(hf_config, 'qk_rope_head_dim', None)
        num_heads = getattr(hf_config, 'num_attention_heads', None)

        if nope_dim is None or rope_dim is None or num_heads is None:
            logger.debug(
                "MLA RoPE fusion not enabled: missing MLA config "
                "(qk_nope_head_dim=%s, qk_rope_head_dim=%s, "
                "num_attention_heads=%s)",
                nope_dim, rope_dim, num_heads,
            )
            return

        # Adjust num_heads for tensor parallelism
        tp_size = getattr(
            vllm_config.parallel_config, 'tensor_parallel_size', 1
        )
        num_local_heads = num_heads // tp_size

        logger.debug(
            "MLA RoPE fusion enabled: nope_dim=%d, rope_dim=%d, "
            "num_local_heads=%d",
            nope_dim, rope_dim, num_local_heads,
        )

        MlaRopeFusionPattern(
            vllm_config=vllm_config,
            nope_dim=nope_dim,
            rope_dim=rope_dim,
            num_heads=num_local_heads,
        ).register(self.pattern_match_passes)

    def __call__(self, graph: torch.fx.Graph) -> None:  # type: ignore[override]
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Fused %s MLA RoPE patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        return True
