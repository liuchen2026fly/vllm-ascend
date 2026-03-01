#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

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


class MulsAddPattern(BasePattern):
    """
    Pattern that matches an element-wise mul + add sequence:
        tmp = x * scale
        out = tmp + y
    and replaces it with a call to the muls_add_triton kernel.
    """

    def __init__(self, vllm_config: VllmConfig, scale: float = 1.0):
        super().__init__(vllm_config)
        self.scale = scale

    def get_inputs(self) -> list[torch.Tensor]:
        """
        Generate example inputs for the MulsAddPattern.

        The exact shapes are not important for pattern matching; they only
        provide meta information for the pattern matcher.
        """
        x = torch.randn(2, 2048, device="npu", dtype=self.dtype)
        y = torch.randn(2, 2048, device="npu", dtype=self.dtype)
        # Only tensor inputs are needed here. The scalar scale is stored on the
        # pattern instance (self.scale) instead of being passed as an input.
        return [x, y]

    def get_pattern(self):
        def pattern(x: torch.Tensor, y: torch.Tensor):
            """
            Pattern for element-wise x * scale + y.
            """
            tmp = x * self.scale
            out = tmp + y
            return out

        return pattern

    def get_replacement(self):
        def replacement(x: torch.Tensor, y: torch.Tensor):
            """
            Replacement that calls the muls_add_triton kernel using the
            class-level scalar self.scale.
            """
            return torch.ops.vllm.muls_add(x, y, self.scale)

        return replacement

    def register(self, pm_pass):
        """Override base register to use scale-based pattern_id instead of
        eps-based, so multiple MulsAddPattern instances with different scales
        can coexist."""
        import torch._inductor.pattern_matcher as pm
        import torchair

        pattern_id = f"{self.__class__.__name__}_scale_{self.scale}"
        if pattern_id in _registered_patterns:
            return

        pattern_fn = self.get_pattern()
        replacement_fn = self.get_replacement()
        example_inputs = self.get_inputs()

        pm.register_replacement(
            pattern_fn, replacement_fn, example_inputs, pm.fwd_only, pm_pass
        )

        torchair.register_replacement(
            search_fn=pattern_fn,
            replace_fn=replacement_fn,
            example_inputs=example_inputs,
            extra_check=self.get_extra_stream_scope_check(),
        )

        _registered_patterns.add(pattern_id)


class MulsAddFusionPass(VllmInductorPass):
    """
    A fusion pass that replaces simple element-wise x * scale + y patterns
    with the Triton-based muls_add_triton kernel on Ascend.
    """

    def __init__(self, vllm_config: VllmConfig):
        super().__init__(vllm_config)
        self.pattern_match_passes: PatternMatcherPass = PatternMatcherPass(pass_name="muls_add_fusion_pass")

        # For now we enable this pass for all floating-point dtypes that the
        # model is configured to use.
        dtype = vllm_config.model_config.dtype
        if dtype not in (torch.float16, torch.bfloat16, torch.float32):
            logger.debug("MulsAdd fusion not enabled: unsupported dtype %s", dtype)
            return

        # Register default scale=1.0 pattern
        MulsAddPattern(vllm_config, scale=1.0).register(self.pattern_match_passes)

        # Register routed_scaling_factor from model config (e.g. GLM5 uses 2.5)
        routed_scaling_factor = getattr(
            vllm_config.model_config.hf_text_config,
            'routed_scaling_factor', None
        )
        if routed_scaling_factor is not None and routed_scaling_factor != 1.0:
            MulsAddPattern(vllm_config, scale=routed_scaling_factor).register(
                self.pattern_match_passes
            )

    def __call__(self, graph: torch.fx.Graph) -> None:  # type: ignore[override]
        self.begin()
        self.matched_count = self.pattern_match_passes.apply(graph)
        logger.debug("Fused %s muls_add patterns", self.matched_count)
        self.end_and_log()

    def is_applicable_for_range(self, compile_range: Range) -> bool:
        """
        Check if the pass is applicable for the current configuration.

        For now, muls_add fusion is always allowed for the selected ranges.
        This hook exists so that we can add more fine-grained range control
        in the future if needed.
        """
        return True
