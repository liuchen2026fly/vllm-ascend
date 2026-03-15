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
Performance patch for Qwen3.5 GatedDeltaNet on Ascend NPU.

Qwen3.5 uses 4 separate input projections:
  in_proj_qkv  (hidden → key_dim*2 + value_dim)  — MergedColumnParallelLinear
  in_proj_z    (hidden → value_dim)               — ColumnParallelLinear
  in_proj_b    (hidden → num_v_heads)             — ColumnParallelLinear
  in_proj_a    (hidden → num_v_heads)             — ColumnParallelLinear

This results in 4 GEMM calls per forward, each reading hidden_states from HBM.
The b/a projections are especially wasteful: num_v_heads is tiny (e.g., 64)
compared to hidden_size (e.g., 4096), giving extremely low compute utilization.

This patch fuses them into 2 projections:
  fused_qkvz   (hidden → key_dim*2 + value_dim*2)  — 1 GEMM
  fused_ba     (hidden → num_v_heads*2)             — 1 GEMM

Benefits:
  - Halves the number of hidden_states HBM reads
  - Eliminates 2 tiny, inefficient GEMM kernels
  - Reduces kernel launch overhead
"""

import logging
import os
import torch
import torch.nn.functional as F
from einops import rearrange

logger = logging.getLogger(__name__)

try:
    from vllm.model_executor.models.qwen3_5 import Qwen3_5GatedDeltaNet
except ImportError:
    Qwen3_5GatedDeltaNet = None

# Import fused RMSNormGated Triton kernel
try:
    from vllm_ascend.ops.triton.fla.fused_rmsnorm_gated import fused_rmsnorm_gated
    _TRITON_RMSNORM_AVAILABLE = True
except ImportError:
    _TRITON_RMSNORM_AVAILABLE = False

# Import fused split/reshape Triton kernel
try:
    from vllm_ascend.ops.triton.fla.fused_qwen3_5_split import fused_qwen3_5_split_reshape
    from vllm.triton_utils import triton
    _TRITON_SPLIT_AVAILABLE = True
except ImportError:
    _TRITON_SPLIT_AVAILABLE = False
    triton = None

# Save original forward for fallback (quantized / LoRA models)
if Qwen3_5GatedDeltaNet is not None:
    _original_qwen3_5_forward = Qwen3_5GatedDeltaNet.forward

# Configurable threshold for RMSNormGated Triton kernel
# Environment variable allows runtime tuning without code changes
# Default: 32 (use Triton for batches <= 32, PyTorch for larger batches)
# Tuning guide: Try 16, 24, 32, 48, 64 and benchmark to find optimal value
_RMSNORM_TRITON_THRESHOLD = int(os.environ.get('QWEN3_5_RMSNORM_THRESHOLD', '32'))


class AscendQwen3_5GatedDeltaNet:
    """Optimized Qwen3.5 GatedDeltaNet forward for Ascend NPU."""

    def _fuse_projections(self):
        """
        Fuse 4 separate projection weights into 2 fused weights.
        Called lazily on first forward. Returns True if fusion succeeded.

        Fusion layout:
          _fused_qkvz_weight = cat([in_proj_qkv.weight, in_proj_z.weight])
          _fused_ba_weight   = cat([in_proj_b.weight,   in_proj_a.weight])

        After GEMM, output is split by recorded dimensions:
          qkvz_out → mixed_qkv[:qkv_dim] + z[qkv_dim:]
          ba_out   → b[:b_dim]            + a[b_dim:]
        """
        if hasattr(self, '_projections_fused'):
            return self._projections_fused

        # Verify all projections have accessible weight tensors
        modules = [self.in_proj_qkv, self.in_proj_z,
                   self.in_proj_b, self.in_proj_a]
        if not all(hasattr(m, 'weight') and m.weight is not None
                   for m in modules):
            logger.warning(
                "[Qwen3.5 Patch] Projection fusion FAILED: "
                "missing weight tensors. Using original 4-GEMM path."
            )
            self._projections_fused = False
            return False

        qkv_w = self.in_proj_qkv.weight
        z_w = self.in_proj_z.weight
        b_w = self.in_proj_b.weight
        a_w = self.in_proj_a.weight

        # Only fuse plain float weights (not quantized int4/int8)
        if not all(w.is_floating_point() for w in [qkv_w, z_w, b_w, a_w]):
            logger.warning(
                "[Qwen3.5 Patch] Projection fusion FAILED: "
                "non-float weights detected (quantized?). "
                "dtypes: qkv=%s, z=%s, b=%s, a=%s. "
                "Using original 4-GEMM path.",
                qkv_w.dtype, z_w.dtype, b_w.dtype, a_w.dtype,
            )
            self._projections_fused = False
            return False

        # Record split dimensions (after TP sharding)
        self._qkv_out_dim = qkv_w.shape[0]
        self._z_out_dim = z_w.shape[0]
        self._b_out_dim = b_w.shape[0]
        self._a_out_dim = a_w.shape[0]

        # Fuse qkv + z weights
        self._fused_qkvz_weight = torch.cat(
            [qkv_w.data, z_w.data], dim=0,
        ).contiguous()

        # Fuse b + a weights
        self._fused_ba_weight = torch.cat(
            [b_w.data, a_w.data], dim=0,
        ).contiguous()

        logger.warning(
            "[Qwen3.5 Patch] Projection fusion SUCCESS: "
            "qkvz=[%d+%d], ba=[%d+%d], dtype=%s. "
            "Triton split=%s, Triton RMSNorm=%s (threshold=%d).",
            self._qkv_out_dim, self._z_out_dim,
            self._b_out_dim, self._a_out_dim,
            qkv_w.dtype,
            _TRITON_SPLIT_AVAILABLE,
            _TRITON_RMSNORM_AVAILABLE,
            _RMSNORM_TRITON_THRESHOLD,
        )
        self._projections_fused = True
        return True

    def forward(
        self,
        hidden_states: torch.Tensor,
        output: torch.Tensor,
    ):
        """
        Optimized forward: 2 GEMMs instead of 4.

        Part 1: Fused projection
          GEMM 1 (qkvz): hidden_states @ fused_qkvz_weight^T → split → mixed_qkv, z
          GEMM 2 (ba):   hidden_states @ fused_ba_weight^T   → split → b, a
        Part 2: Core attention (_forward_core inherited from patch_qwen3_next.py)
        Part 3: RMSNormGated + output projection

        ACLGraph Support:
          This implementation is graph-compatible for decode optimization.
          Enable via: --enforce-eager=False
          The fused kernels (split, RMSNormGated) work seamlessly with ACLGraph.
          Graph capture happens automatically in decode mode for maximum performance.
        """
        # Lazy fusion on first call; fallback if fusion not possible
        if not hasattr(self, '_projections_fused'):
            if not self._fuse_projections():
                return _original_qwen3_5_forward(self, hidden_states, output)
        if not self._projections_fused:
            return _original_qwen3_5_forward(self, hidden_states, output)

        num_tokens = hidden_states.size(0)

        # ============================================================
        # Part 1: Fused Input Projections (2 GEMMs instead of 4)
        # ============================================================

        # GEMM 1: qkvz — fuses in_proj_qkv + in_proj_z
        qkvz_out = F.linear(hidden_states, self._fused_qkvz_weight)

        # GEMM 2: ba — fuses in_proj_b + in_proj_a
        ba_out = F.linear(hidden_states, self._fused_ba_weight)

        # Conditional optimization: use fused Triton kernel for split/reshape
        # Requirements:
        #   1. Triton kernel available
        #   2. Grid size within hardware limit (< 65536)
        use_fused_split = (
            _TRITON_SPLIT_AVAILABLE
            and num_tokens < 65536  # NPU grid size limit
        )

        if use_fused_split:
            # Fused path: single Triton kernel for all splits + reshape
            # Replaces 5 operations: 4 splits + 1 reshape
            num_v_heads_local = self.num_v_heads // self.tp_size
            mixed_qkv, z, b, a = fused_qwen3_5_split_reshape(
                qkvz_out,
                ba_out,
                self._qkv_out_dim,
                self._b_out_dim,
                num_v_heads_local,
                self.head_v_dim,
            )
        else:
            # Python fallback path: separate splits + reshape
            mixed_qkv = qkvz_out[:, :self._qkv_out_dim].contiguous()
            z = qkvz_out[:, self._qkv_out_dim:].contiguous()
            z = z.reshape(z.size(0), -1, self.head_v_dim)
            b = ba_out[:, :self._b_out_dim].contiguous()
            a = ba_out[:, self._b_out_dim:].contiguous()

        # ============================================================
        # Part 2: Core Attention (Custom Op)
        # ============================================================
        core_attn_out = torch.zeros(
            (num_tokens, self.num_v_heads // self.tp_size, self.head_v_dim),
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )

        torch.ops.vllm.gdn_attention_core(
            mixed_qkv,
            b,
            a,
            core_attn_out,
            self.prefix,
        )

        # ============================================================
        # Part 3: Output Projection with Fused RMSNormGated
        # ============================================================
        z_shape_og = z.shape
        num_tokens = core_attn_out.shape[0]

        # Conditional optimization based on batch size:
        # - Large batch (Prefill, num_tokens > threshold): Use PyTorch (better parallelism)
        # - Small batch (Decode, num_tokens <= threshold): Use Triton fused kernel (less overhead)
        # Threshold is configurable via QWEN3_5_RMSNORM_THRESHOLD environment variable
        use_fused_kernel = (
            _TRITON_RMSNORM_AVAILABLE
            and hasattr(self.norm, 'weight')
            and num_tokens <= _RMSNORM_TRITON_THRESHOLD
        )

        if use_fused_kernel:
            # Fused path: RMSNorm + Gating in one kernel
            # Optimized for Decode stage (small batch)
            core_attn_out = fused_rmsnorm_gated(
                core_attn_out,  # shape: (num_tokens, num_v_heads // tp_size, head_v_dim)
                z,               # shape: (num_tokens, num_v_heads // tp_size, head_v_dim)
                self.norm.weight,
                eps=self.norm.eps,
            )
        else:
            # PyTorch path: separate RMSNorm + Gating
            # Optimized for Prefill stage (large batch) or when Triton unavailable
            core_attn_out = core_attn_out.reshape(-1, core_attn_out.shape[-1])
            z = z.reshape(-1, z.shape[-1])
            core_attn_out = self.norm(core_attn_out, z)
            core_attn_out = core_attn_out.reshape(z_shape_og)

        core_attn_out = rearrange(core_attn_out, "... h d -> ... (h d)")
        output[:num_tokens], _ = self.out_proj(core_attn_out)


if Qwen3_5GatedDeltaNet is not None:
    Qwen3_5GatedDeltaNet._fuse_projections = (
        AscendQwen3_5GatedDeltaNet._fuse_projections
    )
    Qwen3_5GatedDeltaNet.forward = AscendQwen3_5GatedDeltaNet.forward
