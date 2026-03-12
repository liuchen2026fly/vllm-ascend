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
Unit tests for fused_qwen3_5_split_reshape kernel.

Tests correctness against Python reference implementation across
various tensor shapes and data types.
"""

import pytest
import torch

from vllm_ascend.ops.triton.fla.fused_qwen3_5_split import fused_qwen3_5_split_reshape


class TestQwen35SplitKernel:
    """Test suite for Qwen3.5 fused split/reshape kernel."""

    @pytest.fixture
    def device(self):
        """Device to run tests on."""
        if torch.npu.is_available():
            return 'npu'
        return 'cpu'

    def reference_implementation(
        self,
        qkvz_out: torch.Tensor,
        ba_out: torch.Tensor,
        qkv_out_dim: int,
        b_out_dim: int,
        num_v_heads: int,
        head_v_dim: int,
    ):
        """
        Python reference implementation matching patch_qwen3_5.py original code.
        """
        # Split qkvz
        mixed_qkv = qkvz_out[:, :qkv_out_dim].contiguous()
        z = qkvz_out[:, qkv_out_dim:].contiguous()
        z = z.reshape(z.size(0), num_v_heads, head_v_dim)

        # Split ba
        b = ba_out[:, :b_out_dim].contiguous()
        a = ba_out[:, b_out_dim:].contiguous()

        return mixed_qkv, z, b, a

    @pytest.mark.parametrize("num_tokens", [1, 8, 16, 32, 64, 128])
    @pytest.mark.parametrize("dtype", [torch.float16, torch.float32])
    def test_correctness_various_batch_sizes(self, device, num_tokens, dtype):
        """Test correctness across different batch sizes and dtypes."""
        if device == 'cpu':
            pytest.skip("Triton kernel requires NPU")

        # Qwen3.5-7B typical dimensions (after TP=4 sharding)
        qkv_out_dim = 4096  # 2 * key_dim + value_dim
        z_out_dim = 2048    # value_dim
        b_out_dim = 64      # num_v_heads / tp_size
        a_out_dim = 64
        num_v_heads = 64
        head_v_dim = z_out_dim // num_v_heads

        # Generate random inputs
        torch.manual_seed(42 + num_tokens)  # Different seed per test
        qkvz_out = torch.randn(num_tokens, qkv_out_dim + z_out_dim, device=device, dtype=dtype)
        ba_out = torch.randn(num_tokens, b_out_dim + a_out_dim, device=device, dtype=dtype)

        # Reference implementation
        mixed_qkv_ref, z_ref, b_ref, a_ref = self.reference_implementation(
            qkvz_out, ba_out, qkv_out_dim, b_out_dim, num_v_heads, head_v_dim
        )

        # Triton implementation
        mixed_qkv_tri, z_tri, b_tri, a_tri = fused_qwen3_5_split_reshape(
            qkvz_out, ba_out, qkv_out_dim, b_out_dim, num_v_heads, head_v_dim
        )

        # Verify correctness
        rtol = 1e-5 if dtype == torch.float32 else 1e-3
        atol = 1e-5 if dtype == torch.float32 else 1e-3

        torch.testing.assert_close(mixed_qkv_tri, mixed_qkv_ref, rtol=rtol, atol=atol,
                                   msg=f"mixed_qkv mismatch (num_tokens={num_tokens}, dtype={dtype})")
        torch.testing.assert_close(z_tri, z_ref, rtol=rtol, atol=atol,
                                   msg=f"z mismatch (num_tokens={num_tokens}, dtype={dtype})")
        torch.testing.assert_close(b_tri, b_ref, rtol=rtol, atol=atol,
                                   msg=f"b mismatch (num_tokens={num_tokens}, dtype={dtype})")
        torch.testing.assert_close(a_tri, a_ref, rtol=rtol, atol=atol,
                                   msg=f"a mismatch (num_tokens={num_tokens}, dtype={dtype})")

    @pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
    def test_correctness_various_tp_sizes(self, device, tp_size):
        """Test correctness with different tensor parallel sizes."""
        if device == 'cpu':
            pytest.skip("Triton kernel requires NPU")

        # Qwen3.5-7B base dimensions
        num_tokens = 16
        dtype = torch.float16

        # Dimensions scale with TP
        total_key_dim = 2048
        total_value_dim = 4096
        total_num_v_heads = 256

        # After TP sharding
        qkv_out_dim = (2 * total_key_dim + total_value_dim) // tp_size
        z_out_dim = total_value_dim // tp_size
        num_v_heads = total_num_v_heads // tp_size
        b_out_dim = num_v_heads
        a_out_dim = num_v_heads
        head_v_dim = z_out_dim // num_v_heads

        # Generate inputs
        torch.manual_seed(42 + tp_size)
        qkvz_out = torch.randn(num_tokens, qkv_out_dim + z_out_dim, device=device, dtype=dtype)
        ba_out = torch.randn(num_tokens, b_out_dim + a_out_dim, device=device, dtype=dtype)

        # Reference
        mixed_qkv_ref, z_ref, b_ref, a_ref = self.reference_implementation(
            qkvz_out, ba_out, qkv_out_dim, b_out_dim, num_v_heads, head_v_dim
        )

        # Triton
        mixed_qkv_tri, z_tri, b_tri, a_tri = fused_qwen3_5_split_reshape(
            qkvz_out, ba_out, qkv_out_dim, b_out_dim, num_v_heads, head_v_dim
        )

        # Verify
        torch.testing.assert_close(mixed_qkv_tri, mixed_qkv_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(z_tri, z_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(b_tri, b_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(a_tri, a_ref, rtol=1e-3, atol=1e-3)

    def test_output_shapes(self, device):
        """Test that output shapes are correct."""
        if device == 'cpu':
            pytest.skip("Triton kernel requires NPU")

        num_tokens = 16
        qkv_out_dim = 4096
        z_out_dim = 2048
        b_out_dim = 64
        a_out_dim = 64
        num_v_heads = 64
        head_v_dim = 32

        qkvz_out = torch.randn(num_tokens, qkv_out_dim + z_out_dim, device=device)
        ba_out = torch.randn(num_tokens, b_out_dim + a_out_dim, device=device)

        mixed_qkv, z, b, a = fused_qwen3_5_split_reshape(
            qkvz_out, ba_out, qkv_out_dim, b_out_dim, num_v_heads, head_v_dim
        )

        assert mixed_qkv.shape == (num_tokens, qkv_out_dim), f"mixed_qkv shape mismatch: {mixed_qkv.shape}"
        assert z.shape == (num_tokens, num_v_heads, head_v_dim), f"z shape mismatch: {z.shape}"
        assert b.shape == (num_tokens, b_out_dim), f"b shape mismatch: {b.shape}"
        assert a.shape == (num_tokens, a_out_dim), f"a shape mismatch: {a.shape}"

    def test_contiguity(self, device):
        """Test that outputs are contiguous."""
        if device == 'cpu':
            pytest.skip("Triton kernel requires NPU")

        num_tokens = 16
        qkv_out_dim = 1024
        z_out_dim = 512
        b_out_dim = 32
        a_out_dim = 32
        num_v_heads = 16
        head_v_dim = 32

        qkvz_out = torch.randn(num_tokens, qkv_out_dim + z_out_dim, device=device)
        ba_out = torch.randn(num_tokens, b_out_dim + a_out_dim, device=device)

        mixed_qkv, z, b, a = fused_qwen3_5_split_reshape(
            qkvz_out, ba_out, qkv_out_dim, b_out_dim, num_v_heads, head_v_dim
        )

        assert mixed_qkv.is_contiguous(), "mixed_qkv is not contiguous"
        assert z.is_contiguous(), "z is not contiguous"
        assert b.is_contiguous(), "b is not contiguous"
        assert a.is_contiguous(), "a is not contiguous"

    def test_edge_case_single_token(self, device):
        """Test edge case: single token (decode mode)."""
        if device == 'cpu':
            pytest.skip("Triton kernel requires NPU")

        num_tokens = 1
        qkv_out_dim = 2048
        z_out_dim = 1024
        b_out_dim = 32
        a_out_dim = 32
        num_v_heads = 32
        head_v_dim = 32

        qkvz_out = torch.randn(num_tokens, qkv_out_dim + z_out_dim, device=device, dtype=torch.float16)
        ba_out = torch.randn(num_tokens, b_out_dim + a_out_dim, device=device, dtype=torch.float16)

        # Reference
        mixed_qkv_ref, z_ref, b_ref, a_ref = self.reference_implementation(
            qkvz_out, ba_out, qkv_out_dim, b_out_dim, num_v_heads, head_v_dim
        )

        # Triton
        mixed_qkv_tri, z_tri, b_tri, a_tri = fused_qwen3_5_split_reshape(
            qkvz_out, ba_out, qkv_out_dim, b_out_dim, num_v_heads, head_v_dim
        )

        torch.testing.assert_close(mixed_qkv_tri, mixed_qkv_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(z_tri, z_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(b_tri, b_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(a_tri, a_ref, rtol=1e-3, atol=1e-3)

    def test_edge_case_large_batch(self, device):
        """Test edge case: large batch (prefill mode)."""
        if device == 'cpu':
            pytest.skip("Triton kernel requires NPU")

        num_tokens = 1024  # Large prefill batch
        qkv_out_dim = 1024
        z_out_dim = 512
        b_out_dim = 16
        a_out_dim = 16
        num_v_heads = 16
        head_v_dim = 32

        qkvz_out = torch.randn(num_tokens, qkv_out_dim + z_out_dim, device=device, dtype=torch.float16)
        ba_out = torch.randn(num_tokens, b_out_dim + a_out_dim, device=device, dtype=torch.float16)

        # Reference
        mixed_qkv_ref, z_ref, b_ref, a_ref = self.reference_implementation(
            qkvz_out, ba_out, qkv_out_dim, b_out_dim, num_v_heads, head_v_dim
        )

        # Triton
        mixed_qkv_tri, z_tri, b_tri, a_tri = fused_qwen3_5_split_reshape(
            qkvz_out, ba_out, qkv_out_dim, b_out_dim, num_v_heads, head_v_dim
        )

        torch.testing.assert_close(mixed_qkv_tri, mixed_qkv_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(z_tri, z_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(b_tri, b_ref, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(a_tri, a_ref, rtol=1e-3, atol=1e-3)

    def test_validation_batch_size_mismatch(self, device):
        """Test that batch size mismatch raises assertion."""
        if device == 'cpu':
            pytest.skip("Triton kernel requires NPU")

        qkvz_out = torch.randn(16, 1024, device=device)
        ba_out = torch.randn(8, 128, device=device)  # Different batch size

        with pytest.raises(AssertionError, match="Batch size mismatch"):
            fused_qwen3_5_split_reshape(qkvz_out, ba_out, 512, 64, 16, 32)

    def test_validation_dimension_mismatch(self, device):
        """Test that dimension mismatches raise assertions."""
        if device == 'cpu':
            pytest.skip("Triton kernel requires NPU")

        qkvz_out = torch.randn(16, 1024, device=device)
        ba_out = torch.randn(16, 128, device=device)

        # qkv_out_dim exceeds tensor dimension
        with pytest.raises(AssertionError):
            fused_qwen3_5_split_reshape(qkvz_out, ba_out, 2000, 64, 16, 32)

        # z reshape dimension mismatch
        with pytest.raises(AssertionError, match="z dimension mismatch"):
            fused_qwen3_5_split_reshape(qkvz_out, ba_out, 512, 64, 16, 64)  # Wrong head_v_dim


if __name__ == "__main__":
    # Run basic smoke test
    test_suite = TestQwen35SplitKernel()
    device = 'npu' if torch.npu.is_available() else 'cpu'

    if device == 'npu':
        print("Running Qwen3.5 split kernel tests on NPU...")
        test_suite.test_correctness_various_batch_sizes(device, num_tokens=32, dtype=torch.float16)
        test_suite.test_output_shapes(device)
        test_suite.test_contiguity(device)
        test_suite.test_edge_case_single_token(device)
        print("✓ All tests passed!")
    else:
        print("⚠ NPU not available, skipping tests")
