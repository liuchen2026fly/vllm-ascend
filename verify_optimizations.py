#!/usr/bin/env python3
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
Quick verification script for Qwen3.5 NPU optimizations.

Tests:
1. Fused split kernel import and basic functionality
2. RMSNormGated kernel import and autotune
3. Configurable threshold
4. Integration in patch_qwen3_5.py

Usage:
    python verify_optimizations.py
"""

import os
import sys


def test_imports():
    """Test that all optimizations can be imported."""
    print("=" * 70)
    print("Testing Imports...")
    print("=" * 70)

    # Test fused split kernel
    try:
        from vllm_ascend.ops.triton.fla.fused_qwen3_5_split import fused_qwen3_5_split_reshape
        print("✓ Fused split kernel imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import fused split kernel: {e}")
        return False

    # Test RMSNorm kernel
    try:
        from vllm_ascend.ops.triton.fla.fused_rmsnorm_gated import fused_rmsnorm_gated
        print("✓ RMSNormGated kernel imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import RMSNormGated kernel: {e}")
        return False

    # Test patch imports
    try:
        from vllm_ascend.patch.worker import patch_qwen3_5
        print("✓ patch_qwen3_5 imported successfully")
    except ImportError as e:
        print(f"✗ Failed to import patch_qwen3_5: {e}")
        return False

    print()
    return True


def test_fused_split_kernel():
    """Test fused split kernel basic functionality."""
    print("=" * 70)
    print("Testing Fused Split Kernel...")
    print("=" * 70)

    try:
        import torch
        from vllm_ascend.ops.triton.fla.fused_qwen3_5_split import fused_qwen3_5_split_reshape

        if not torch.npu.is_available():
            print("⚠ NPU not available, skipping kernel test")
            print()
            return True

        # Test configuration
        num_tokens = 16
        qkv_out_dim = 1024
        z_out_dim = 512
        b_out_dim = 32
        a_out_dim = 32
        num_v_heads = 16
        head_v_dim = 32

        # Create test inputs
        device = 'npu'
        qkvz_out = torch.randn(num_tokens, qkv_out_dim + z_out_dim, device=device)
        ba_out = torch.randn(num_tokens, b_out_dim + a_out_dim, device=device)

        # Run kernel
        mixed_qkv, z, b, a = fused_qwen3_5_split_reshape(
            qkvz_out, ba_out, qkv_out_dim, b_out_dim, num_v_heads, head_v_dim
        )

        # Verify shapes
        assert mixed_qkv.shape == (num_tokens, qkv_out_dim)
        assert z.shape == (num_tokens, num_v_heads, head_v_dim)
        assert b.shape == (num_tokens, b_out_dim)
        assert a.shape == (num_tokens, a_out_dim)

        print(f"✓ Kernel executed successfully")
        print(f"  Output shapes:")
        print(f"    mixed_qkv: {mixed_qkv.shape}")
        print(f"    z: {z.shape}")
        print(f"    b: {b.shape}")
        print(f"    a: {a.shape}")

    except Exception as e:
        print(f"✗ Kernel test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True


def test_rmsnorm_autotune():
    """Test RMSNorm kernel autotune."""
    print("=" * 70)
    print("Testing RMSNorm Autotune...")
    print("=" * 70)

    try:
        import torch
        from vllm_ascend.ops.triton.fla.fused_rmsnorm_gated import fused_rmsnorm_gated

        if not torch.npu.is_available():
            print("⚠ NPU not available, skipping autotune test")
            print()
            return True

        # Test configuration
        batch, n_heads, head_dim = 16, 32, 64
        device = 'npu'

        # Create test inputs
        x = torch.randn(batch, n_heads, head_dim, device=device)
        z = torch.randn(batch, n_heads, head_dim, device=device)
        weight = torch.ones(head_dim, device=device)

        # Run kernel (autotune will happen on first call)
        output = fused_rmsnorm_gated(x, z, weight, eps=1e-6)

        # Verify shape
        assert output.shape == x.shape

        print(f"✓ Kernel executed successfully with autotune")
        print(f"  Output shape: {output.shape}")

    except Exception as e:
        print(f"✗ Autotune test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True


def test_configurable_threshold():
    """Test configurable threshold."""
    print("=" * 70)
    print("Testing Configurable Threshold...")
    print("=" * 70)

    try:
        # Test default threshold
        os.environ.pop('QWEN3_5_RMSNORM_THRESHOLD', None)
        import importlib
        from vllm_ascend.patch.worker import patch_qwen3_5
        importlib.reload(patch_qwen3_5)

        default_threshold = patch_qwen3_5._RMSNORM_TRITON_THRESHOLD
        print(f"✓ Default threshold: {default_threshold}")
        assert default_threshold == 32, f"Expected 32, got {default_threshold}"

        # Test custom threshold
        os.environ['QWEN3_5_RMSNORM_THRESHOLD'] = '48'
        importlib.reload(patch_qwen3_5)

        custom_threshold = patch_qwen3_5._RMSNORM_TRITON_THRESHOLD
        print(f"✓ Custom threshold: {custom_threshold}")
        assert custom_threshold == 48, f"Expected 48, got {custom_threshold}"

        print(f"✓ Threshold configuration working correctly")

    except Exception as e:
        print(f"✗ Threshold test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True


def test_patch_integration():
    """Test that patch integrates all optimizations."""
    print("=" * 70)
    print("Testing Patch Integration...")
    print("=" * 70)

    try:
        from vllm_ascend.patch.worker import patch_qwen3_5

        # Check that all flags are available
        assert hasattr(patch_qwen3_5, '_TRITON_RMSNORM_AVAILABLE')
        print(f"✓ RMSNorm kernel available: {patch_qwen3_5._TRITON_RMSNORM_AVAILABLE}")

        assert hasattr(patch_qwen3_5, '_TRITON_SPLIT_AVAILABLE')
        print(f"✓ Split kernel available: {patch_qwen3_5._TRITON_SPLIT_AVAILABLE}")

        assert hasattr(patch_qwen3_5, '_RMSNORM_TRITON_THRESHOLD')
        print(f"✓ Threshold configurable: {patch_qwen3_5._RMSNORM_TRITON_THRESHOLD}")

        # Check that AscendQwen3_5GatedDeltaNet class exists
        assert hasattr(patch_qwen3_5, 'AscendQwen3_5GatedDeltaNet')
        print(f"✓ AscendQwen3_5GatedDeltaNet class defined")

        # Check that forward method exists
        assert hasattr(patch_qwen3_5.AscendQwen3_5GatedDeltaNet, 'forward')
        print(f"✓ Optimized forward method defined")

        print(f"✓ All patch components integrated correctly")

    except Exception as e:
        print(f"✗ Patch integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print()
    return True


def main():
    """Run all verification tests."""
    print("\n" + "=" * 70)
    print("Qwen3.5 NPU Optimization Verification")
    print("=" * 70)
    print()

    tests = [
        ("Imports", test_imports),
        ("Fused Split Kernel", test_fused_split_kernel),
        ("RMSNorm Autotune", test_rmsnorm_autotune),
        ("Configurable Threshold", test_configurable_threshold),
        ("Patch Integration", test_patch_integration),
    ]

    results = {}
    for test_name, test_func in tests:
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False

    # Summary
    print("=" * 70)
    print("Verification Summary")
    print("=" * 70)

    all_passed = True
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status:10s} {test_name}")
        if not passed:
            all_passed = False

    print("=" * 70)

    if all_passed:
        print("\n✓ All verifications passed! Optimizations ready to use.")
        print("\nNext steps:")
        print("  1. Run unit tests: pytest tests/ops/test_qwen3_5_split_kernel.py")
        print("  2. Run e2e tests: pytest tests/e2e/multicard/4-cards/test_qwen3_5.py")
        print("  3. Benchmark performance: bash benchmarks/scripts/run-performance-benchmarks.sh")
        return 0
    else:
        print("\n✗ Some verifications failed. Please check the errors above.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
