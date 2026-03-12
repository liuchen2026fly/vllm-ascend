"""
KvRmsNormRopeCache — NPU Triton Ascend 通用实现

支持两种 method 模式:
  V1: kv=[V|K] 拼接, K→RoPE, V→RMSNorm       (如 D=576=512+64)
  V2: kv=K 单独, V 单独输入, K→RMSNorm+RoPE   (如 D=192, V=128)

自动从输入 shape 推导 D_RMS / D_ROPE / method mode.

NPU Hard constraints:
  - Grid <= get_vectorcore_num()
  - tl.range core-internal loop
  - UB < 192KB per iteration
  - No uint64 / float64 / chained booleans
  - float32 intermediate, store original dtype
"""

import torch
import triton
import triton.language as tl

try:
    from vllm_ascend.ops.triton.triton_utils import get_vectorcore_num
except ImportError:
    def get_vectorcore_num():
        return 20

UB_LIMIT = 192 * 1024  # 192 KB


# ===================================================================
# METHOD V1 Kernel: kv = [V(D_RMS) | K(D_ROPE)]
#   K → RoPE → k_cache
#   V → RMSNorm → v_cache
# ===================================================================
@triton.jit(do_not_specialize=["num_tokens"])
def _kv_rnrc_v1_kernel(
    kv_ptr,    kv_stride,
    gamma_ptr,
    cos_ptr,   cos_stride,
    sin_ptr,   sin_stride,
    index_ptr,
    k_cache_ptr, v_cache_ptr,
    k_out_ptr,   v_out_ptr,
    num_tokens,
    D_RMS:           tl.constexpr,
    D_ROPE:          tl.constexpr,
    D_ROPE_HALF:     tl.constexpr,
    PAD_D_RMS:       tl.constexpr,
    PAD_D_ROPE_HALF: tl.constexpr,
    EPS:             tl.constexpr,
    IS_OUTPUT_KV:    tl.constexpr,
    K_CACHE_DIM:     tl.constexpr,   # k_cache per-slot dim (= D_ROPE)
    V_CACHE_DIM:     tl.constexpr,   # v_cache per-slot dim (= D_RMS)
):
    pid = tl.program_id(0).to(tl.int64)
    num_cores = tl.num_programs(0)

    h = tl.arange(0, PAD_D_ROPE_HALF)
    h_mask = h < D_ROPE_HALF
    d = tl.arange(0, PAD_D_RMS)
    d_mask = d < D_RMS

    gamma = tl.load(gamma_ptr + d, mask=d_mask, other=0.0).to(tl.float32)

    for token_idx in tl.range(pid, num_tokens, num_cores):
        kv_base = token_idx * kv_stride

        # ---------- RoPE on K (interleaved complex) ----------
        k_base = kv_base + D_RMS
        k_real = tl.load(kv_ptr + k_base + h * 2,
                         mask=h_mask, other=0.0).to(tl.float32)
        k_imag = tl.load(kv_ptr + k_base + h * 2 + 1,
                         mask=h_mask, other=0.0).to(tl.float32)

        freq_base = token_idx * cos_stride
        cos_lo = tl.load(cos_ptr + freq_base + h,
                         mask=h_mask, other=0.0).to(tl.float32)
        cos_hi = tl.load(cos_ptr + freq_base + D_ROPE_HALF + h,
                         mask=h_mask, other=0.0).to(tl.float32)
        sin_lo = tl.load(sin_ptr + freq_base + h,
                         mask=h_mask, other=0.0).to(tl.float32)
        sin_hi = tl.load(sin_ptr + freq_base + D_ROPE_HALF + h,
                         mask=h_mask, other=0.0).to(tl.float32)

        rope_real = k_real * cos_lo - k_imag * sin_lo
        rope_imag = k_imag * cos_hi + k_real * sin_hi

        # ---------- RMSNorm on V ----------
        v = tl.load(kv_ptr + kv_base + d,
                    mask=d_mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(v * v) / D_RMS
        inv_rms = tl.rsqrt(mean_sq + EPS)
        v_norm = v * inv_rms * gamma

        # ---------- PA Scatter ----------
        pa_slot = tl.load(index_ptr + token_idx).to(tl.int32)
        valid = pa_slot >= 0
        safe_slot = tl.maximum(pa_slot, 0).to(tl.int64)
        out_dtype = kv_ptr.dtype.element_ty

        k_dst = safe_slot * K_CACHE_DIM
        tl.store(k_cache_ptr + k_dst + h,
                 rope_real.to(out_dtype), mask=h_mask & valid)
        tl.store(k_cache_ptr + k_dst + D_ROPE_HALF + h,
                 rope_imag.to(out_dtype), mask=h_mask & valid)

        v_dst = safe_slot * V_CACHE_DIM
        tl.store(v_cache_ptr + v_dst + d,
                 v_norm.to(out_dtype), mask=d_mask & valid)

        if IS_OUTPUT_KV:
            tl.store(k_out_ptr + token_idx * D_ROPE + h,
                     rope_real.to(out_dtype), mask=h_mask)
            tl.store(k_out_ptr + token_idx * D_ROPE + D_ROPE_HALF + h,
                     rope_imag.to(out_dtype), mask=h_mask)
            tl.store(v_out_ptr + token_idx * D_RMS + d,
                     v_norm.to(out_dtype), mask=d_mask)


# ===================================================================
# METHOD V2 Kernel: kv = K(D_RMS), v_input = V(D_V) separate
#   K → RMSNorm(D_RMS) → first D_ROPE dims RoPE → k_cache
#   V → passthrough → v_cache
# ===================================================================
@triton.jit(do_not_specialize=["num_tokens"])
def _kv_rnrc_v2_kernel(
    kv_ptr,     kv_stride,          # K data [num_tokens, D_RMS]
    v_ptr,      v_stride,           # V data [num_tokens, D_V]
    gamma_ptr,
    cos_ptr,    cos_stride,
    sin_ptr,    sin_stride,
    index_ptr,
    k_cache_ptr, v_cache_ptr,
    k_out_ptr,   v_out_ptr,
    num_tokens,
    D_RMS:           tl.constexpr,  # K dim (e.g. 192)
    D_V:             tl.constexpr,  # V dim (e.g. 128)
    D_ROPE:          tl.constexpr,  # 64
    D_ROPE_HALF:     tl.constexpr,  # 32
    PAD_D_RMS:       tl.constexpr,
    PAD_D_V:         tl.constexpr,
    PAD_D_ROPE_HALF: tl.constexpr,
    EPS:             tl.constexpr,
    IS_OUTPUT_KV:    tl.constexpr,
    K_CACHE_DIM:     tl.constexpr,  # = D_RMS
    V_CACHE_DIM:     tl.constexpr,  # = D_V
):
    pid = tl.program_id(0).to(tl.int64)
    num_cores = tl.num_programs(0)

    dk = tl.arange(0, PAD_D_RMS)
    dk_mask = dk < D_RMS
    dv = tl.arange(0, PAD_D_V)
    dv_mask = dv < D_V
    h = tl.arange(0, PAD_D_ROPE_HALF)
    h_mask = h < D_ROPE_HALF

    gamma = tl.load(gamma_ptr + dk, mask=dk_mask, other=0.0).to(tl.float32)

    for token_idx in tl.range(pid, num_tokens, num_cores):
        # ---------- RMSNorm on K (full D_RMS) ----------
        k = tl.load(kv_ptr + token_idx * kv_stride + dk,
                    mask=dk_mask, other=0.0).to(tl.float32)
        mean_sq = tl.sum(k * k) / D_RMS
        inv_rms = tl.rsqrt(mean_sq + EPS)
        k_norm = k * inv_rms * gamma                    # [D_RMS]

        # ---------- RoPE on first D_ROPE dims of k_norm (interleaved) ----------
        # Extract real/imag from first D_ROPE elements of k_norm
        k_real = tl.load(kv_ptr + token_idx * kv_stride + h * 2,
                         mask=h_mask, other=0.0).to(tl.float32)
        k_imag = tl.load(kv_ptr + token_idx * kv_stride + h * 2 + 1,
                         mask=h_mask, other=0.0).to(tl.float32)
        # Apply RMSNorm to these elements too (they're part of k_norm)
        k_real_norm = k_real * inv_rms
        k_imag_norm = k_imag * inv_rms
        # Apply gamma to real/imag portions
        gamma_real = tl.load(gamma_ptr + h * 2,
                             mask=h_mask, other=0.0).to(tl.float32)
        gamma_imag = tl.load(gamma_ptr + h * 2 + 1,
                             mask=h_mask, other=0.0).to(tl.float32)
        k_real_norm = k_real_norm * gamma_real
        k_imag_norm = k_imag_norm * gamma_imag

        freq_base = token_idx * cos_stride
        cos_lo = tl.load(cos_ptr + freq_base + h,
                         mask=h_mask, other=0.0).to(tl.float32)
        cos_hi = tl.load(cos_ptr + freq_base + D_ROPE_HALF + h,
                         mask=h_mask, other=0.0).to(tl.float32)
        sin_lo = tl.load(sin_ptr + freq_base + h,
                         mask=h_mask, other=0.0).to(tl.float32)
        sin_hi = tl.load(sin_ptr + freq_base + D_ROPE_HALF + h,
                         mask=h_mask, other=0.0).to(tl.float32)

        rope_real = k_real_norm * cos_lo - k_imag_norm * sin_lo
        rope_imag = k_imag_norm * cos_hi + k_real_norm * sin_hi

        # Write rotated values back into k_norm's first D_ROPE positions
        # k_norm is [D_RMS] in fp32; overwrite first D_ROPE with rotated result
        # We'll store k_norm contiguously, then overwrite the rope portion
        # Since Triton doesn't support in-place update on a local vector,
        # we store in two passes: rope portion + remaining portion

        out_dtype = kv_ptr.dtype.element_ty
        pa_slot = tl.load(index_ptr + token_idx).to(tl.int32)
        valid = pa_slot >= 0
        safe_slot = tl.maximum(pa_slot, 0).to(tl.int64)

        # Store k_cache: first write the full k_norm, then overwrite rope portion
        k_dst = safe_slot * K_CACHE_DIM
        # Full D_RMS (non-rope portion)
        tl.store(k_cache_ptr + k_dst + dk,
                 k_norm.to(out_dtype), mask=dk_mask & valid)
        # Overwrite first D_ROPE dims with rotated values
        tl.store(k_cache_ptr + k_dst + h * 2,
                 rope_real.to(out_dtype), mask=h_mask & valid)
        tl.store(k_cache_ptr + k_dst + h * 2 + 1,
                 rope_imag.to(out_dtype), mask=h_mask & valid)

        # V passthrough → v_cache
        v_data = tl.load(v_ptr + token_idx * v_stride + dv,
                         mask=dv_mask, other=0.0)
        v_dst = safe_slot * V_CACHE_DIM
        tl.store(v_cache_ptr + v_dst + dv,
                 v_data, mask=dv_mask & valid)

        if IS_OUTPUT_KV:
            # k_out: write full D_RMS with rope applied
            tl.store(k_out_ptr + token_idx * D_RMS + dk,
                     k_norm.to(out_dtype), mask=dk_mask)
            tl.store(k_out_ptr + token_idx * D_RMS + h * 2,
                     rope_real.to(out_dtype), mask=h_mask)
            tl.store(k_out_ptr + token_idx * D_RMS + h * 2 + 1,
                     rope_imag.to(out_dtype), mask=h_mask)
            tl.store(v_out_ptr + token_idx * D_V + dv,
                     v_data, mask=dv_mask)


# ===================================================================
# Unified Wrapper
# ===================================================================
def kv_rms_norm_rope_cache(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    index: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    v_input: torch.Tensor = None,   # METHOD_V2: separate V
    epsilon: float = 1e-6,
    is_output_kv: bool = True,
) -> tuple:
    """
    Unified entry supporting both METHOD_V1 and METHOD_V2.

    Shape inference:
      - D_ROPE = cos.shape[-1]
      - D_RMS  = gamma.shape[0]
      - V1 if kv.shape[-1] == D_RMS + D_ROPE  (kv = [V | K] concatenated)
      - V2 if kv.shape[-1] == D_RMS            (kv = K only, v_input required)
    """
    # ---- Derive dimensions from input shapes ----
    D_ROPE = cos.shape[-1]
    D_ROPE_HALF = D_ROPE // 2
    D_RMS = gamma.shape[0]
    D_kv_last = kv.shape[-1]

    is_v2 = (v_input is not None)
    if not is_v2:
        assert D_kv_last == D_RMS + D_ROPE, (
            f"V1 mode: kv last dim {D_kv_last} != D_RMS({D_RMS}) + D_ROPE({D_ROPE})")
        D_TOTAL = D_RMS + D_ROPE
    else:
        assert D_kv_last == D_RMS, (
            f"V2 mode: kv last dim {D_kv_last} != D_RMS({D_RMS})")
        D_TOTAL = D_RMS

    # ---- Flatten to 2-D ----
    num_tokens = kv.numel() // D_TOTAL
    kv_flat = kv.reshape(num_tokens, D_TOTAL).contiguous()
    cos_flat = cos.reshape(num_tokens, D_ROPE).contiguous()
    sin_flat = sin.reshape(num_tokens, D_ROPE).contiguous()

    # ---- Pad to power-of-2 ----
    PAD_D_RMS = triton.next_power_of_2(D_RMS)
    PAD_D_ROPE_HALF = triton.next_power_of_2(D_ROPE_HALF)

    # ---- UB overflow guard ----
    ub_estimate = PAD_D_RMS * 4 * 3 + PAD_D_ROPE_HALF * 4 * 8
    if is_v2 and v_input is not None:
        D_V = v_input.shape[-1]
        PAD_D_V = triton.next_power_of_2(D_V)
        ub_estimate += PAD_D_V * 4 * 2
    assert ub_estimate <= UB_LIMIT, (
        f"UB overflow: estimated {ub_estimate} bytes > {UB_LIMIT}. "
        f"Need sub-blocking for D_RMS={D_RMS}")

    # ---- Cache dims (per-slot element count) ----
    K_CACHE_DIM = k_cache.shape[-1]
    V_CACHE_DIM = v_cache.shape[-1]

    # ---- Grid ----
    num_vectorcore = get_vectorcore_num()
    grid = (min(num_tokens, num_vectorcore),)

    if not is_v2:
        # ============ METHOD V1 ============
        k_out = torch.empty(num_tokens, D_ROPE, dtype=kv.dtype,
                            device=kv.device) if is_output_kv else kv.new_empty(0)
        v_out = torch.empty(num_tokens, D_RMS, dtype=kv.dtype,
                            device=kv.device) if is_output_kv else kv.new_empty(0)

        _kv_rnrc_v1_kernel[grid](
            kv_flat, kv_flat.stride(0),
            gamma,
            cos_flat, cos_flat.stride(0),
            sin_flat, sin_flat.stride(0),
            index,
            k_cache, v_cache,
            k_out, v_out,
            num_tokens,
            D_RMS=D_RMS,
            D_ROPE=D_ROPE,
            D_ROPE_HALF=D_ROPE_HALF,
            PAD_D_RMS=PAD_D_RMS,
            PAD_D_ROPE_HALF=PAD_D_ROPE_HALF,
            EPS=epsilon,
            IS_OUTPUT_KV=is_output_kv,
            K_CACHE_DIM=K_CACHE_DIM,
            V_CACHE_DIM=V_CACHE_DIM,
            multibuffer=True,
        )
    else:
        # ============ METHOD V2 ============
        D_V = v_input.shape[-1]
        PAD_D_V = triton.next_power_of_2(D_V)
        v_flat = v_input.reshape(num_tokens, D_V).contiguous()

        k_out = torch.empty(num_tokens, D_RMS, dtype=kv.dtype,
                            device=kv.device) if is_output_kv else kv.new_empty(0)
        v_out = torch.empty(num_tokens, D_V, dtype=kv.dtype,
                            device=kv.device) if is_output_kv else kv.new_empty(0)

        _kv_rnrc_v2_kernel[grid](
            kv_flat, kv_flat.stride(0),
            v_flat,  v_flat.stride(0),
            gamma,
            cos_flat, cos_flat.stride(0),
            sin_flat, sin_flat.stride(0),
            index,
            k_cache, v_cache,
            k_out, v_out,
            num_tokens,
            D_RMS=D_RMS,
            D_V=D_V,
            D_ROPE=D_ROPE,
            D_ROPE_HALF=D_ROPE_HALF,
            PAD_D_RMS=PAD_D_RMS,
            PAD_D_V=PAD_D_V,
            PAD_D_ROPE_HALF=PAD_D_ROPE_HALF,
            EPS=epsilon,
            IS_OUTPUT_KV=is_output_kv,
            K_CACHE_DIM=K_CACHE_DIM,
            V_CACHE_DIM=V_CACHE_DIM,
            multibuffer=True,
        )

    return k_out, v_out, k_cache, v_cache


# ===================================================================
# NPU-compatible Wrapper (drop-in for torch_npu.npu_kv_rmsnorm_rope_cache)
# ===================================================================
def npu_kv_rmsnorm_rope_cache(
    kv: torch.Tensor,
    gamma: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    slots: torch.Tensor,
    k_pe_cache: torch.Tensor,
    k_nope_cache: torch.Tensor,
    epsilon: float = 1e-6,
    cache_mode: str = "PA",
    is_output_kv: bool = False,
) -> tuple:
    """
    Drop-in replacement for torch_npu.npu_kv_rmsnorm_rope_cache.

    Fused kernel: RMSNorm + RoPE + PA Cache Write.

    For MLA V1 mode (auto-detected when kv.shape[-1] == D_RMS + D_ROPE):
      kv = [compressed_kv(kv_lora_rank) | rope_k(qk_rope_head_dim)]
      - compressed_kv -> RMSNorm -> k_nope output + k_nope_cache write
      - rope_k        -> RoPE    -> k_pe   output + k_pe_cache write

    Args:
        kv: Input tensor [B, N, S, kv_lora_rank + qk_rope_head_dim]
            (or any shape, last dim = D)
        gamma: RMSNorm weight [kv_lora_rank]
        cos: RoPE cos embeddings [B, ...., qk_rope_head_dim]
        sin: RoPE sin embeddings [B, ...., qk_rope_head_dim]
        slots: PA slot indices [num_tokens] (int64)
        k_pe_cache: Cache tensor for RoPE'd K (corresponds to kv_cache[1])
        k_nope_cache: Cache tensor for RMSNorm'd compressed_kv
            (corresponds to kv_cache[0])
        epsilon: RMSNorm epsilon (default 1e-6)
        cache_mode: "PA" supported; "PA_NZ" raises NotImplementedError
        is_output_kv: False = decode mode, True = prefill mode

    Returns:
        Decode  (is_output_kv=False):
            (k_pe_cache, k_nope_cache, None, None)
            Cache tensors returned directly for attention kernel to read from.
        Prefill (is_output_kv=True):
            (None, None, k_pe, k_nope)
            k_pe:   [*batch_shape, qk_rope_head_dim]
            k_nope: [*batch_shape, kv_lora_rank]
    """
    if cache_mode == "PA_NZ":
        raise NotImplementedError(
            "PA_NZ cache mode not supported by Triton kernel. "
            "Use torch_npu.npu_kv_rmsnorm_rope_cache for NZ format."
        )

    batch_shape = kv.shape[:-1]  # e.g. [B, N, S]
    D_ROPE = cos.shape[-1]
    D_RMS = gamma.shape[0]
    D = kv.shape[-1]

    # Flatten kv to 2D [num_tokens, D]
    num_tokens = kv.numel() // D
    kv_flat = kv.reshape(num_tokens, D).contiguous()

    # Flatten cos/sin — handle shape mismatch when kv has extra dims (N, S)
    cos_tokens = cos.numel() // D_ROPE
    if cos_tokens == num_tokens:
        cos_flat = cos.reshape(num_tokens, D_ROPE).contiguous()
        sin_flat = sin.reshape(num_tokens, D_ROPE).contiguous()
    elif num_tokens % cos_tokens == 0:
        # Multi-head/sequence expansion: replicate cos/sin per head
        repeat_factor = num_tokens // cos_tokens
        cos_flat = (cos.reshape(cos_tokens, D_ROPE)
                    .repeat_interleave(repeat_factor, dim=0).contiguous())
        sin_flat = (sin.reshape(cos_tokens, D_ROPE)
                    .repeat_interleave(repeat_factor, dim=0).contiguous())
    else:
        raise ValueError(
            f"cos tokens ({cos_tokens}) incompatible with "
            f"kv tokens ({num_tokens})")

    if is_output_kv:
        # Prefill: compute outputs + write cache
        k_out, v_out, _, _ = kv_rms_norm_rope_cache(
            kv_flat, gamma, cos_flat, sin_flat, slots,
            k_pe_cache, k_nope_cache,
            epsilon=epsilon,
            is_output_kv=True,
        )
        # Reshape outputs to match input batch dimensions
        k_pe = k_out.view(*batch_shape, D_ROPE)
        k_nope = v_out.view(*batch_shape, D_RMS)
        return (None, None, k_pe, k_nope)
    else:
        # Decode: write cache only, return cache tensors directly
        kv_rms_norm_rope_cache(
            kv_flat, gamma, cos_flat, sin_flat, slots,
            k_pe_cache, k_nope_cache,
            epsilon=epsilon,
            is_output_kv=False,
        )
        return (k_pe_cache, k_nope_cache, None, None)


# ===================================================================
# PyTorch reference (both methods)
# ===================================================================
def kv_rms_norm_rope_cache_ref(
    kv, gamma, cos, sin, index, k_cache, v_cache,
    v_input=None, epsilon=1e-6,
):
    D_ROPE = cos.shape[-1]
    D_ROPE_HALF = D_ROPE // 2
    D_RMS = gamma.shape[0]

    cos_f = cos.reshape(-1, D_ROPE).float()
    sin_f = sin.reshape(-1, D_ROPE).float()

    if v_input is None:
        # V1: kv = [V(D_RMS) | K(D_ROPE)]
        D_TOTAL = D_RMS + D_ROPE
        N = kv.numel() // D_TOTAL
        kv_2d = kv.reshape(N, D_TOTAL)

        k_in = kv_2d[:, D_RMS:].float()
        k_real = k_in[:, 0::2]
        k_imag = k_in[:, 1::2]
        rope_real = k_real * cos_f[:, :D_ROPE_HALF] - k_imag * sin_f[:, :D_ROPE_HALF]
        rope_imag = k_imag * cos_f[:, D_ROPE_HALF:] + k_real * sin_f[:, D_ROPE_HALF:]
        k_rope = torch.cat([rope_real, rope_imag], dim=-1).to(kv.dtype)

        v_in = kv_2d[:, :D_RMS].float()
        var = v_in.pow(2).mean(dim=-1, keepdim=True)
        v_norm = (v_in / torch.sqrt(var + epsilon)
                  * gamma.float().unsqueeze(0)).to(kv.dtype)

        for i in range(N):
            if i < index.shape[0]:
                s = index[i].item()
                if s >= 0:
                    k_cache.view(-1, k_cache.shape[-1])[s] = k_rope[i]
                    v_cache.view(-1, v_cache.shape[-1])[s] = v_norm[i]
        return k_rope, v_norm, k_cache, v_cache
    else:
        # V2: kv = K(D_RMS), v_input = V(D_V)
        N = kv.numel() // D_RMS
        k_2d = kv.reshape(N, D_RMS).float()
        D_V = v_input.shape[-1]
        v_2d = v_input.reshape(N, D_V)

        var = k_2d.pow(2).mean(dim=-1, keepdim=True)
        k_norm = (k_2d / torch.sqrt(var + epsilon)
                  * gamma.float().unsqueeze(0))

        k_real = k_norm[:, 0:D_ROPE:2]
        k_imag = k_norm[:, 1:D_ROPE:2]
        rope_real = k_real * cos_f[:, :D_ROPE_HALF] - k_imag * sin_f[:, :D_ROPE_HALF]
        rope_imag = k_imag * cos_f[:, D_ROPE_HALF:] + k_real * sin_f[:, D_ROPE_HALF:]

        k_out = k_norm.clone()
        k_out[:, 0:D_ROPE:2] = rope_real
        k_out[:, 1:D_ROPE:2] = rope_imag
        k_out = k_out.to(kv.dtype)

        for i in range(N):
            if i < index.shape[0]:
                s = index[i].item()
                if s >= 0:
                    k_cache.view(-1, k_cache.shape[-1])[s] = k_out[i]
                    v_cache.view(-1, v_cache.shape[-1])[s] = v_2d[i]
        return k_out, v_2d, k_cache, v_cache
