# Qwen3.5 Performance Optimization Guide for vllm-ascend

本文档说明如何启用 Qwen3.5 模型在 vllm-ascend（Ascend NPU）上的性能优化。

## 优化列表

我们实现了三个 P0 级别的性能优化：

| 优化项 | 预期收益 | 实现状态 | 自动启用 |
|--------|---------|---------|---------|
| **P0-1: RMSNormGated Triton kernel** | 2-4% | ✅ 已实现 | ✅ 是 |
| **P0-2: AllReduce+RMSNorm 融合** | 1-2% | ✅ 已实现 | ⚠️ 需启用编译模式 |
| **P0-3: Triton split 条件放宽** | 1-3% | ✅ 已实现 | ✅ 是 |

**总计预期收益**: ~4-9% 端到端加速

---

## 优化 1: RMSNormGated Triton Kernel

### 实现位置
- `vllm_ascend/ops/triton/fla/fused_rmsnorm_gated.py` - Triton 融合 kernel
- `vllm_ascend/patch/worker/patch_qwen3_5.py` - 在 GatedDeltaNet 中集成使用

### 功能描述
将 GatedDeltaNet 层中的两个操作融合为单个 Triton kernel：
1. RMS Normalization: `x_normed = x / sqrt(mean(x^2) + eps) * weight`
2. Element-wise gating: `output = x_normed * z`

### 启用方式
**自动启用**，无需额外配置。

当 Triton 可用时，patch 会自动使用融合 kernel，否则回退到 PyTorch 实现。

---

## 优化 2: AllReduce+RMSNorm 融合

### 实现位置
- `vllm_ascend/compilation/passes/allreduce_rmsnorm_fusion_pass.py` - 编译 Pass
- `vllm_ascend/compilation/npugraph_ex_passes/graphex_allreduce_rmsnorm_fusion_pass.py` - NPU Graph EX Pass

### 功能描述
在 Full Attention 层的输出投影后，将以下操作融合：
```
Matmul (RowParallelLinear.o_proj)
  → AllReduce (TP通信)
  → AddRMSNorm (residual + RMSNorm)
```

### 启用方式
启用 NPU Graph 编译模式（默认已启用融合）。

---

## 优化 3: Triton Split Kernel 条件放宽

### 实现位置
- `vllm_ascend/patch/worker/patch_qwen3_next.py` - 条件判断逻辑修改

### 功能描述
扩展 `fused_qkvzba_split_reshape_cat` Triton kernel 的使用条件，提高其在常规推理中的应用率。

### 修改内容

**旧条件**（限制过严）：
```python
if (self.num_v_heads // self.num_k_heads in [1, 2, 4]  # 仅支持特定GQA比例
    and is_cuda_graph                                   # 必须在CUDAGraph模式
    and divide_grid < 65536):                          # Grid大小限制
```

**新条件**（更宽松）：
```python
gqa_ratio = self.num_v_heads // self.num_k_heads
is_gqa_ratio_valid = (self.num_v_heads % self.num_k_heads == 0)  # 任意整数比例
is_grid_size_valid = divide_grid < 65536                         # Grid大小限制

if is_gqa_ratio_valid and is_grid_size_valid:  # 移除了CUDAGraph要求
```

### 启用方式
**自动启用**，无需额外配置。

---

## 使用示例

```python
from vllm import LLM, SamplingParams

# 初始化模型（所有优化自动启用）
llm = LLM(
    model="Qwen/Qwen3.5-32B-Instruct",
    tensor_parallel_size=4,
    gpu_memory_utilization=0.8,
    max_model_len=4096,
)

# 生成
prompts = ["Explain quantum computing in simple terms"]
sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=512)
outputs = llm.generate(prompts, sampling_params)
```

---

## 预期性能提升

基于优化分析，相对于**未优化的 Qwen3.5** 基线：

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| **TTFT** | 100ms | ~92ms | ~8% |
| **TPOT** | 10ms | ~9.2ms | ~8% |
| **Throughput** | 100 tokens/s | ~108 tokens/s | ~8% |
