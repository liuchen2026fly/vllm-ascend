# Qwen3.6 BF16 TP4 在 112 机器的复现

本次将 18 机已验收的镜像、源码和算子制品迁移到用户指定的 `112.29.145.3`，在新机器内对比关闭和开启 MegaMoe。换机结果与 18 机分开报告。四轮 OFF/ON 对照、精度检查和测试后的资源清理均已完成。

## 环境和内容一致性

| 项目 | 本次固定值 |
| --- | --- |
| 主机 | `112.29.145.3`，`bms-75847414-002` |
| 设备 | Ascend 910B4-1，物理 0、1、2、3，TP4/EP4 |
| SSH 与 HCCL 地址 | SSH 使用公网地址；HCCL 使用 `bond1 / 192.168.0.2` |
| 主机驱动 | `npu-smi 26.0.rc1` |
| 模型 | `/data01/models/Qwen3.6-35B-A3B` |
| 模型身份 | 26 个分片、37 个模型/配置/分词文件 SHA256 与 18 机一致；1045 个 BF16 张量 |
| 镜像 | `sha256:8dd949bff550c8e33211bbf6f0daa899c13eab3a1a140105bb2001509a6b568b` |
| 镜像归档 | `runtime_image.tar.gz`，8,701,842,121 字节，SHA256 `334f69d7b2d80507785d77e7b273620b8901c3dae510a1d6f37e00cfb7f1914e` |
| 框架源码 | 验收代码 `481b9570b7bf83b483f81d1a5450515e5f2a0e4a`；归档 `69306eee2ad9fdbcfc8f462777888ac42ca61071` 的运行代码相同 |
| 算子源码 | `23617531fd023f5a90612e6bddb8d21f951af234`，156 个扩展源码文件与 18 机逐文件一致 |
| 运行时 | 固定镜像及原挂载源码中的 vLLM 0.27.1、Python 3.12.13、torch-npu 2.10.0.post4、CANN 9.1 |
| 服务端口 | 8001 |

主机上原有的 vLLM 0.26/0.28 容器未作为本次测试运行时。模型下载目录中的 `.hf-manifest.json` 只存在于 18 机；它是下载元数据，不属于运行时模型输入。其余 37 个文件全部相同。

本次采集到 18 机主机驱动为 25.5.1，新机器为 26.0.rc1。相同镜像并不覆盖主机驱动、CPU 和设备拓扑。收益只用新机器内部 OFF/ON 数据计算；跨机器的绝对吞吐差异不能算作 MegaMoe 优化收益，也不能在未做消融时归因为驱动版本。

两机均为 192 逻辑 CPU 的 Kunpeng-920，但 18 机报告 8 个 NUMA 节点，新机器为 4 个；内核分别为 Kylin 4.19.90 与 Ubuntu 5.15.0。新机器 0/1 卡归属 NUMA 3，2/3 卡归属 NUMA 2。两组测试均开启 CPU binding；相同开关在不同拓扑上的实际绑核分配不应假定相同。这些是已记录的环境差异，尚未做单独归因实验。

日志核对显示，新机四轮实际 CPU 分配完全相同：rank 0 的 main 为 2–45，rank 2 为 98–141，rank 1 和 rank 3 均为 146–189，两者 acl/release 也共用 190/191。18 机每 rank 使用独立的 20 个 main 核。自动绑核在新机存在核重叠及分配差异，记录在 `environment/actual_cpu_bindings.json` 和原始服务日志中。该差异对性能的具体影响未做消融验证，本次没有在四轮中途修改绑核策略。

源码从精确 Git 提交归档；生成的 ABI、vendor 和扩展缓存单独打包，共 2513 个文件/链接，逐项验证。镜像和权重不放入 Git，保留在本机 `/data02/megamoe_bf16_repro_20260913/incoming` 和模型目录。凭据不属于交付内容。

| 迁移制品 | SHA256 |
| --- | --- |
| `framework_source_69306eee2ad9.tar.gz` | `4a5380cd218b28f29b5223f43f81c7ea8a5649a36f426da7db25d1f293829785` |
| `runtime_artifacts.tar.gz` | `e92a0fe7588536e027d573045877587ee9f446981c9f04e91d0eb199ec32cdbb` |
| `runtime_artifacts_manifest.json` | `f0d91d9855c5adaccf7cb0b9f84ab1ad7122ea0b9db820906253a9c586c2dd95` |
| `vllm-0.27.1.tar.gz` | `eec2d54d137ac1e59cb4c39226dfee1943eefc8f4788f5821d7300d6acbdb646` |

vLLM 是镜像内的可编辑安装，实际导入 `/data1/megamoe_q36/vllm-0.27.1`；单独复制镜像不足以复现。该源码目录来自保留的源码发行包，没有 Git 元数据。6374 个文件中，6373 个与发行包相同，只有生成的 `vllm/_version.py` 不同；本次复制了原运行时该文件，SHA256 为 `bdfab76f813ba913c311038049a09c966f2be5c1d07d541c29e522e2937bf965`，恢复后与 18 机全部一致。

容器保活入口沿用已验证方式：`--entrypoint /bin/bash IMAGE -c 'exec sleep infinity'`。直接在该镜像后传 `sleep infinity` 会被默认 Bash 入口当成脚本而退出。入口和可编辑源码挂载均须核对，不能仅检查镜像 ID。

新机器静态验收已通过：动态库 `ldd` 解析无缺失，运行时源码导入路径一致，225 项 CPU 测试通过，且 CPU 测试未初始化 NPU。

镜像内 Ascend 插件的可编辑安装还指向原机 `/data1/megamoe_q36/vllm-ascend`。实际服务一直通过 `PYTHONPATH` 使用交付源码；压测客户端是另一条 `docker exec` 命令，不能自动继承服务环境。本次对所有 OFF/ON 客户端统一显式设置：

```bash
PYTHONPATH=/data1/megamoe_gain_20260905/bf16_e2e_20260907/model_v31_runtime/vllm-ascend-bf16:/data1/megamoe_gain_20260905/bf16_opt_20260907/torch_extension_v26
ASCEND_RT_VISIBLE_DEVICES=
```

隐藏设备只作用于压测客户端，服务仍使用物理 0–3 卡。每一对测试在启动服务前先运行 `vllm bench serve --help` 和 `vllm serve --help`，确认实际客户端环境可导入全部命令。客户端仍使用原生 `vllm bench serve`；服务源码、服务环境、压测实现及参数未因该修复改变。

失败的首次准备保存在 `bf16_repro_112_off_on_20260913`：OFF 服务与固定精度检查通过，但客户端因上述源码路径缺失在导入时退出，未生成性能结果；随后服务清理成功。该次不计入性能样本。正式四轮结果保存在独立的 `bf16_repro_112_clientfixed_*_20260913` 目录。

## 对照设置

OFF 与 ON 共享同一容器、源码、模型和四张卡，DP/PP/PCP 均为 1。BF16，max-num-seqs=32，max-model-len=131072，max-num-batched-tokens=8192，显存利用率 0.90，seed=1024；启用异步调度、EP、chunked prefill、CPU binding 和 FULL_DECODE_ONLY，关闭 prefix cache。

双方 shared expert overlap、fuse_muls_add、enable_npugraph_ex 均开启，HCCL_DETERMINISTIC=true。只有三个配套开关不同：`MEGAMOE` 为 0/1，`enable_fused_mc2` 启动值为 0/2，`mega_moe_local_partial` 为 false/true。

压测使用现有 `vllm bench serve`：4K/6K 输入，256 输出，32 并发，每组 160 请求，固定 seed=1024、temperature=0、ignore-eos、random-range-ratio=0。关闭 profiling，计时期间没有额外源码构建、镜像加载或 CPU 测试任务。

4K/6K 指 `--random-input-len` 的请求值。加上 chat template 后，实际输入分别为 4106–4109 / 6153–6157 tokens，每组输入总量为 657026 / 984705 tokens。第一组 OFF/ON 的全部 160 条实际输入输出长度和压测参数，与 18 机逐项一致；跨主机的客户端导入环境修复没有改变这些负载设置。

每轮先检查短请求正确性，2K/4K 重复输出，以及 511/513/6144 tokens 的文本与 logprobs；检查实际四个 worker 和候选 vendor 的加载情况，再开始性能测试。OFF/ON 的真实输入输出长度必须相同。有限精度用例通过不等于证明任意输入无损。

## 在保留部署上重放

[本机输入包](megamoe_bf16_112_reproduction_inputs.zip) SHA256：`c8fac169906c3a2cd3e9a9a784093981e33206936c21e218864ab3e7c1fb55a7`。解压后的顶层目录为 `megamoe_bf16_112_reproduction/`，可将下方 INPUTS 指向它；保留部署中相同内容位于 `$ROOT/client_fixed_inputs`。

复现包以本次验收后的容器身份为准。先确认物理 0–3 卡以及 8001/29541 端口空闲。使用新的结果目录，保留已有数据。

```bash
set -euo pipefail
cd /
ROOT=/data02/megamoe_bf16_repro_20260913
BASE=/data1/megamoe_gain_20260905/bf16_e2e_20260907
INPUTS="$ROOT/client_fixed_inputs"
A="$BASE/bf16_repro_112_user_A"
B="$BASE/bf16_repro_112_user_B"
python3 -I "$INPUTS/prepare_current_machine.py" --work "$A" --order off-on
python3 -I "$INPUTS/prepare_current_machine.py" --work "$B" --order on-off
bash "$A/run_performance.sh"
bash "$A/collect_results.sh" > "$A/collected_results.log"
bash "$B/run_performance.sh"
bash "$B/collect_results.sh" > "$B/collected_results.log"
```

顺序为 OFF A1 → ON B1 → ON B2 → OFF A2。准备器验证源码、二进制、模型状态和容器身份，准备阶段不启动服务。控制器与收集器从 `/` 运行，准备器使用 `python3 -I`；新机器 `/root/struct.py` 会遮蔽同名标准库模块，不能从 `/root` 用未隔离的 Python 运行这些宿主机工具。

验收要求四轮精度检查通过；八组压测各 160 请求成功、输出 40960 tokens、每条 256 tokens、无错误；controller/evaluation/stop 均为 0；最终设备、端口和 worker 全部清理。吞吐收益按 `(ON 均值 / OFF 均值 - 1) × 100%` 计算，TTFT 同时报告改善或退化，不根据首次结果筛选轮次。

## 本次结果

| 输入 | OFF 吞吐 tokens/s | ON 吞吐 tokens/s | 吞吐变化 | OFF TTFT ms | ON TTFT ms | TTFT 变化 | TPOT 变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 730.006 | 729.728 | -0.038% | 1551.450 | 1564.966 | +0.871% | -0.112% |
| 6144 | 614.517 | 606.672 | -1.277% | 1901.474 | 1945.619 | +2.322% | +1.115% |

变化统一按 `(ON / OFF - 1) × 100%` 计算；延迟正值代表变慢。吞吐与延迟为两轮同模式结果的算术均值。

| 输入 | OFF A1 | ON B1 | ON B2 | OFF A2 | 配对一吞吐变化 | 配对二吞吐变化 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 4096 | 727.483 | 728.128 | 731.328 | 732.530 | +0.089% | -0.164% |
| 6144 | 611.520 | 604.963 | 608.381 | 617.514 | -1.072% | -1.479% |

本次未全部通过预定实用检查（两次配对均至少提高 0.2%，且最慢 ON 高于最快 OFF），不能据此宣称新机器稳定复现吞吐收益。该检查不是统计显著性检验，两轮样本也不能保证未来运行得到同样百分比。

四轮固定精度与重复请求检查均通过；八组压测各 160 请求成功，输出均为 40960 tokens，每条 256 tokens，无错误；两对 controller/evaluation、四轮 stop 及最终资源清理均成功。

[完整原始证据和汇总脚本](megamoe_bf16_112_evidence.zip) SHA256：`70098e4ba662ec14d8b4b284483a957c8171f49e11226a0de2080665a4a608b1`。包内包含原始结果、实际命令、精度输出、服务日志、环境清单和失败准备的记录。解压后运行：

```bash
python3 summary/summarize_delivery_abba.py \
  bf16_repro_112_clientfixed_off_on_20260913/collected_results.log \
  bf16_repro_112_clientfixed_on_off_20260913/collected_results.log \
  --output recomputed_abba.json
```

证据包已逐文件核验 SHA256，并从打包后的文件重新执行汇总，除本地输入路径字段外与原始汇总完全相同。18 机的历史结果、单算子测量范围及接入说明见[主交付报告](megamoe_bf16_delivery.md)。本次换机验收是整网对照，没有将 18 机单算子数字标记为 112 机测量结果。

额外离线核对：4K/6K 压测各 160 条保存的返回文本，在四轮之间逐条完全一致。压测结果不返回 logprobs；logprobs 一致性由前面的固定精度用例验证。完整计数见证据包 `summary/benchmark_text_comparison.json`。这些有限请求不构成任意模型输入无损的证明。
