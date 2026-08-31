# FlagOpBench — FlagOS 单算子性能测试框架

对比 FlagOS 算子库（FlagGems-vllm / FlagAttention / FlagGems）与 vLLM 原生实现的单算子性能。
在同一台 GPU 上，分别通过两个 provider 加载实现，跑相同 workload，生成 JSON 结果后做对比。

## 当前状态

- **27 个算子**已接入（operators/ 目录）
- **24 个**双边对比完成（flagos + vllm 均跑通）
- **12 个模型配置**支持自动 case 生成
- 覆盖 Attention、MoE、KV Cache、Norm/RoPE、Activation、Pack/Unpack 等类别

## 目录结构

```
FlagOpBench/
├── run.py                  # 执行 benchmark
├── compare.py              # 对比结果，生成 compare JSON
├── gen_cases.py            # 从模型配置自动生成 workload YAML
├── framework/              # 核心框架
│   ├── base_operator.py    # 算子基类（prepare_inputs / compute_flops / compute_bytes）
│   ├── registry.py         # 算子注册表（自动发现 operators/ 下的实现）
│   ├── timer.py            # GPU 计时（cudaEvent）
│   ├── runner.py           # 测试执行器
│   └── reporter.py         # JSON 输出
├── providers/              # 算子加载层
│   ├── flagos_provider.py  # FlagOS：按 library 属性 import 对应库的函数
│   └── vllm_provider.py    # vLLM：加载 baseline 实现
├── operators/              # 算子定义（每个算子一个目录）
│   ├── fused_moe/
│   ├── flash_mla_with_kvcache/
│   ├── chunk_gated_delta_rule_flaggems_vllm/
│   └── ...（共 27 个）
├── cases/                  # Workload 定义
│   ├── generated/merged/   # gen_cases.py 产出的 merged YAML（跑 benchmark 用这里）
│   └── _template.yaml      # 参考模板
├── casegen/                # Case 生成逻辑
│   ├── model_parser.py     # 解析模型 JSON 配置
│   ├── param_mapping.py    # 算子参数映射规则
│   ├── profile_loader.py   # 部署场景（online_serving 等）
│   └── generator.py        # 生成 merged YAML
├── model_configs/          # 模型架构参数（JSON）
│   ├── deepseek_v4_pro.json
│   ├── kimi_k3.json
│   ├── qwen3.8-2.4t-a95b.json
│   └── ...（共 12 个）
├── operator_registry.yaml  # 算子 → 模型参数映射注册表
└── results/                # 输出目录
    ├── {op}_flagos.json    # FlagOS 测试结果
    ├── {op}_vllm.json      # vLLM 测试结果
    └── {op}_compare.json   # 对比结果
```

## 快速开始

```bash
cd /data/jianheng/works/Flagos/FlagOpBench

# 1. 生成 case（首次或模型/算子变动后执行）
python gen_cases.py

# 2. 跑 FlagOS
python run.py --provider flagos --case cases/generated/merged/fused_moe.yaml

# 3. 跑 vLLM baseline
python run.py --provider vllm --case cases/generated/merged/fused_moe.yaml

# 4. 对比
python compare.py --op fused_moe --save
# 输出: results/fused_moe_compare.json + 终端表格

# 一键全量对比（需要所有 result JSON 都已生成）
python compare.py --all --save
```

## 性能结果概览

| 算子 | Geo-Mean Speedup | 结论 |
|------|-----------------|------|
| group_gemm | **1.45x** | FlagOS 大幅领先 |
| fused_moe | **1.14x** | FlagOS 领先 |
| pack_seq_triton | **1.27x** | FlagOS 领先 |
| unpack_seq_triton | **1.05x** | FlagOS 略快 |
| chunk_gated_delta_rule_flaggems_vllm | **1.05x** | FlagOS TLE 路径 prefill 快 12-24% |
| chunk_gated_delta_rule_flag_gems | 1.00x | 持平（同一 kernel） |
| fused_q_kv_rmsnorm | 0.94x | 接近持平 |
| chunk_gated_delta_rule_flag_attn | 0.98x | 接近持平 |
| flash_attn_varlen_func | 0.82x | vLLM 快（FlashAttention CUDA） |
| topk_softplus_sqrt | 0.69x | vLLM 快 |
| flash_mla_with_kvcache | 0.56x | vLLM 快（FlashMLA CUDA） |
| swiglu | 0.55x | vLLM 快 |
| grouped_topk | 0.28x | vLLM 大幅领先 |

> Speedup > 1.0 表示 FlagOS 更快。完整数据见 `results/*_compare.json`。

## 添加新算子

详见 [WORKFLOW.md](WORKFLOW.md) 完整流程。简要步骤：

1. `operators/{op_name}/operator.py` — 实现 `BaseOperator` 子类
2. `operator_registry.yaml` — 注册算子的参数映射规则
3. `providers/flagos_provider.py` — 确认 library 对应的 import 路径
4. `providers/vllm_provider.py` — 添加 `_load_{op_name}` baseline 加载
5. `python gen_cases.py --operators {op_name}` — 生成 case
6. 跑 benchmark + compare

## 环境要求

本框架在 FlagOS Docker 容器内运行，已预装：
- Python 3.12, PyTorch 2.x, Triton 3.6+
- flaggems, flaggems_vllm, flag_attn
- vLLM（通过 vllm_provider 加载 baseline）

## 相关文档

- [WORKFLOW.md](WORKFLOW.md) — 从零添加算子的完整流程（含 7 步操作指南）
- [OPERATOR_STATUS.md](OPERATOR_STATUS.md) — 各算子集成状态和结果摘要
