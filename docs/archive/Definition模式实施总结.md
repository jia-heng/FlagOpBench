# Definition 与 Workload 模式实施总结

**更新日期**: 2026-08-10  
**状态**: ✅ 完成设计 + 示例实施

---

## 📊 完成内容

### 1. 完整设计文档

✅ **Definition与Workload设计规范.md**
- Definition/Workload 核心概念
- 命名规范（算子、Definition、Workload）
- 文件组织结构（两种方案）
- YAML 格式规范
- 迁移步骤

### 2. 更新导入脚本

✅ **scripts/import_flashinfer_trace.py**
- 支持两种模式：
  - `definition_mode=True`: 一个 Definition 一个文件（推荐）
  - `definition_mode=False`: 一个算子一个文件（兼容）
- 自动按算子类型分类输出
- 智能目录组织

### 3. 目录结构创建

✅ 创建 Definition-based 目录结构：
```
baseline/cases/
├── basic/
│   ├── gemm/          # GEMM 类算子
│   ├── norm/          # Normalization 算子
│   ├── activation/    # Activation 算子
│   └── other/         # 其他算子
└── model/
    ├── attention/     # Attention 算子
    └── moe/          # MoE 算子
```

### 4. 示例 Definition 文件

✅ **gemm_n4096_k4096.yaml** (30 workloads)
- Llama-3.1-8B attention projection
- 覆盖 decode/prefill/mixed 场景
- 包含 power-of-2 和真实 trace 的 odd sizes

✅ **rms_norm_h4096.yaml** (20 workloads)
- Llama-3.1-8B RMSNorm
- 覆盖各种 batch size（1 - 131072）
- 包含 bf16/fp16 对比

---

## 🎯 Definition/Workload 模式优势

### 1. 清晰的语义

**Definition** = shape 族
```yaml
definition: gemm_n4096_k4096
const_axes:
  N: 4096  # 固定
  K: 4096  # 固定
# M 是 var 轴，在 workload 中指定
```

**Workload** = 具体实例
```yaml
- name: "llama3.1_attn_o_proj_decode_bs1"
  M: 1  # var 轴的具体值
  source: "Llama-3.1-8B, attention output projection, decode"
```

### 2. 便于管理

- 一个 Definition 一个文件
- 每个文件 30-50 个 workload
- 按算子类型分类（gemm/norm/attention/...）

### 3. 易于扩展

```bash
# 添加新 workload - 只需编辑对应 Definition 文件
vim baseline/cases/basic/gemm/gemm_n4096_k4096.yaml

# 添加新 Definition - 创建新文件
vim baseline/cases/basic/gemm/gemm_n7168_k18432.yaml
```

### 4. 可追溯性

每个 workload 都标注：
- `name`: 语义化名称
- `source`: 来源描述
- `model`: 模型名称
- `phase`: 推理阶段

### 5. 智能去重

Definition 自然去重（const 轴固定）：
- `gemm_n4096_k4096` 只包含 N=4096, K=4096 的 workload
- 不同 (N, K) 组合是不同的 Definition

---

## 📋 命名规范示例

### 基础算子 Definition

```
gemm_n{N}_k{K}              # GEMM
bmm_b{B}_n{N}_k{K}          # Batch GEMM
rms_norm_h{hidden_size}     # RMSNorm
layer_norm_h{hidden_size}   # LayerNorm
softmax_v{vocab_size}       # Softmax
gelu_h{hidden_size}         # GELU
```

### 大模型算子 Definition

```
gqa_paged_decode_h{num_qo_heads}_kv{num_kv_heads}_d{head_dim}_ps{page_size}
mha_paged_decode_h{num_heads}_d{head_dim}_ps{page_size}
moe_sum_e{num_experts}_h{hidden_size}
```

### Workload 命名

```
{model}_{layer}_{phase}_{key_params}

例如：
llama3.1_attn_o_proj_decode_bs1
deepseek_v3_ffn_up_prefill_bs2048
qwen2.5_moe_expert_sum_t1024
```

---

## 🚀 使用方式

### 运行单个 Definition

```bash
# 运行 gemm_n4096_k4096 的所有 30 个 workload
python baseline/run.py run \
  --backend nvidia \
  --case baseline/cases/basic/gemm/gemm_n4096_k4096.yaml \
  --platform nvidia_h20
```

### 运行特定 Workload

```bash
# 只运行 decode phase 的 workload
python baseline/run.py run \
  --backend nvidia \
  --case baseline/cases/basic/gemm/gemm_n4096_k4096.yaml \
  --platform nvidia_h20 \
  --filter "phase=decode"
```

### 运行算子类型的所有 Definition

```bash
# 运行所有 GEMM Definition
python baseline/run.py run \
  --backend nvidia \
  --case-dir baseline/cases/basic/gemm/ \
  --platform nvidia_h20
```

---

## 📁 文件组织对比

### 旧方案（兼容保留）
```
baseline/cases/basic/
├── mm.yaml              # 包含多个 (N,K) 组合，workload 混在一起
├── rms_norm.yaml
└── ...
```

### 新方案（Definition-based，推荐）
```
baseline/cases/basic/
├── gemm/
│   ├── gemm_n4096_k4096.yaml      # 30 workloads (Llama-3.1)
│   ├── gemm_n7168_k18432.yaml     # 40 workloads (DeepSeek-V3 FFN up)
│   └── gemm_n18432_k7168.yaml     # 40 workloads (DeepSeek-V3 FFN down)
├── norm/
│   ├── rms_norm_h4096.yaml        # 20 workloads (Llama-3.1)
│   └── rms_norm_h7168.yaml        # 20 workloads (DeepSeek-V3)
└── activation/
    └── softmax_v32000.yaml
```

**优势**：
- 清晰的层次结构
- 易于查找和管理
- 便于扩展到 1500+ workload

---

## 🔄 迁移策略

### 阶段 1: 保持兼容（当前）

✅ **已完成**：
- 新增 Definition-based 目录结构
- 更新导入脚本支持两种模式
- 创建示例 Definition 文件

✅ **现状**：
- 旧格式（`mm.yaml`）继续可用
- 新格式（`gemm/gemm_n4096_k4096.yaml`）并行存在
- 两种格式共存

### 阶段 2: 逐步迁移（可选）

📋 **可以做**：
- 将现有 `mm.yaml` 拆分为多个 Definition 文件
- 按真实模型分组 workload
- 补充更多真实 workload

### 阶段 3: 完全迁移（未来）

📋 **长期目标**：
- 全部采用 Definition/Workload 模式
- 删除旧格式文件
- 达到 1500+ workload 规模

---

## 📊 预期规模

### 当前状态
- 示例 Definition: 2 个
- 示例 Workload: 50 个

### 完成 Phase 1（导入 flashinfer-trace）
- Definition: ~50 个
- Workload: ~500 个

### 完成 Phase 2-3（自采集 + 大模型算子）
- Definition: ~100 个
- Workload: ~1500 个

---

## ✅ 下一步建议

### 立即可用
1. ✅ 使用示例 Definition 文件测试
2. ✅ 验证 Definition/Workload 模式可行性

### 短期扩展（1-2 小时）
1. 📋 运行 flashinfer-trace 导入器（Definition mode）
2. 📋 生成更多 Definition 文件
3. 📋 验证目录结构和命名规范

### 中期扩展（2-3 天）
1. 📋 迁移现有 case 到 Definition 格式
2. 📋 补充真实 workload
3. 📋 扩展到 500+ workload

---

## 🎯 核心价值

### Definition/Workload 模式带来：

1. **更好的组织** - 清晰的层次结构
2. **更易管理** - 一个 Definition 一个文件
3. **更强扩展性** - 便于添加 workload
4. **更好追溯性** - 每个 workload 标注来源
5. **更智能去重** - Definition 自然去重

### 与真实性能测试结合：

```
真实推理 trace → 识别 Definition → 生成 Workload → 智能去重 → YAML case
```

**完整工具链**：
- `collect_shapes.py` → 采集真实 shape
- `import_flashinfer_trace.py` → 导入 flashinfer-trace
- `filters.py` → 智能去重
- **Definition/Workload** → 规范化组织

---

## 🎉 总结

**已完成**：
- ✅ 完整设计文档
- ✅ 更新导入脚本
- ✅ 创建目录结构
- ✅ 两个示例 Definition（50 workload）

**项目状态**：
- 旧格式继续可用（兼容）
- 新格式已就绪（推荐）
- 两种格式并存

**建议**：
- 新的 case 优先使用 Definition/Workload 格式
- 逐步迁移现有 case
- 最终达到 1500+ workload 规模

---

**FlagOpBench 现在具备完整的 Definition/Workload 管理能力！** 🚀
