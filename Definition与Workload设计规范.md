# Definition 与 Workload 设计规范

## 核心概念

### Definition（算子定义）
描述一个算子的 **shape 族**，而非单个 shape。

**特点**：
- 包含 **const 轴**（由模型架构决定，固定不变）
- 不包含 **var 轴**（运行时动态变化）
- 一个 Definition 覆盖多个 Workload

**命名规范**：
```
{operator}_{const_axis1}_{const_axis2}_...

例如：
- gemm_n4096_k4096        # GEMM，N=4096, K=4096 固定，M 可变
- rms_norm_h4096          # RMSNorm，hidden_size=4096 固定
- gqa_decode_h32_kv8_d128 # GQA Decode，heads/dim 固定
```

### Workload（具体用例）
为 Definition 的 var 轴绑定具体值。

**特点**：
- 继承 Definition 的 const 轴
- 指定 var 轴的具体值
- 标注来源（模型、场景）

---

## FlagOpBench 命名规范

### 基础算子 Definition 命名

#### 1. GEMM 类（mm, bmm）

**命名模式**：`gemm_n{N}_k{K}` 或 `bmm_b{B}_n{N}_k{K}`

**const 轴**：N, K（权重 shape）  
**var 轴**：M（batch × seq_len）

**示例**：
```yaml
# Definition: gemm_n4096_k4096
const_axes:
  N: 4096  # output features
  K: 4096  # input features

# Workload 1
- name: "llama3.1_attn_o_proj_m256"
  M: 256
  source: "Llama-3.1-8B, attention output projection, batch=1, seq=256"

# Workload 2
- name: "llama3.1_attn_o_proj_m2048"
  M: 2048
  source: "Llama-3.1-8B, attention output projection, batch=1, seq=2048"
```

**常见 Definition**：
- `gemm_n4096_k4096` - Llama-3.1 attention projections
- `gemm_n7168_k18432` - DeepSeek-V3 FFN up
- `gemm_n18432_k7168` - DeepSeek-V3 FFN down
- `bmm_b32_n128_k128` - Attention score computation

#### 2. Normalization 类（layernorm, rms_norm）

**命名模式**：`{norm_type}_h{hidden_size}`

**const 轴**：hidden_size  
**var 轴**：batch_size

**示例**：
```yaml
# Definition: rms_norm_h4096
const_axes:
  hidden_size: 4096

# Workload 1
- name: "llama3.1_decode_bs1"
  batch_size: 1
  source: "Llama-3.1-8B, decode phase, batch=1"

# Workload 2
- name: "llama3.1_prefill_bs32"
  batch_size: 32
  source: "Llama-3.1-8B, prefill phase, batch=32"
```

**常见 Definition**：
- `rms_norm_h4096` - Llama-3.1-8B
- `rms_norm_h7168` - DeepSeek-V3
- `layer_norm_h4096` - GPT-style models

#### 3. Activation 类（softmax, gelu, silu）

**命名模式**：`{activation}_v{vocab_size}` 或 `{activation}_h{hidden_size}`

**const 轴**：vocab_size 或 hidden_size  
**var 轴**：batch_size, seq_len

**示例**：
```yaml
# Definition: softmax_v32000
const_axes:
  vocab_size: 32000

# Workload
- name: "llama3.1_sampling_bs1"
  batch_size: 1
  seq_len: 1
  source: "Llama-3.1-8B, sampling phase"
```

---

### 大模型算子 Definition 命名

#### 1. Attention 类

**命名模式**：`{attn_type}_h{num_qo_heads}_kv{num_kv_heads}_d{head_dim}_ps{page_size}`

**const 轴**：num_qo_heads, num_kv_heads, head_dim, page_size  
**var 轴**：batch_size, num_kv_indices, num_pages

**示例**：
```yaml
# Definition: gqa_paged_decode_h32_kv8_d128_ps1
const_axes:
  num_qo_heads: 32
  num_kv_heads: 8
  head_dim: 128
  page_size: 1

# Workload 1
- name: "llama3.1_decode_bs1_kvlen73"
  batch_size: 1
  num_kv_indices: 73
  num_pages: 73
  source: "Llama-3.1-8B, decode, avg_kv_len=73"

# Workload 2
- name: "llama3.1_decode_bs8_kvlen512"
  batch_size: 8
  num_kv_indices: 4096
  num_pages: 4096
  source: "Llama-3.1-8B, decode, avg_kv_len=512"
```

**常见 Definition**：
- `gqa_paged_decode_h32_kv8_d128_ps1` - Llama-3.1-8B GQA decode
- `mha_paged_decode_h32_kv32_d128_ps1` - GPT-style MHA decode
- `mla_paged_decode_h32_d192_ps1` - DeepSeek-V3 MLA

#### 2. MoE 类

**命名模式**：`moe_{type}_e{num_experts}_h{hidden_size}`

**const 轴**：num_experts, hidden_size  
**var 轴**：num_tokens

**示例**：
```yaml
# Definition: moe_sum_e8_h4096
const_axes:
  num_experts: 8
  hidden_size: 4096

# Workload
- name: "deepseek_v3_moe_t2048"
  num_tokens: 2048
  source: "DeepSeek-V3, MoE layer, 2048 tokens"
```

---

## 文件组织结构

### 方案 1: 按 Definition 组织（推荐）

```
baseline/cases/
├── basic/
│   ├── gemm/
│   │   ├── gemm_n4096_k4096.yaml          # 一个 Definition 一个文件
│   │   ├── gemm_n7168_k18432.yaml
│   │   └── gemm_n18432_k7168.yaml
│   ├── norm/
│   │   ├── rms_norm_h4096.yaml
│   │   ├── rms_norm_h7168.yaml
│   │   └── layer_norm_h4096.yaml
│   └── activation/
│       ├── softmax_v32000.yaml
│       └── gelu_h4096.yaml
└── model/
    ├── attention/
    │   ├── gqa_paged_decode_h32_kv8_d128_ps1.yaml
    │   └── mha_paged_decode_h32_kv32_d128_ps1.yaml
    └── moe/
        └── moe_sum_e8_h4096.yaml
```

**优点**：
- 清晰的层次结构
- 易于查找特定 Definition
- 便于管理大量 Workload

### 方案 2: 按算子类型组织（当前）

```
baseline/cases/
├── basic/
│   ├── mm.yaml                    # 包含多个 Definition
│   ├── bmm.yaml
│   ├── rms_norm.yaml
│   └── ...
└── model/
    ├── paged_attention.yaml
    └── moe.yaml
```

**优点**：
- 文件数量少
- 便于快速浏览

---

## YAML 格式规范

### 单 Definition 文件（推荐）

```yaml
# baseline/cases/basic/gemm/gemm_n4096_k4096.yaml

# Definition 信息
definition: gemm_n4096_k4096
operator: mm
level: basic
source: flashinfer-trace

# const 轴（固定）
const_axes:
  N: 4096
  K: 4096

# 测试配置
warmup: 10
iters: 100

# Workloads（var 轴）
workloads:
  # ========== Llama-3.1-8B ==========
  - name: "llama3.1_attn_o_proj_decode_bs1"
    M: 1
    dtype: bf16
    source: "Llama-3.1-8B, attention output projection, decode, batch=1"
    model: "meta-llama/Llama-3.1-8B"
    phase: "decode"

  - name: "llama3.1_attn_o_proj_prefill_bs256"
    M: 256
    dtype: bf16
    source: "Llama-3.1-8B, attention output projection, prefill, seq=256"
    model: "meta-llama/Llama-3.1-8B"
    phase: "prefill"

  - name: "llama3.1_attn_o_proj_prefill_bs2048"
    M: 2048
    dtype: bf16
    source: "Llama-3.1-8B, attention output projection, prefill, seq=2048"
    model: "meta-llama/Llama-3.1-8B"
    phase: "prefill"

  # ========== DeepSeek-V3 ==========
  - name: "deepseek_v3_attn_o_proj_decode_bs1"
    M: 1
    dtype: bf16
    source: "DeepSeek-V3, attention output projection, decode, batch=1"
    model: "deepseek-ai/DeepSeek-V3"
    phase: "decode"

  # ... 更多 Workload（30-50 个）
```

### 多 Definition 文件（紧凑）

```yaml
# baseline/cases/basic/mm.yaml

operator: mm
level: basic
source: flashinfer-trace

# Definition 1
definitions:
  - name: gemm_n4096_k4096
    const_axes:
      N: 4096
      K: 4096
    workloads:
      - name: "llama3.1_m256"
        M: 256
        dtype: bf16
      # ... 更多

  - name: gemm_n7168_k18432
    const_axes:
      N: 7168
      K: 18432
    workloads:
      - name: "deepseek_v3_m2048"
        M: 2048
        dtype: bf16
      # ... 更多
```

---

## Workload 命名规范

### 基本格式

```
{model}_{layer}_{phase}_{key_params}

例如：
- llama3.1_attn_o_proj_decode_bs1
- deepseek_v3_ffn_up_prefill_bs2048
- qwen_moe_expert_sum_t1024
```

### 命名要素

1. **model**: 模型名称缩写
   - `llama3.1` - Llama-3.1
   - `deepseek_v3` - DeepSeek-V3
   - `qwen2.5` - Qwen-2.5

2. **layer**: 算子所在层
   - `attn_o_proj` - Attention output projection
   - `ffn_up` - FFN up projection
   - `ffn_down` - FFN down projection
   - `moe_expert` - MoE expert layer

3. **phase**: 推理阶段
   - `decode` - Decode phase (seq_len=1)
   - `prefill` - Prefill phase (seq_len>1)

4. **key_params**: 关键参数
   - `bs{N}` - batch_size=N
   - `m{N}` - M=N (GEMM)
   - `t{N}` - num_tokens=N (MoE)
   - `kvlen{N}` - avg_kv_len=N (Attention)

---

## 迁移步骤

### Step 1: 识别 Definition

分析当前的 case，提取 const 轴：

```bash
# 当前: baseline/cases/basic/mm.yaml
# 包含多个不同的 (N, K) 组合

# 识别出的 Definition:
# - gemm_n4096_k4096    (Llama-3.1 attention)
# - gemm_n7168_k18432   (DeepSeek-V3 FFN up)
# - gemm_n18432_k7168   (DeepSeek-V3 FFN down)
```

### Step 2: 重组文件结构

```bash
# 创建目录
mkdir -p baseline/cases/basic/{gemm,norm,activation}
mkdir -p baseline/cases/model/{attention,moe}

# 拆分现有 case
# mm.yaml → gemm_n4096_k4096.yaml, gemm_n7168_k18432.yaml, ...
```

### Step 3: 更新导入脚本

```python
# scripts/import_flashinfer_trace.py

# 修改输出逻辑：
# 旧: 一个算子一个文件（mm.yaml）
# 新: 一个 Definition 一个文件（gemm_n4096_k4096.yaml）

def write_definition_case(definition_name, operator, workloads):
    # 按 Definition 输出到独立文件
    output_file = f"baseline/cases/basic/{operator}/{definition_name}.yaml"
    ...
```

### Step 4: 更新 CLI 工具

```python
# baseline/run.py

# 支持两种模式：
# 1. 运行单个 Definition
python baseline/run.py run \
  --case baseline/cases/basic/gemm/gemm_n4096_k4096.yaml

# 2. 运行算子的所有 Definition
python baseline/run.py run \
  --operator mm \
  --backend nvidia
```

---

## 实施建议

### 阶段 1: 保持兼容（推荐）

**当前不改**，保持现有结构可用。

**新增支持**：
- 新的 Definition-based 命名
- 新的文件组织结构
- 两种格式共存

### 阶段 2: 逐步迁移

**优先迁移**：
- GEMM 类（最多 Workload）
- Attention 类（最复杂）

**保留原有**：
- 简单算子（activation, norm）

### 阶段 3: 完全迁移（可选）

**全部采用** Definition/Workload 模式。

---

## 优势总结

### Definition/Workload 模式的优势

1. **清晰的语义**
   - Definition 描述 shape 族
   - Workload 描述具体实例
   - 易于理解和维护

2. **便于管理**
   - 一个 Definition 一个文件
   - Workload 按模型/场景分组
   - 支持大量 Workload（40-50 个）

3. **易于扩展**
   - 添加新 Workload 不改 Definition
   - 添加新 Definition 不影响现有
   - 便于从 flashinfer-trace 导入

4. **可追溯性**
   - 每个 Workload 标注来源
   - 便于复现和验证

5. **智能去重**
   - Definition 自然去重（const 轴固定）
   - Workload 按语义去重（var 轴）

---

## 下一步行动

### 立即可做

1. **更新导入脚本**
   ```bash
   # 修改 scripts/import_flashinfer_trace.py
   # 按 Definition 输出文件
   ```

2. **创建目录结构**
   ```bash
   mkdir -p baseline/cases/basic/{gemm,norm,activation}
   mkdir -p baseline/cases/model/{attention,moe}
   ```

3. **迁移一个示例**
   ```bash
   # 将 mm.yaml 拆分为多个 Definition 文件
   # gemm_n4096_k4096.yaml, gemm_n7168_k18432.yaml, ...
   ```

### 可选扩展

4. **更新 CLI 工具** - 支持 Definition 级别的运行
5. **更新文档** - 说明 Definition/Workload 规范
6. **批量迁移** - 迁移所有现有 case

---

需要我立即实施这个方案吗？我可以：

1. ✅ 更新导入脚本（按 Definition 输出）
2. ✅ 创建新的目录结构
3. ✅ 迁移一个完整示例（GEMM）
4. ✅ 更新文档说明

还是先保持现有结构，作为可选扩展？
