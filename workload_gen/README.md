# Workload Generator - 配置推导式 Workload 生成器

## 功能

从模型配置文件（config.json）推导真实推理 workload，**无需部署模型**，**零 GPU 依赖**。

## 快速开始

### 1. 生成 workload

```bash
# 从本地配置文件生成（推荐）
python -m workload_gen generate \
  --config model_configs/llama_3.1_8b.json \
  --output baseline/cases/traced/llama_3.1_8b/ \
  --scenarios standard

# 自定义场景
python -m workload_gen generate \
  --config model_configs/qwen2.5_7b.json \
  --output baseline/cases/traced/qwen2.5_7b/ \
  --scenarios "decode:1,4,8,16;prefill:128,512,2048"
```

### 2. 运行生成的 workload

```bash
# 运行单个算子
python baseline/run.py run \
  --backend nvidia \
  --case baseline/cases/traced/llama_3.1_8b/rms_norm_llama_3.1_8b.yaml

# 运行整个目录
python baseline/run.py run \
  --backend nvidia \
  --case-dir baseline/cases/traced/llama_3.1_8b/
```

## 模型配置

### 已支持模型

| 模型 | 配置文件 | 架构 | 参数量 | 特点 |
|------|---------|------|--------|------|
| Llama-3.1-8B | `llama_3.1_8b.json` | llama | 8B | Meta 标准架构 |
| Qwen2.5-7B | `qwen2.5_7b.json` | llama | 7B | 阿里 Dense 模型 |

**注意：** 所有使用 GQA + SwiGLU 架构的模型（Llama/Qwen/Yi/Baichuan/InternLM）都可使用 `llama` 架构类型。

### 配置文件格式

```json
{
  "model_name": "llama_3.1_8b",
  "model_type": "llama",
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "intermediate_size": 14336,
  "num_hidden_layers": 32,
  "vocab_size": 128256,
  "rope_theta": 500000.0,
  "max_position_embeddings": 131072,
  "rms_norm_eps": 1e-5
}
```

### 获取新模型配置

#### 方法 1：使用下载脚本（需要代理）

```bash
bash scripts/download_model_configs.sh
```

#### 方法 2：手动从 HuggingFace 复制

1. 访问模型主页（如 https://huggingface.co/meta-llama/Llama-3.1-8B）
2. 点击 "Files and versions" → 下载 `config.json`
3. 保存到 `model_configs/` 目录
4. 添加 `model_name` 和 `model_type` 字段

## 推理场景

### 标准场景（`--scenarios standard`）

**Decode（6 个）：**
- batch_size: 1, 4, 8, 16, 32, 64
- seq_len: 1（单 token 生成）
- kv_len: 2048（假设 2K context）

**Prefill（7 个）：**
- batch_size: 1
- seq_len: 128, 256, 512, 1024, 2048, 4096, 8192

总共 **13 个场景**，覆盖主流推理模式。

### 自定义场景

格式：`"phase:values;phase:values"`

```bash
# 示例：3 个 decode + 2 个 prefill
--scenarios "decode:1,4,8;prefill:128,512"
```

## 架构支持

### 当前支持（P0）

| 架构类型 | 覆盖模型 | 算子特征 |
|---------|---------|---------|
| `llama` | Llama, Qwen Dense, Yi, Baichuan, InternLM, Mistral-7B | GQA + SwiGLU FFN |

### 后续支持

- **P1**: `mixtral`, `qwen_moe` — MoE 架构（+25% 覆盖）
- **P2**: `deepseek_v3` — MLA + MoE（+10% 覆盖）
- **P3**: `glm`, `gemma` — 特殊 Attention（+5% 覆盖）

查看支持的架构：
```bash
python -m workload_gen list
```

## 输出格式

生成的 YAML 文件可直接被 `baseline/run.py` 执行：

```yaml
operator: rms_norm
description: RMS_NORM - Traced from llama_3.1_8b
source: traced

definition:
  const_axes:
    dtype: bf16
    eps: 1.0e-05
  var_axes:
    num_tokens: [1, 4, 8, 128, 512]
    hidden_size: [4096]

workloads:
  - name: llama_3.1_8b_decode_1
    num_tokens: 1
    hidden_size: 4096
    dtype: bf16
    eps: 1.0e-05
    phase: decode
    source: llama_3.1_8b/layer_input/decode
  ...
```

## 工作原理

### 推导逻辑

```
模型配置 → 架构类 → 单层算子列表
```

**示例：Llama-3.1-8B，decode phase，batch=4**

```python
# 输入
config.hidden_size = 4096
config.num_attention_heads = 32
config.intermediate_size = 14336
scenario.phase = "decode"
scenario.batch_size = 4

# 推导
1. RMSNorm: num_tokens=4, hidden_size=4096
2. QKV Proj (mm): M=4, K=4096, N=6144
3. RoPE: batch=4, seq_len=1, num_heads=32, head_dim=128
4. Attention Decode (paged_attention_decode): batch=4, ...
5. O Proj (mm): M=4, K=4096, N=4096
6. RMSNorm: num_tokens=4, hidden_size=4096
7. Gate Proj (mm): M=4, K=4096, N=14336
8. Up Proj (mm): M=4, K=4096, N=14336
9. SiLU and Mul: M=4, N=14336
10. Down Proj (mm): M=4, K=14336, N=4096
```

### 目录结构

```
workload_gen/
├── __init__.py
├── __main__.py          # python -m workload_gen 入口
├── cli.py               # CLI 命令行
├── config.py            # 数据类（ModelConfig, InferenceScenario, ...）
├── generator.py         # 核心生成器
├── exporter.py          # YAML 导出
└── architectures/       # 架构实现
    ├── __init__.py      # 架构注册表
    ├── base.py          # 基类
    └── llama.py         # Llama 架构实现

model_configs/           # 模型配置文件
├── llama_3.1_8b.json
├── qwen2.5_7b.json
└── README.md

baseline/cases/traced/   # 生成的 workload（gitignore）
├── llama_3.1_8b/
│   ├── rms_norm_llama_3.1_8b.yaml
│   ├── mm_llama_3.1_8b.yaml
│   ├── rope_llama_3.1_8b.yaml
│   ├── flash_attention_llama_3.1_8b.yaml
│   ├── paged_attention_decode_llama_3.1_8b.yaml
│   └── silu_and_mul_llama_3.1_8b.yaml
└── qwen2.5_7b/
    └── ...
```

## CLI 参考

### generate

生成 workload YAML 文件。

```bash
python -m workload_gen generate [OPTIONS]
```

**选项：**
- `--config`, `-c`: 模型配置文件路径（必须）
- `--output`, `-o`: 输出目录（默认：`baseline/cases/traced/`）
- `--scenarios`: 推理场景
  - `standard`: 使用标准场景（13 个）
  - 自定义格式：`"decode:1,4,8;prefill:128,512"`

**示例：**
```bash
# 标准场景
python -m workload_gen generate \
  --config model_configs/llama_3.1_8b.json \
  --output baseline/cases/traced/llama_3.1_8b/

# 自定义场景
python -m workload_gen generate \
  --config model_configs/qwen2.5_7b.json \
  --scenarios "decode:1,4;prefill:128,512" \
  --output baseline/cases/traced/qwen2.5_7b/
```

### list

列出支持的架构类型。

```bash
python -m workload_gen list
```

## 测试验证

生成 Llama-3.1-8B workload 并运行：

```bash
# 1. 生成 workload
python -m workload_gen generate \
  --config model_configs/llama_3.1_8b.json \
  --output baseline/cases/traced/llama_3.1_8b/

# 2. 运行 rms_norm（26 个 workload）
CUDA_VISIBLE_DEVICES=2 python baseline/run.py run \
  --backend nvidia \
  --case baseline/cases/traced/llama_3.1_8b/rms_norm_llama_3.1_8b.yaml

# 预期输出：26/26 workloads passed
```

## 扩展新模型

### 添加 Llama 系列模型

只需添加配置文件，无需修改代码：

```bash
# 1. 创建配置文件
cat > model_configs/yi_34b.json << EOF
{
  "model_name": "yi_34b",
  "model_type": "llama",
  "hidden_size": 7168,
  "num_attention_heads": 56,
  "num_key_value_heads": 8,
  "intermediate_size": 20480,
  ...
}
EOF

# 2. 生成 workload
python -m workload_gen generate \
  --config model_configs/yi_34b.json \
  --output baseline/cases/traced/yi_34b/
```

### 添加新架构

需要实现新的 Architecture 类（参考 `llama.py`）：

1. 创建 `workload_gen/architectures/new_arch.py`
2. 继承 `BaseArchitecture`
3. 实现 `generate_layer_workloads()` 方法
4. 在 `architectures/__init__.py` 注册

## 优势

1. **零部署依赖** — 不需要 GPU、权重、推理服务
2. **精确性** — shape 从模型结构确定性推导
3. **覆盖度** — 支持所有开源模型（包括无法部署的 405B/671B）
4. **可扩展** — 新增模型只需配置文件，新增架构只需一个类
5. **兼容性** — 生成的 YAML 直接可被现有 baseline 框架执行

## 参考

- 模型配置：`model_configs/README.md`
- 设计文档：`.plan`
- flashinfer-trace: https://huggingface.co/datasets/flashinfer/flashinfer-trace
