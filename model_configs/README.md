# Model Configurations

本目录维护主流开源 LLM 的配置文件，用于生成真实推理 workload。

## 已支持模型

| 模型 | 配置文件 | 架构类型 | 参数量 |
|------|---------|---------|--------|
| Llama-3.1-8B | `llama_3.1_8b.json` | llama | 8B |
| Qwen2.5-7B | `qwen2.5_7b.json` | llama | 7B |

## 配置文件格式

```json
{
  "model_name": "模型标识",
  "model_type": "架构类型 (llama/mixtral/deepseek_v3/...)",
  "hidden_size": 隐藏层维度,
  "num_attention_heads": 注意力头数,
  "num_key_value_heads": KV 头数 (GQA),
  "intermediate_size": FFN 中间层维度,
  "num_hidden_layers": 层数,
  "vocab_size": 词表大小,
  "rope_theta": RoPE 基数,
  "max_position_embeddings": 最大位置编码,
  "rms_norm_eps": RMSNorm epsilon
}
```

## 获取配置

### 方法 1：使用下载脚本（需要代理）

```bash
bash scripts/download_model_configs.sh
```

### 方法 2：手动从 HuggingFace 复制

访问模型主页，下载 `config.json`：
- https://huggingface.co/meta-llama/Llama-3.1-8B/blob/main/config.json
- https://huggingface.co/Qwen/Qwen2.5-7B/blob/main/config.json

保存到本目录后，添加 `model_name` 和 `model_type` 字段。

## 使用

```bash
# 从配置生成 workload
python -m workload_gen generate --config model_configs/llama_3.1_8b.json
```
