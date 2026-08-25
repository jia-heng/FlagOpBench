#!/bin/bash
# 从 HuggingFace 下载模型配置文件
# 需要在有代理的环境执行

set -e

MODELS=(
    "meta-llama/Llama-3.1-8B:llama_3.1_8b"
    "meta-llama/Llama-3.1-70B:llama_3.1_70b"
    "Qwen/Qwen2.5-7B:qwen2.5_7b"
    "Qwen/Qwen2.5-72B:qwen2.5_72b"
    "01-ai/Yi-1.5-34B:yi_1.5_34b"
)

OUTPUT_DIR="model_configs"
mkdir -p "$OUTPUT_DIR"

echo "🔽 Downloading model configs from HuggingFace..."
echo "=================================================="

for entry in "${MODELS[@]}"; do
    IFS=':' read -r model name <<< "$entry"
    echo ""
    echo "📦 $model → $name.json"

    url="https://huggingface.co/$model/raw/main/config.json"
    output="$OUTPUT_DIR/${name}.json"

    # 下载原始 config.json
    if wget -q -O "$output.tmp" "$url"; then
        # 添加 model_name 和 model_type 字段
        python3 -c "
import json
with open('$output.tmp') as f:
    data = json.load(f)

# 添加元信息
data['model_name'] = '$name'
data['model_type'] = 'llama'  # 默认 llama，后续手动调整

with open('$output', 'w') as f:
    json.dump(data, f, indent=2, ensure_ascii=False)
"
        rm "$output.tmp"
        echo "   ✓ Saved to $output"
    else
        echo "   ✗ Failed to download from $url"
    fi
done

echo ""
echo "✅ Done! All configs saved to $OUTPUT_DIR/"
echo "   Review and adjust 'model_type' field if needed."
