#!/usr/bin/env bash
# 摩尔机 NV 同款 FlagOS vs 基线：门禁 → compare → 出表
# 用法（容器内）:
#   export MUSA_VISIBLE_DEVICES=0,1,2,3
#   export GEMS_VENDOR=mthreads
#   cd /workspace/works/FlagOpBench-new   # 按实际路径改
#   bash scripts/run_mthreads_nv_table.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export MUSA_VISIBLE_DEVICES="${MUSA_VISIBLE_DEVICES:-0,1,2,3}"
export GEMS_VENDOR="${GEMS_VENDOR:-mthreads}"
WARMUP="${WARMUP:-10}"
REPEAT="${REPEAT:-100}"
PLATFORM=mthreads
OUT_ROOT="${OUT_ROOT:-results}"
XLSX_OUT="${XLSX_OUT:-results/国产_mthreads_测试结果.xlsx}"

echo "=== gate: musa ==="
python -c "import torch, torch_musa; print('musa', torch.musa.is_available(), torch.musa.get_device_name(0))"

echo "=== gate: flag_gems ==="
python -c "import flag_gems; print('flag_gems OK', getattr(flag_gems,'__file__',None))"

OPS=(
  "swiglu:cases/generated/merged/swiglu.yaml"
  "moe_sum:cases/generated/merged/moe_sum.yaml"
  "group_gemm:cases/generated/merged/group_gemm.yaml"
  "silu_and_mul_with_clamp:cases/demo/silu_and_mul_with_clamp.yaml"
)

# 第一阶段只跑 swiglu；设 RUN_PILOT4=1 跑齐 4 个
if [[ "${RUN_PILOT4:-0}" != "1" ]]; then
  OPS=("swiglu:cases/generated/merged/swiglu.yaml")
fi

for entry in "${OPS[@]}"; do
  op="${entry%%:*}"
  case_path="${entry#*:}"
  echo "=== compare $op ==="
  python run.py --platform "$PLATFORM" --mode compare \
    --case "$case_path" --output "$OUT_ROOT" \
    --warmup "$WARMUP" --repeat "$REPEAT"

  baseline_json="$OUT_ROOT/$op/${op}_mthreads.json"
  # provider 名可能是 mthreads 而非平台名；兼容探测
  if [[ ! -f "$baseline_json" ]]; then
    baseline_json=$(ls "$OUT_ROOT/$op/${op}_"*.json 2>/dev/null | grep -v flagos | grep -v compare | head -1 || true)
  fi
  flagos_json="$OUT_ROOT/$op/${op}_flagos_${PLATFORM}.json"
  echo "baseline=$baseline_json"
  echo "flagos=$flagos_json"
  python scripts/gen_compare_result.py --baseline "$baseline_json" --flagos "$flagos_json"
done

echo "=== emit xlsx ==="
python scripts/gen_nv_style_xlsx.py --compare-dir "$OUT_ROOT" --output "$XLSX_OUT"
echo "DONE: $XLSX_OUT"
