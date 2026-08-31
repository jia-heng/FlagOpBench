#!/bin/bash
# 批量跑所有算子的性能测试
# Usage:
#   ./run_all.sh                          # 默认: nvidia平台, baseline + flagos
#   ./run_all.sh --platform nvidia        # 指定平台
#   ./run_all.sh --platform ascend        # 昇腾平台
#   ./run_all.sh --mode baseline          # 只跑基线
#   ./run_all.sh --mode flagos            # 只跑flagos
#   ./run_all.sh --mode compare           # 对比模式

cd "$(dirname "$0")"

PLATFORM="${PLATFORM:-nvidia}"
MODE="${MODE:-all}"  # all / baseline / flagos / compare

# 解析参数
while [[ $# -gt 0 ]]; do
    case $1 in
        --platform) PLATFORM="$2"; shift 2 ;;
        --mode) MODE="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

CASES=$(ls cases/demo/*.yaml 2>/dev/null | sort)
if [ -z "$CASES" ]; then
    echo "No cases found in cases/demo/"
    exit 1
fi

LOGFILE="results/run_all_${PLATFORM}.log"
mkdir -p results

echo "=== Flagtests Batch Run $(date) ===" > "$LOGFILE"
echo "  Platform: $PLATFORM" | tee -a "$LOGFILE"
echo "  Mode: $MODE" | tee -a "$LOGFILE"
echo "" | tee -a "$LOGFILE"

for CASE in $CASES; do
    OP=$(basename "$CASE" .yaml)
    echo "---------- $OP ----------" | tee -a "$LOGFILE"

    if [ "$MODE" = "compare" ]; then
        # 对比模式
        echo "[$(date +%H:%M:%S)] Running $OP / compare ..." | tee -a "$LOGFILE"
        timeout 360 python run.py --platform "$PLATFORM" --mode compare --case "$CASE" >> "$LOGFILE" 2>&1
        RC=$?
        if [ $RC -eq 0 ]; then
            echo "  compare: OK" | tee -a "$LOGFILE"
        elif [ $RC -eq 124 ]; then
            echo "  compare: TIMEOUT" | tee -a "$LOGFILE"
        else
            echo "  compare: FAILED (rc=$RC)" | tee -a "$LOGFILE"
        fi
    else
        # baseline
        if [ "$MODE" = "all" ] || [ "$MODE" = "baseline" ]; then
            echo "[$(date +%H:%M:%S)] Running $OP / baseline ($PLATFORM) ..." | tee -a "$LOGFILE"
            timeout 180 python run.py --platform "$PLATFORM" --case "$CASE" >> "$LOGFILE" 2>&1
            RC=$?
            if [ $RC -eq 0 ]; then
                echo "  baseline: OK" | tee -a "$LOGFILE"
            elif [ $RC -eq 124 ]; then
                echo "  baseline: TIMEOUT" | tee -a "$LOGFILE"
            else
                echo "  baseline: FAILED (rc=$RC)" | tee -a "$LOGFILE"
            fi
        fi

        # flagos
        if [ "$MODE" = "all" ] || [ "$MODE" = "flagos" ]; then
            echo "[$(date +%H:%M:%S)] Running $OP / flagos ..." | tee -a "$LOGFILE"
            timeout 180 python run.py --platform "$PLATFORM" --impl flagos --case "$CASE" >> "$LOGFILE" 2>&1
            RC=$?
            if [ $RC -eq 0 ]; then
                echo "  flagos: OK" | tee -a "$LOGFILE"
            elif [ $RC -eq 124 ]; then
                echo "  flagos: TIMEOUT" | tee -a "$LOGFILE"
            else
                echo "  flagos: FAILED (rc=$RC)" | tee -a "$LOGFILE"
            fi
        fi
    fi
done

echo "" | tee -a "$LOGFILE"
echo "=== Done $(date) ===" | tee -a "$LOGFILE"
echo "Results in results/"
ls results/*.json 2>/dev/null | wc -l
