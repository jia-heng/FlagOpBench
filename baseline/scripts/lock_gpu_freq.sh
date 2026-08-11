#!/bin/bash
# NVIDIA GPU 锁频脚本
# 用途：锁定 GPU 频率以获得稳定的 benchmark 结果
# 使用：sudo bash lock_gpu_freq.sh [lock|unlock]

set -e

ACTION=${1:-lock}

get_max_freq() {
    nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -1
}

case $ACTION in
    lock)
        MAX_FREQ=$(get_max_freq)
        echo "Locking GPU frequency to ${MAX_FREQ} MHz..."
        sudo nvidia-smi -lgc ${MAX_FREQ},${MAX_FREQ}
        echo "GPU frequency locked."
        nvidia-smi --query-gpu=clocks.current.graphics,clocks.max.graphics --format=csv
        ;;
    unlock)
        echo "Resetting GPU frequency to default..."
        sudo nvidia-smi -rgc
        echo "GPU frequency unlocked."
        ;;
    status)
        echo "Current GPU clock status:"
        nvidia-smi --query-gpu=clocks.current.graphics,clocks.max.graphics,clocks.current.memory --format=csv
        ;;
    *)
        echo "Usage: $0 [lock|unlock|status]"
        exit 1
        ;;
esac
