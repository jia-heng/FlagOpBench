#!/bin/bash
# 监控 vLLM 编译进度

LOG_FILE="/tmp/vllm_install.log"
TASK_OUTPUT="/tmp/claude-0/-data-jianheng-works-FlagOpBench/5db2bf1c-326e-4d8c-b380-c31bc0ff8990/tasks/bjird3xbm.output"

echo "========================================"
echo "vLLM 编译进度监控"
echo "========================================"
echo ""

# 检查进程是否还在运行
if ps aux | grep "pip install -e" | grep -v grep > /dev/null; then
    echo "✅ 编译进程正在运行"
    echo ""

    # 显示进程信息
    echo "📊 进程信息:"
    ps aux | grep "pip install" | grep -v grep | grep -v monitor | awk '{printf "  PID: %s, CPU: %s%%, MEM: %s%%, TIME: %s\n", $2, $3, $4, $10}'
    echo ""

    # 显示最新日志
    echo "📝 最新日志 (最后 20 行):"
    echo "----------------------------------------"
    if [ -f "$LOG_FILE" ]; then
        tail -20 "$LOG_FILE"
    else
        echo "  日志文件尚未创建"
    fi
    echo "----------------------------------------"
    echo ""

    # 显示任务输出
    echo "📄 后台任务输出:"
    echo "----------------------------------------"
    if [ -f "$TASK_OUTPUT" ]; then
        tail -20 "$TASK_OUTPUT"
    else
        echo "  任务输出文件尚未创建"
    fi
    echo "----------------------------------------"
    echo ""

    echo "⏳ 编译仍在进行中..."
    echo "💡 提示: 预计需要 10-20 分钟"
    echo ""
    echo "持续监控命令:"
    echo "  watch -n 10 './monitor_vllm_build.sh'"

else
    echo "⚠️  未检测到编译进程"
    echo ""
    echo "检查编译结果:"

    # 检查是否成功安装
    if python -c "import torch.ops._C" 2>/dev/null; then
        echo "  ✅ vLLM CUDA ops 可用!"
        echo ""
        echo "下一步: 运行验证脚本"
        echo "  python verify_vllm_ops.py"
    else
        echo "  ❌ vLLM CUDA ops 不可用"
        echo ""
        echo "查看完整日志:"
        echo "  cat $LOG_FILE"
        echo ""
        echo "重新编译:"
        echo "  cd /data/jianheng/works/FlagOpBench/vllm"
        echo "  pip install -e ."
    fi
fi

echo ""
echo "========================================"
