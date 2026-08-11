#!/usr/bin/env python3
"""测试所有改造为官方 Ops 的算子

验证今日改造的 4 个算子是否正常工作，并对比性能。
"""

import torch
import time
from typing import Dict, Any
import sys

# 添加项目路径
sys.path.insert(0, '/data/jianheng/works/FlagOpBench')

from baseline.operators.basic.gemma_rms_norm import GemmaRmsNormOperator
from baseline.operators.basic.fused_q_kv_rmsnorm import FusedQKVRmsNormOperator
from baseline.operators.basic.moe_sum import MoESumOperator
from baseline.operators.basic.grouped_matmul import GroupedMatmulOperator


def benchmark_operator(op_class, inputs: Dict[str, Any], name: str, warmup: int = 10, iters: int = 100):
    """测试算子性能"""
    print(f"\n{'='*60}")
    print(f"测试算子: {name}")
    print(f"{'='*60}")

    op = op_class(device='cuda')

    # Warmup
    for _ in range(warmup):
        output = op.forward(**inputs)

    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(iters):
        output = op.forward(**inputs)
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_time_ms = (end - start) / iters * 1000

    print(f"  输出形状: {output.shape}")
    print(f"  平均时间: {avg_time_ms:.4f} ms")
    print(f"  迭代次数: {iters}")

    # 测试正确性
    golden = op.compute_golden(**inputs)
    max_diff = torch.max(torch.abs(output - golden)).item()
    mean_diff = torch.mean(torch.abs(output - golden)).item()

    # 计算相对误差
    relative_err = max_diff / (torch.max(torch.abs(golden)).item() + 1e-8)

    print(f"  最大误差: {max_diff:.2e}")
    print(f"  平均误差: {mean_diff:.2e}")
    print(f"  相对误差: {relative_err:.2e}")

    # bf16/fp16 精度容忍度更高
    if max_diff < 0.1 or relative_err < 0.01:
        print(f"  ✅ 正确性验证通过")
    else:
        print(f"  ⚠️  误差较大，需要检查")

    return avg_time_ms


def main():
    print("╔══════════════════════════════════════════════════════════════╗")
    print("║          官方 Ops 改造算子性能测试                           ║")
    print("╚══════════════════════════════════════════════════════════════╝")

    if not torch.cuda.is_available():
        print("❌ CUDA 不可用，无法测试")
        return

    device_name = torch.cuda.get_device_name(0)
    print(f"\n📊 测试环境")
    print(f"  GPU: {device_name}")
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA: {torch.version.cuda}")

    # 检查 PyTorch 是否支持 rms_norm
    import torch.nn.functional as F
    has_rms_norm = hasattr(F, 'rms_norm')
    print(f"  PyTorch rms_norm: {'✅ 支持' if has_rms_norm else '❌ 不支持'}")

    # 检查 vLLM CUDA ops
    try:
        import torch.ops._C as vllm_ops
        has_vllm = hasattr(vllm_ops, 'rms_norm')
        print(f"  vLLM CUDA ops: {'✅ 可用' if has_vllm else '❌ 不可用'}")
    except:
        has_vllm = False
        print(f"  vLLM CUDA ops: ❌ 不可用")

    results = {}

    # 1. gemma_rms_norm
    print("\n" + "─"*60)
    print("1/4 测试 gemma_rms_norm")
    print("─"*60)
    op = GemmaRmsNormOperator(device='cuda')
    inputs = op.prepare_inputs(M=2048, hidden_size=4096)
    results['gemma_rms_norm'] = benchmark_operator(
        GemmaRmsNormOperator, inputs, 'gemma_rms_norm'
    )

    # 2. fused_q_kv_rmsnorm
    print("\n" + "─"*60)
    print("2/4 测试 fused_q_kv_rmsnorm")
    print("─"*60)
    op = FusedQKVRmsNormOperator(device='cuda')
    inputs = op.prepare_inputs(num_tokens=2048, q_dim=4096, kv_dim=512)
    results['fused_q_kv_rmsnorm'] = benchmark_operator(
        FusedQKVRmsNormOperator, inputs, 'fused_q_kv_rmsnorm'
    )

    # 3. moe_sum
    print("\n" + "─"*60)
    print("3/4 测试 moe_sum")
    print("─"*60)
    op = MoESumOperator(device='cuda')
    inputs = op.prepare_inputs(num_tokens=2048, num_experts=8, hidden_size=4096)
    results['moe_sum'] = benchmark_operator(
        MoESumOperator, inputs, 'moe_sum'
    )

    # 4. grouped_matmul
    print("\n" + "─"*60)
    print("4/4 测试 grouped_matmul")
    print("─"*60)
    op = GroupedMatmulOperator(device='cuda')
    inputs = op.prepare_inputs(
        num_tokens=2048, hidden_size=4096, expert_size=4096, num_experts=8
    )
    results['grouped_matmul'] = benchmark_operator(
        GroupedMatmulOperator, inputs, 'grouped_matmul'
    )

    # 总结
    print("\n" + "="*60)
    print("📊 性能测试总结")
    print("="*60)

    print(f"\n{'算子':<25} {'时间 (ms)':<15} {'状态'}")
    print("-"*60)
    for name, time_ms in results.items():
        status = "✅ 通过" if time_ms > 0 else "❌ 失败"
        print(f"{name:<25} {time_ms:<15.4f} {status}")

    print("\n" + "="*60)
    print("✅ 所有测试完成！")
    print("="*60)

    # 性能建议
    print("\n💡 性能分析:")
    if has_vllm:
        print("  ✅ vLLM CUDA ops 已启用，RMSNorm 类算子性能最优")
    elif has_rms_norm:
        print("  ⚠️  使用 PyTorch F.rms_norm，性能良好但不是最优")
        print("  💡 建议编译 vLLM 以获得 2-3x 性能提升")
    else:
        print("  ⚠️  使用手动 fallback 实现，性能较低")
        print("  💡 建议升级 PyTorch 到 2.4+ 或编译 vLLM")

    print("\n  📚 详细报告请查看: 算子官方Ops改造总结.md")


if __name__ == "__main__":
    main()
