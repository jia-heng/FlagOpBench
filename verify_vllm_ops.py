#!/usr/bin/env python3
"""验证 vLLM CUDA ops 是否可用

编译完成后运行此脚本检查 vLLM CUDA kernel 注册情况。
"""

import sys
import torch

def check_vllm_ops():
    """检查 vLLM CUDA ops 可用性"""
    print("=" * 70)
    print("vLLM CUDA Ops 验证")
    print("=" * 70)

    # 1. 检查 PyTorch 和 CUDA 环境
    print("\n📋 环境信息:")
    print(f"  PyTorch 版本: {torch.__version__}")
    print(f"  CUDA 可用: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA 设备数量: {torch.cuda.device_count()}")
        print(f"  当前设备: {torch.cuda.get_device_name(0)}")

    # 2. 尝试导入 vLLM CUDA ops
    print("\n🔍 检查 vLLM CUDA ops:")
    try:
        import torch.ops._C as vllm_ops
        print("  ✅ torch.ops._C 模块可导入")

        # 列出所有可用的 ops
        ops_list = [attr for attr in dir(vllm_ops) if not attr.startswith('_')]
        print(f"\n  📦 发现 {len(ops_list)} 个 CUDA ops:")

        # 按类别分组显示
        categories = {
            'Normalization': ['rms_norm', 'fused_add_rms_norm'],
            'Attention': ['rotary_embedding', 'paged_attention', 'fused_kda_decode'],
            'MoE': ['cutlass_moe_mm', 'moe_align_block_size', 'topk_softmax'],
            'TopK': ['top_k_per_row_prefill', 'top_k_per_row_decode'],
            'Quantization': ['cutlass_scaled_mm', 'marlin_gemm', 'awq_marlin_repack'],
        }

        found_ops = {}
        for category, op_names in categories.items():
            found = [op for op in op_names if hasattr(vllm_ops, op)]
            if found:
                found_ops[category] = found

        for category, ops in found_ops.items():
            print(f"\n  {category}:")
            for op in ops:
                print(f"    ✓ {op}")

        # 列出其他未分类的 ops
        categorized = set(sum([ops for ops in found_ops.values()], []))
        other_ops = [op for op in ops_list if op not in categorized]
        if other_ops:
            print(f"\n  Other ops ({len(other_ops)}):")
            for op in sorted(other_ops)[:10]:  # 只显示前 10 个
                print(f"    • {op}")
            if len(other_ops) > 10:
                print(f"    ... and {len(other_ops) - 10} more")

        return True

    except ImportError as e:
        print(f"  ❌ 无法导入 torch.ops._C: {e}")
        return False
    except Exception as e:
        print(f"  ❌ 检查失败: {e}")
        return False

def test_rms_norm():
    """测试 rms_norm CUDA kernel"""
    print("\n" + "=" * 70)
    print("测试 rms_norm CUDA Kernel")
    print("=" * 70)

    try:
        import torch.ops._C as vllm_ops

        if not hasattr(vllm_ops, 'rms_norm'):
            print("  ⚠️  rms_norm op 不可用")
            return False

        # 准备测试数据
        M, hidden_size = 2048, 4096
        x = torch.randn(M, hidden_size, dtype=torch.float16, device='cuda')
        weight = torch.ones(hidden_size, dtype=torch.float16, device='cuda')
        out = torch.empty_like(x)
        eps = 1e-6

        # 调用 CUDA kernel
        print(f"\n  输入形状: {x.shape}")
        print(f"  数据类型: {x.dtype}")
        print(f"  调用 vllm_ops.rms_norm(out, x, weight, {eps})")

        vllm_ops.rms_norm(out, x, weight, eps)

        print(f"  输出形状: {out.shape}")
        print(f"  输出范围: [{out.min().item():.4f}, {out.max().item():.4f}]")
        print("  ✅ rms_norm CUDA kernel 工作正常!")
        return True

    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_topk():
    """测试 top_k CUDA kernel"""
    print("\n" + "=" * 70)
    print("测试 top_k_per_row_decode CUDA Kernel")
    print("=" * 70)

    try:
        import torch.ops._C as vllm_ops

        if not hasattr(vllm_ops, 'top_k_per_row_decode'):
            print("  ⚠️  top_k_per_row_decode op 不可用")
            return False

        # 准备测试数据
        num_tokens, num_experts = 128, 64
        k = 8
        x = torch.randn(num_tokens, num_experts, dtype=torch.float16, device='cuda')

        print(f"\n  输入形状: {x.shape}")
        print(f"  Top-K: {k}")
        print(f"  调用 vllm_ops.top_k_per_row_decode(x, {k})")

        values, indices = vllm_ops.top_k_per_row_decode(x, k)

        print(f"  输出 values 形状: {values.shape}")
        print(f"  输出 indices 形状: {indices.shape}")
        print("  ✅ top_k_per_row_decode CUDA kernel 工作正常!")
        return True

    except Exception as e:
        print(f"  ❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    # 检查 CUDA ops 可用性
    ops_available = check_vllm_ops()

    if not ops_available:
        print("\n" + "=" * 70)
        print("⚠️  vLLM CUDA ops 不可用")
        print("=" * 70)
        print("\n请先编译 vLLM:")
        print("  cd /data/jianheng/works/FlagOpBench/vllm")
        print("  pip install -e .")
        sys.exit(1)

    # 运行功能测试
    print("\n")
    test_passed = 0
    test_total = 2

    if test_rms_norm():
        test_passed += 1

    if test_topk():
        test_passed += 1

    # 总结
    print("\n" + "=" * 70)
    print(f"测试结果: {test_passed}/{test_total} 通过")
    print("=" * 70)

    if test_passed == test_total:
        print("✅ 所有测试通过! vLLM CUDA ops 已正确安装。")
        print("\n下一步:")
        print("  1. 运行基准测试: python baseline/run.py run --backend nvidia \\")
        print("       --case baseline/cases/basic/rms_norm.yaml --platform nvidia_h20")
        print("  2. 验证性能提升: 应该看到约 2x 性能提升")
        sys.exit(0)
    else:
        print("⚠️  部分测试失败，请检查 vLLM 安装。")
        sys.exit(1)

if __name__ == "__main__":
    main()
