#!/usr/bin/env python3
"""FlagOpBench - 单算子性能测试CLI

Usage:
    # NV平台最优基线
    python run.py --platform nvidia --case cases/demo/swiglu.yaml

    # FlagOS在NV平台的性能
    python run.py --platform nvidia --impl flagos --case cases/demo/swiglu.yaml

    # 对比模式: FlagOS vs 平台基线
    python run.py --platform nvidia --mode compare --case cases/demo/swiglu.yaml

    # 批量运行目录
    python run.py --platform nvidia --case-dir cases/demo/

    # 兼容旧用法
    python run.py --provider vllm --case cases/demo/swiglu.yaml
    python run.py --provider flagos --case cases/demo/swiglu.yaml
"""
import argparse
import sys
from pathlib import Path

# 确保项目根目录在path中
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from framework.registry import import_all_operators
from framework.runner import Runner
from framework.reporter import Reporter
from framework.timer import create_timer
import providers  # 触发所有Provider注册
from providers.registry import get_provider, list_providers


def _resolve_provider_args(args):
    """解析provider参数，兼容新旧两种用法

    新用法: --platform nvidia [--impl flagos]
    旧用法: --provider vllm / --provider flagos
    """
    if args.provider:
        # 旧用法兼容
        if args.provider == "vllm":
            return "nvidia", None
        elif args.provider == "flagos":
            return "nvidia", "flagos"
        else:
            return "nvidia", args.provider
    else:
        return args.platform, args.impl


def run_single(platform, impl, case_path, case_dir, warmup, repeat, output):
    """单一模式: 跑一个provider"""
    provider = get_provider(platform, impl)
    timer = create_timer(platform, warmup=warmup, repeat=repeat)
    runner = Runner(provider, warmup=warmup, repeat=repeat, timer=timer)

    results = []
    if case_path:
        print(f"  Case file: {case_path}")
        results.extend(runner.run_case_file(case_path))
    elif case_dir:
        print(f"  Case directory: {case_dir}")
        results.extend(runner.run_case_dir(case_dir))

    if not results:
        print("\n  No results generated. Check if cases are valid.")
        return

    # 生成报告
    print(f"\n[4/4] Generating report...")
    reporter = Reporter(provider_name=provider.name, platform=platform)
    reporter.add_results(results)
    reporter.print_summary()

    output_path = reporter.save(output_dir=output)
    print(f"\n  Results saved to: {output_path}")

    provider.teardown()


def run_compare(platform, case_path, case_dir, warmup, repeat, output):
    """对比模式: 平台基线 vs FlagOS"""
    baseline_provider = get_provider(platform, impl=None)
    flagos_provider = get_provider(platform, impl="flagos")

    timer = create_timer(platform, warmup=warmup, repeat=repeat)
    runner = Runner(baseline_provider, warmup=warmup, repeat=repeat, timer=timer)

    if case_path:
        print(f"  Case file: {case_path}")
        compare_results = runner.run_compare(case_path, baseline_provider, flagos_provider)
    elif case_dir:
        # 对比模式下逐文件对比
        case_p = Path(case_dir)
        yaml_files = sorted(case_p.rglob("*.yaml"))
        yaml_files = [f for f in yaml_files if f.name != "_template.yaml"]
        compare_results = []
        for f in yaml_files:
            print(f"\n  Case: {f.relative_to(case_p)}")
            compare_results.extend(runner.run_compare(str(f), baseline_provider, flagos_provider))
    else:
        print("  No case specified.")
        return

    if not compare_results:
        print("\n  No comparison results generated.")
        return

    # 打印对比摘要
    print(f"\n{'='*70}")
    print(f"  Compare: {baseline_provider.name} (baseline) vs flagos")
    print(f"  Platform: {platform}")
    print(f"{'='*70}")
    print(f"  {'Operator':<20} {'Workload':<30} {'Baseline(ms)':<14} {'FlagOS(ms)':<12} {'Speedup':<8}")
    print(f"  {'-'*84}")
    for cr in compare_results:
        marker = "✓" if cr.speedup > 1.05 else ("✗" if cr.speedup < 0.95 else "≈")
        print(
            f"  {cr.operator:<20} "
            f"{cr.workload:<30} "
            f"{cr.baseline.timing.mean_ms:<14.4f} "
            f"{cr.flagos.timing.mean_ms:<12.4f} "
            f"{cr.speedup:<7.3f}x {marker}"
        )

    faster = sum(1 for cr in compare_results if cr.speedup > 1.05)
    slower = sum(1 for cr in compare_results if cr.speedup < 0.95)
    par = len(compare_results) - faster - slower
    print(f"\n  Summary: {faster} faster, {slower} slower, {par} on par")
    print(f"{'='*70}")

    baseline_provider.teardown()
    flagos_provider.teardown()


def main():
    parser = argparse.ArgumentParser(description="FlagOpBench - 单算子性能测试")

    # 新用法参数
    parser.add_argument(
        "--platform",
        type=str,
        default="nvidia",
        choices=["nvidia", "ascend", "metax", "mthreads", "iluvatar"],
        help="目标平台 (default: nvidia)",
    )
    parser.add_argument(
        "--impl",
        type=str,
        default=None,
        help="指定实现: 默认使用平台最优基线，'flagos'使用FlagOS实现",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="single",
        choices=["single", "compare"],
        help="运行模式: single(默认) / compare(对比FlagOS vs 基线)",
    )

    # 兼容旧用法
    parser.add_argument(
        "--provider",
        type=str,
        default=None,
        help="[兼容旧用法] 等价于 --impl (vllm→平台基线, flagos→FlagOS)",
    )

    # 通用参数
    parser.add_argument("--case", type=str, help="Case file (yaml)")
    parser.add_argument("--case-dir", type=str, help="Case directory")
    parser.add_argument("--output", type=str, default=None, help="Output directory")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations (default: 10)")
    parser.add_argument("--repeat", type=int, default=100, help="Repeat iterations (default: 100)")

    args = parser.parse_args()

    if not args.case and not args.case_dir:
        parser.error("Must specify --case or --case-dir")

    # 解析provider参数
    platform, impl = _resolve_provider_args(args)

    # 导入所有算子
    print("\n[1/4] Importing operators...")
    import operators  # 触发自动导入
    import_all_operators()

    # 确定模式
    mode = args.mode
    if mode == "compare" and impl is not None and impl != "flagos":
        print(f"  [WARN] --mode compare 忽略 --impl '{impl}'，将对比平台基线 vs flagos")

    print(f"\n[2/4] Platform: {platform} | Impl: {impl or 'baseline'} | Mode: {mode}")

    # 执行
    print(f"\n[3/4] Running benchmarks...")
    if mode == "compare":
        run_compare(platform, args.case, args.case_dir, args.warmup, args.repeat, args.output)
    else:
        run_single(platform, impl, args.case, args.case_dir, args.warmup, args.repeat, args.output)


if __name__ == "__main__":
    main()
