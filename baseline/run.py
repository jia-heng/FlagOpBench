"""FlagOpBench 性能基线平台 - 统一 CLI 入口"""

import argparse
import sys
import json
from pathlib import Path

# 确保项目根目录在 path 中
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def create_backend(backend_name: str, device_id: int = 0):
    """创建后端实例"""
    if backend_name == "nvidia":
        from baseline.backends.nvidia import NvidiaBackend
        return NvidiaBackend(device_id=device_id)
    elif backend_name == "ascend":
        # TODO: 昇腾后端
        raise NotImplementedError("Ascend backend not yet implemented")
    elif backend_name == "muxin":
        # TODO: 沐曦后端
        raise NotImplementedError("Muxin backend not yet implemented")
    else:
        raise ValueError(f"Unknown backend: {backend_name}. Available: nvidia, ascend, muxin")


def cmd_run(args):
    """执行基线测试"""
    from baseline.operators.registry import import_all_operators
    from baseline.framework.runner import BenchmarkRunner
    from baseline.framework.validator import Validator
    from baseline.framework.roofline import RooflineAnalyzer
    from baseline.framework.reporter import Reporter

    # 导入所有算子
    import_all_operators()

    # 创建后端
    backend = create_backend(args.backend, device_id=args.device)
    backend.setup()

    # 创建 Roofline 分析器
    roofline = None
    hw_specs_path = Path(__file__).parent / "hardware_specs.yaml"
    if hw_specs_path.exists() and args.platform:
        try:
            roofline = RooflineAnalyzer.from_yaml(str(hw_specs_path), args.platform)
        except (ValueError, KeyError) as e:
            print(f"  [WARN] Roofline disabled: {e}")

    # 创建执行引擎
    validator = Validator()
    runner = BenchmarkRunner(backend=backend, validator=validator, roofline=roofline)

    # 执行测试
    print(f"\n  Backend: {backend.name}")
    print(f"  Platform: {backend.platform_name}")
    print(f"  {'='*50}")

    results = []
    if args.case:
        for case_path in args.case:
            print(f"\n  Running case: {case_path}")
            results.extend(runner.run_case_file(case_path))
    elif args.case_dir:
        print(f"\n  Running case dir: {args.case_dir}")
        results.extend(runner.run_case_dir(args.case_dir))
    else:
        # 默认跑 cases/basic/
        default_dir = Path(__file__).parent / "cases" / "basic"
        if default_dir.exists():
            print(f"\n  Running default case dir: {default_dir}")
            results.extend(runner.run_case_dir(str(default_dir)))
        else:
            print("  No cases specified. Use --case or --case-dir.")
            return

    # 输出结果
    env_info = backend.collect_env_info()
    reporter = Reporter(
        platform=backend.platform_name,
        backend_name=backend.name,
        env_info=env_info,
    )
    reporter.add_results(results)
    reporter.print_summary()

    # 保存结果
    if args.output:
        saved_path = reporter.save(args.output)
        print(f"  Results saved to: {saved_path}")
    else:
        # 自动保存
        saved_path = reporter.save()
        print(f"  Results saved to: {saved_path}")

    backend.teardown()


def cmd_compare(args):
    """跨平台对比"""
    results_files = args.results
    all_data = []
    for f in results_files:
        with open(f, "r") as fp:
            all_data.append(json.load(fp))

    print(f"\n  Comparing {len(all_data)} result files...")
    print(f"  {'='*70}")
    print(f"  {'Operator':<25} {'Scenario':<30}", end="")
    for data in all_data:
        print(f" {data['backend']:<12}", end="")
    print()
    print(f"  {'-'*70}")

    # 收集所有 (operator, scenario) 对
    all_cases = set()
    for data in all_data:
        for r in data.get("results", []):
            if "error" not in r:
                all_cases.add((r["operator"], r["scenario"]))

    for op, scen in sorted(all_cases):
        print(f"  {op:<25} {scen:<30}", end="")
        for data in all_data:
            found = False
            for r in data.get("results", []):
                if r["operator"] == op and r["scenario"] == scen:
                    perf = r.get("performance", {})
                    t = perf.get("device_time", {}).get("mean_ms", 0)
                    print(f" {t:<12.4f}", end="")
                    found = True
                    break
            if not found:
                print(f" {'N/A':<12}", end="")
        print()


def cmd_regression(args):
    """回归检测"""
    with open(args.baseline, "r") as f:
        baseline_data = json.load(f)
    with open(args.current, "r") as f:
        current_data = json.load(f)

    threshold = args.threshold

    print(f"\n  Regression Detection (threshold: {threshold}%)")
    print(f"  Baseline: {args.baseline}")
    print(f"  Current:  {args.current}")
    print(f"  {'='*70}")

    baseline_map = {}
    for r in baseline_data.get("results", []):
        key = (r["operator"], r["scenario"])
        baseline_map[key] = r

    regressions = 0
    improvements = 0

    for r in current_data.get("results", []):
        key = (r["operator"], r["scenario"])
        if key not in baseline_map:
            continue

        b = baseline_map[key]
        b_time = b.get("performance", {}).get("device_time", {}).get("mean_ms", 0)
        c_time = r.get("performance", {}).get("device_time", {}).get("mean_ms", 0)

        if b_time == 0:
            continue

        delta_pct = (c_time - b_time) / b_time * 100

        if delta_pct > threshold:
            print(f"  [REGRESSION] {key[0]}/{key[1]}: "
                  f"{b_time:.4f}ms → {c_time:.4f}ms (+{delta_pct:.1f}%)")
            regressions += 1
        elif delta_pct < -threshold:
            print(f"  [IMPROVED]   {key[0]}/{key[1]}: "
                  f"{b_time:.4f}ms → {c_time:.4f}ms ({delta_pct:.1f}%)")
            improvements += 1
        else:
            print(f"  [STABLE]     {key[0]}/{key[1]}: "
                  f"{b_time:.4f}ms → {c_time:.4f}ms ({delta_pct:+.1f}%)")

    print(f"\n  Summary: {regressions} regressions, {improvements} improvements")


def cmd_list(args):
    """列出已注册的算子"""
    from baseline.operators.registry import import_all_operators, list_operators
    import_all_operators()
    operators = list_operators()
    print(f"\n  Registered operators ({len(operators)}):")
    for op in operators:
        print(f"    - {op}")


def main():
    parser = argparse.ArgumentParser(
        description="FlagOpBench - 性能基线测试平台",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # 单算子测试
  python run.py run --backend nvidia --case cases/basic/mm.yaml

  # 某一类算子
  python run.py run --backend nvidia --case-dir cases/basic/

  # 全量测试
  python run.py run --backend nvidia --case-dir cases/ --output results/nvidia_h20.json

  # 跨平台对比
  python run.py compare --results results/nvidia_h20.json results/ascend_910b.json

  # 回归检测
  python run.py regression --baseline results/v1.json --current results/v2.json --threshold 5

  # 列出已注册算子
  python run.py list
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # run 子命令
    run_parser = subparsers.add_parser("run", help="Run benchmark")
    run_parser.add_argument("--backend", "-b", required=True,
                           choices=["nvidia", "ascend", "muxin"],
                           help="Backend to use")
    run_parser.add_argument("--case", "-c", nargs="+",
                           help="Case file(s) to run")
    run_parser.add_argument("--case-dir", "-d",
                           help="Case directory to run")
    run_parser.add_argument("--output", "-o",
                           help="Output JSON file path")
    run_parser.add_argument("--device", type=int, default=0,
                           help="GPU device ID (default: 0)")
    run_parser.add_argument("--platform", "-p",
                           help="Platform key for roofline (e.g., nvidia_h20)")
    run_parser.set_defaults(func=cmd_run)

    # compare 子命令
    compare_parser = subparsers.add_parser("compare", help="Compare results")
    compare_parser.add_argument("--results", "-r", nargs="+", required=True,
                               help="Result files to compare")
    compare_parser.set_defaults(func=cmd_compare)

    # regression 子命令
    reg_parser = subparsers.add_parser("regression", help="Regression detection")
    reg_parser.add_argument("--baseline", required=True,
                           help="Baseline result file")
    reg_parser.add_argument("--current", required=True,
                           help="Current result file")
    reg_parser.add_argument("--threshold", type=float, default=5.0,
                           help="Regression threshold percentage (default: 5.0)")
    reg_parser.set_defaults(func=cmd_regression)

    # list 子命令
    list_parser = subparsers.add_parser("list", help="List registered operators")
    list_parser.set_defaults(func=cmd_list)

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
