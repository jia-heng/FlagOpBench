#!/usr/bin/env python3
"""对比两个 provider 的 benchmark 结果，生成 speedup 报告。

Usage:
    python compare.py --op mhc_pre
    python compare.py --op flash_mla --baseline vllm --target flagos
    python compare.py --op flash_mla --baseline nvidia --target flagos
    python compare.py --all
    python compare.py --all --platform ascend --baseline ascend --target flagos
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

RESULTS_DIR = Path(__file__).parent / "results"


def load_results(op_name: str, provider: str) -> Dict[str, dict]:
    """加载某算子某 provider 的结果，返回 {workload_name: result_dict}"""
    filepath = RESULTS_DIR / f"{op_name}_{provider}.json"
    if not filepath.exists():
        return {}
    with open(filepath) as f:
        data = json.load(f)
    results = {}
    for r in data.get("results", []):
        results[r["workload"]] = r
    return results


def compare_operator(op_name: str, baseline: str = "vllm", target: str = "flagos") -> List[dict]:
    """对比一个算子的两个 provider 结果"""
    baseline_results = load_results(op_name, baseline)
    target_results = load_results(op_name, target)

    if not baseline_results:
        print(f"  [SKIP] No {baseline} results for {op_name}", file=sys.stderr)
        return []
    if not target_results:
        print(f"  [SKIP] No {target} results for {op_name}", file=sys.stderr)
        return []

    comparisons = []
    for wl_name in target_results:
        if wl_name not in baseline_results:
            continue
        t_perf = target_results[wl_name]["performance"]["device_time"]
        b_perf = baseline_results[wl_name]["performance"]["device_time"]

        t_ms = t_perf["mean_ms"]
        b_ms = b_perf["mean_ms"]
        speedup = b_ms / t_ms if t_ms > 0 else float("inf")

        comparisons.append({
            "operator": op_name,
            "workload": wl_name,
            f"{target}_ms": t_ms,
            f"{baseline}_ms": b_ms,
            "speedup": round(speedup, 3),
            "status": "faster" if speedup > 1.05 else ("slower" if speedup < 0.95 else "par"),
        })

    return comparisons


def print_comparison_table(comparisons: List[dict], baseline: str, target: str):
    """打印对比表格"""
    if not comparisons:
        print("No comparisons available.")
        return

    # 按算子分组
    from collections import defaultdict
    by_op = defaultdict(list)
    for c in comparisons:
        by_op[c["operator"]].append(c)

    print(f"\n{'='*80}")
    print(f"  Performance Comparison: {target} vs {baseline}")
    print(f"  Speedup > 1.0 means {target} is faster")
    print(f"{'='*80}")

    total_faster = sum(1 for c in comparisons if c["status"] == "faster")
    total_slower = sum(1 for c in comparisons if c["status"] == "slower")
    total_par = sum(1 for c in comparisons if c["status"] == "par")

    for op_name, items in sorted(by_op.items()):
        print(f"\n  [{op_name}]")
        print(f"  {'Workload':<45} {target:>10} {baseline:>10} {'Speedup':>8}")
        print(f"  {'-'*75}")
        for c in items:
            marker = "✓" if c["status"] == "faster" else ("✗" if c["status"] == "slower" else "≈")
            print(
                f"  {c['workload']:<45} "
                f"{c[f'{target}_ms']:>9.4f}ms "
                f"{c[f'{baseline}_ms']:>9.4f}ms "
                f"{c['speedup']:>7.3f}x {marker}"
            )

        # 算子级汇总
        avg_speedup = sum(c["speedup"] for c in items) / len(items)
        geo_speedup = pow(
            eval("*".join(str(c["speedup"]) for c in items)),
            1.0 / len(items)
        )
        print(f"  {'':>45} {'Avg':>10} {'Geo-Mean':>10} {avg_speedup:>7.3f}x  {geo_speedup:.3f}x")

    print(f"\n  Summary: {total_faster} faster, {total_slower} slower, {total_par} on par")
    print(f"{'='*80}\n")


def save_comparison(comparisons: List[dict], baseline: str, target: str, output_path: Path = None):
    """保存对比结果为 JSON"""
    if output_path is None:
        output_path = RESULTS_DIR / f"compare_{target}_vs_{baseline}.json"

    report = {
        "baseline": baseline,
        "target": target,
        "num_workloads": len(comparisons),
        "summary": {
            "faster": sum(1 for c in comparisons if c["status"] == "faster"),
            "slower": sum(1 for c in comparisons if c["status"] == "slower"),
            "par": sum(1 for c in comparisons if c["status"] == "par"),
        },
        "comparisons": comparisons,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"  Comparison saved to: {output_path}", file=sys.stderr)


def main():
    parser = argparse.ArgumentParser(description="Compare benchmark results between providers")
    parser.add_argument("--op", type=str, help="Operator name to compare")
    parser.add_argument("--all", action="store_true", help="Compare all available operators")
    parser.add_argument("--platform", type=str, default="nvidia",
                        choices=["nvidia", "ascend", "metax", "mthreads", "iluvatar"],
                        help="Platform (default: nvidia, affects default baseline)")
    parser.add_argument("--baseline", type=str, default=None,
                        help="Baseline provider (default: platform name, e.g. nvidia/vllm)")
    parser.add_argument("--target", type=str, default="flagos", help="Target provider (default: flagos)")
    parser.add_argument("--save", action="store_true", help="Save comparison JSON")
    args = parser.parse_args()

    if not args.op and not args.all:
        parser.error("Must specify --op or --all")

    # baseline默认: 平台名或vllm（向后兼容）
    if args.baseline is None:
        # 优先查找平台名，fallback到vllm（兼容旧结果文件）
        baseline = args.platform
        # 检查是否有对应文件，如果没有尝试vllm
        if args.op and not (RESULTS_DIR / f"{args.op}_{baseline}.json").exists():
            if (RESULTS_DIR / f"{args.op}_vllm.json").exists():
                baseline = "vllm"
    else:
        baseline = args.baseline
    target = args.target

    if args.all:
        # 找所有同时有 baseline 和 target 结果的算子
        ops = set()
        for f in RESULTS_DIR.glob(f"*_{target}.json"):
            op_name = f.stem.replace(f"_{target}", "")
            if (RESULTS_DIR / f"{op_name}_{baseline}.json").exists():
                ops.add(op_name)
        ops = sorted(ops)
    else:
        ops = [args.op]

    all_comparisons = []
    for op_name in ops:
        comparisons = compare_operator(op_name, baseline, target)
        all_comparisons.extend(comparisons)

    print_comparison_table(all_comparisons, baseline, target)

    if args.save:
        if args.op and not args.all:
            # 单算子对比，输出 {op}_compare.json
            output_path = RESULTS_DIR / f"{args.op}_compare.json"
        else:
            output_path = None  # 使用默认 compare_{target}_vs_{baseline}.json
        save_comparison(all_comparisons, baseline, target, output_path=output_path)


if __name__ == "__main__":
    main()
