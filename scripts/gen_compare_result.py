#!/usr/bin/env python3
"""从两份性能结果JSON生成对比结果JSON。

Usage:
    python scripts/gen_compare_result.py \
        --baseline results/swiglu_nvidia.json \
        --flagos results/swiglu_flagos.json \
        [--output results/swiglu_compare.json]

如果不指定 --output，默认在 results/ 下生成 {op}_compare.json。
"""
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime


def load_results(path):
    """加载结果JSON，返回 (metadata, environment, results_dict)"""
    with open(path) as f:
        data = json.load(f)

    # 以 (operator, workload) 为 key 建立索引
    results_map = {}
    for r in data["results"]:
        key = (r["operator"], r["workload"])
        results_map[key] = r

    return data.get("metadata", {}), data.get("environment", {}), results_map


def compute_speedup(baseline_ms, flagos_ms):
    """计算 speedup = baseline / flagos (>1 表示 flagos 更快)"""
    if flagos_ms <= 0:
        return float("inf")
    return baseline_ms / flagos_ms


def gen_compare(baseline_path, flagos_path, output_path=None):
    baseline_meta, baseline_env, baseline_results = load_results(baseline_path)
    flagos_meta, flagos_env, flagos_results = load_results(flagos_path)

    # 匹配 workload
    all_keys = set(baseline_results.keys()) | set(flagos_results.keys())
    matched_keys = set(baseline_results.keys()) & set(flagos_results.keys())
    baseline_only = set(baseline_results.keys()) - set(flagos_results.keys())
    flagos_only = set(flagos_results.keys()) - set(baseline_results.keys())

    if baseline_only:
        print(f"  [WARN] {len(baseline_only)} workloads only in baseline, skipped", file=sys.stderr)
    if flagos_only:
        print(f"  [WARN] {len(flagos_only)} workloads only in flagos, skipped", file=sys.stderr)

    # 按原始顺序输出（以baseline文件的顺序为准）
    with open(baseline_path) as f:
        baseline_data = json.load(f)
    ordered_keys = [(r["operator"], r["workload"]) for r in baseline_data["results"]]

    comparisons = []
    for key in ordered_keys:
        if key not in matched_keys:
            continue

        br = baseline_results[key]
        fr = flagos_results[key]

        b_mean = br["performance"]["device_time"]["mean_ms"]
        f_mean = fr["performance"]["device_time"]["mean_ms"]
        speedup = compute_speedup(b_mean, f_mean)

        if speedup > 1.05:
            verdict = "faster"
        elif speedup < 0.95:
            verdict = "slower"
        else:
            verdict = "on_par"

        comparisons.append({
            "operator": key[0],
            "workload": key[1],
            "parameters": br.get("parameters", {}),
            "baseline": {
                "provider": br.get("provider", baseline_meta.get("provider", "unknown")),
                "mean_ms": b_mean,
                "std_ms": br["performance"]["device_time"]["std_ms"],
                "min_ms": br["performance"]["device_time"]["min_ms"],
                "max_ms": br["performance"]["device_time"]["max_ms"],
                "bandwidth_gb_s": br["performance"]["throughput"]["bandwidth_gb_s"],
                "gflops": br["performance"]["throughput"]["gflops"],
                "impl_source": br.get("impl_info", {}).get("source", ""),
            },
            "flagos": {
                "provider": fr.get("provider", "flagos"),
                "mean_ms": f_mean,
                "std_ms": fr["performance"]["device_time"]["std_ms"],
                "min_ms": fr["performance"]["device_time"]["min_ms"],
                "max_ms": fr["performance"]["device_time"]["max_ms"],
                "bandwidth_gb_s": fr["performance"]["throughput"]["bandwidth_gb_s"],
                "gflops": fr["performance"]["throughput"]["gflops"],
                "impl_source": fr.get("impl_info", {}).get("source", ""),
            },
            "speedup": round(speedup, 4),
            "verdict": verdict,
        })

    # 汇总统计
    faster_count = sum(1 for c in comparisons if c["verdict"] == "faster")
    slower_count = sum(1 for c in comparisons if c["verdict"] == "slower")
    on_par_count = sum(1 for c in comparisons if c["verdict"] == "on_par")
    avg_speedup = (
        sum(c["speedup"] for c in comparisons) / len(comparisons)
        if comparisons
        else 0
    )

    # 构造输出
    output = {
        "metadata": {
            "timestamp": datetime.now().isoformat(),
            "type": "compare",
            "platform": baseline_meta.get("platform", "unknown"),
            "baseline_provider": baseline_meta.get("provider", "unknown"),
            "flagos_provider": flagos_meta.get("provider", "flagos"),
            "num_workloads": len(comparisons),
            "baseline_file": str(baseline_path),
            "flagos_file": str(flagos_path),
        },
        "environment": baseline_env,
        "summary": {
            "total": len(comparisons),
            "faster": faster_count,
            "slower": slower_count,
            "on_par": on_par_count,
            "avg_speedup": round(avg_speedup, 4),
        },
        "comparisons": comparisons,
    }

    # 决定输出路径
    if output_path is None:
        # 从 baseline 文件名推断算子名和平台
        # e.g. results/swiglu/swiglu_nvidia.json → op=swiglu, platform=nvidia
        baseline_name = Path(baseline_path).stem  # swiglu_nvidia
        op_name = baseline_name.rsplit("_", 1)[0]  # swiglu
        platform = baseline_meta.get("platform", "unknown")

        # 输出到 results/{op}/{op}_compare_{platform}.json
        results_root = Path(baseline_path).parent.parent
        op_dir = results_root / op_name
        op_dir.mkdir(parents=True, exist_ok=True)
        output_path = op_dir / f"{op_name}_compare_{platform}.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n  Compare result saved to: {output_path}")
    print(f"  Total: {len(comparisons)} | Faster: {faster_count} | Slower: {slower_count} | On par: {on_par_count}")
    print(f"  Avg speedup: {avg_speedup:.4f}x")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="从两份性能结果JSON生成对比结果JSON"
    )
    parser.add_argument(
        "--baseline", required=True, help="基线结果JSON (e.g. results/swiglu_nvidia.json)"
    )
    parser.add_argument(
        "--flagos", required=True, help="FlagOS结果JSON (e.g. results/swiglu_flagos.json)"
    )
    parser.add_argument(
        "--output", "-o", default=None, help="输出路径 (默认: results/{op}_compare.json)"
    )
    args = parser.parse_args()

    if not Path(args.baseline).exists():
        print(f"  [ERROR] Baseline file not found: {args.baseline}", file=sys.stderr)
        sys.exit(1)
    if not Path(args.flagos).exists():
        print(f"  [ERROR] FlagOS file not found: {args.flagos}", file=sys.stderr)
        sys.exit(1)

    gen_compare(args.baseline, args.flagos, args.output)


if __name__ == "__main__":
    main()
