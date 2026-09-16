#!/usr/bin/env python3
"""从两份性能结果JSON生成对比结果JSON。

Usage:
    # 指定两个文件
    python scripts/gen_compare_result.py \
        --baseline results/swiglu/swiglu_nvidia.json \
        --flagos results/swiglu/swiglu_flagos_nvidia.json \
        [--output results/swiglu/swiglu_compare_nvidia.json]

    # 自动扫描 results 目录，为所有有 baseline+flagos 的算子生成对比
    python scripts/gen_compare_result.py --auto [--results-dir results/]

如果不指定 --output，默认在 baseline 同目录下生成 {op}_compare_{platform}.json。
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
    baseline_path = Path(baseline_path)
    flagos_path = Path(flagos_path)

    baseline_meta, baseline_env, baseline_results = load_results(baseline_path)
    flagos_meta, flagos_env, flagos_results = load_results(flagos_path)

    matched_keys = set(baseline_results.keys()) & set(flagos_results.keys())
    baseline_only = set(baseline_results.keys()) - set(flagos_results.keys())
    flagos_only = set(flagos_results.keys()) - set(baseline_results.keys())

    if baseline_only:
        print(f"  [WARN] {len(baseline_only)} workloads only in baseline, skipped", file=sys.stderr)
    if flagos_only:
        print(f"  [WARN] {len(flagos_only)} workloads only in flagos, skipped", file=sys.stderr)

    # 按 baseline 文件中的原始顺序输出
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

    faster_count = sum(1 for c in comparisons if c["verdict"] == "faster")
    slower_count = sum(1 for c in comparisons if c["verdict"] == "slower")
    on_par_count = sum(1 for c in comparisons if c["verdict"] == "on_par")
    avg_speedup = (
        sum(c["speedup"] for c in comparisons) / len(comparisons)
        if comparisons else 0
    )

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

    if output_path is None:
        # 输出到 baseline 同目录: {op}_compare_{platform}.json
        platform = baseline_meta.get("platform", "nvidia")
        stem = baseline_path.stem  # e.g. swiglu_nvidia
        if stem.endswith(f"_{platform}"):
            op_name = stem[: -(len(platform) + 1)]
        else:
            op_name = stem.rsplit("_", 1)[0]
        output_path = baseline_path.parent / f"{op_name}_compare_{platform}.json"

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"  {output_path.name}: {len(comparisons)} workloads | "
          f"faster={faster_count} slower={slower_count} par={on_par_count} | "
          f"avg speedup={avg_speedup:.4f}x")
    return output_path


def discover_pairs(results_dir):
    """扫描 results 目录，找到所有 baseline+flagos 配对。

    支持两种目录布局:
      - results/{op}/{op}_nvidia.json  (flat)
      - results/{op}/{op}/{op}_nvidia.json  (nested)

    文件命名约定:
      - baseline: {op}_{platform}.json  (e.g. swiglu_nvidia.json)
      - flagos:   {op}_flagos_{platform}.json  (e.g. swiglu_flagos_nvidia.json)
    """
    results_dir = Path(results_dir)
    # 递归找所有 *_flagos_*.json
    flagos_files = sorted(results_dir.rglob("*_flagos_*.json"))

    pairs = []
    for flagos_path in flagos_files:
        # 从 flagos 文件名推导 baseline 文件名
        # e.g. swiglu_flagos_nvidia.json → swiglu_nvidia.json
        name = flagos_path.name
        baseline_name = name.replace("_flagos_", "_")
        baseline_path = flagos_path.parent / baseline_name

        if baseline_path.exists():
            pairs.append((baseline_path, flagos_path))
        else:
            print(f"  [SKIP] No baseline for {flagos_path.name}", file=sys.stderr)

    return pairs


def main():
    parser = argparse.ArgumentParser(
        description="从两份性能结果JSON生成对比结果JSON"
    )
    parser.add_argument(
        "--baseline", default=None,
        help="基线结果JSON (e.g. results/swiglu/swiglu_nvidia.json)"
    )
    parser.add_argument(
        "--flagos", default=None,
        help="FlagOS结果JSON (e.g. results/swiglu/swiglu_flagos_nvidia.json)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="输出路径 (默认自动推断)"
    )
    parser.add_argument(
        "--auto", action="store_true",
        help="自动扫描 results 目录，为所有配对生成对比"
    )
    parser.add_argument(
        "--results-dir", default=None,
        help="results 根目录 (默认: results/)"
    )
    args = parser.parse_args()

    if args.auto:
        results_dir = args.results_dir or (
            Path(__file__).resolve().parent.parent / "results"
        )
        pairs = discover_pairs(results_dir)
        if not pairs:
            print("No baseline+flagos pairs found.", file=sys.stderr)
            sys.exit(1)

        print(f"Found {len(pairs)} operator pairs:\n")
        for baseline_path, flagos_path in pairs:
            gen_compare(baseline_path, flagos_path)
        print(f"\nDone. Generated {len(pairs)} compare files.")
    else:
        if not args.baseline or not args.flagos:
            parser.error("需要 --baseline 和 --flagos，或使用 --auto")

        if not Path(args.baseline).exists():
            print(f"  [ERROR] Baseline not found: {args.baseline}", file=sys.stderr)
            sys.exit(1)
        if not Path(args.flagos).exists():
            print(f"  [ERROR] FlagOS not found: {args.flagos}", file=sys.stderr)
            sys.exit(1)

        gen_compare(args.baseline, args.flagos, args.output)


if __name__ == "__main__":
    main()
