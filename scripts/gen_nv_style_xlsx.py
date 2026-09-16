#!/usr/bin/env python3
"""从 FlagOpBench compare JSON（或 baseline+flagos 对）生成 NV 同款 summary/detail xlsx。

Usage:
    python scripts/gen_nv_style_xlsx.py --compare results/swiglu/swiglu_compare_mthreads.json
    python scripts/gen_nv_style_xlsx.py --compare-dir results --output 国产_mthreads_测试结果.xlsx
    python scripts/gen_nv_style_xlsx.py \\
        --baseline results/swiglu/swiglu_mthreads.json \\
        --flagos results/swiglu/swiglu_flagos_mthreads.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def geo_mean(values: List[float]) -> float:
    positives = [v for v in values if v and v > 0]
    if not positives:
        return float("nan")
    return math.exp(sum(math.log(v) for v in positives) / len(positives))


def format_env(env: dict) -> str:
    parts = []
    name = env.get("device_name") or env.get("gpu_name") or "Unknown"
    mem = env.get("device_memory_gb") or env.get("gpu_memory_gb")
    if mem:
        parts.append(f"{name} ({mem}GB)")
    else:
        parts.append(str(name))
    if env.get("torch_version"):
        parts.append(f"PyTorch {env['torch_version']}")
    if env.get("cuda_version"):
        parts.append(f"CUDA {env['cuda_version']}")
    if env.get("cann_version"):
        parts.append(f"CANN {env['cann_version']}")
    if env.get("platform"):
        parts.append(f"platform={env['platform']}")
    return "; ".join(parts)


def compare_from_pair(baseline_path: Path, flagos_path: Path) -> dict:
    """就地构造与 gen_compare_result 类似的结构（不写中间文件）。"""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from gen_compare_result import load_results, compute_speedup  # noqa: E402

    baseline_meta, baseline_env, baseline_results = load_results(baseline_path)
    flagos_meta, _flagos_env, flagos_results = load_results(flagos_path)
    matched = set(baseline_results) & set(flagos_results)

    with baseline_path.open("r", encoding="utf-8") as f:
        ordered = [(r["operator"], r["workload"]) for r in json.load(f)["results"]]

    comparisons = []
    for key in ordered:
        if key not in matched:
            continue
        br = baseline_results[key]
        fr = flagos_results[key]
        b_mean = br["performance"]["device_time"]["mean_ms"]
        f_mean = fr["performance"]["device_time"]["mean_ms"]
        speedup = compute_speedup(b_mean, f_mean)
        comparisons.append({
            "operator": key[0],
            "workload": key[1],
            "parameters": br.get("parameters", {}),
            "baseline": {
                "provider": br.get("provider", baseline_meta.get("provider", "unknown")),
                "mean_ms": b_mean,
                "gflops": br["performance"]["throughput"]["gflops"],
                "impl_source": br.get("impl_info", {}).get("source", ""),
            },
            "flagos": {
                "provider": fr.get("provider", "flagos"),
                "mean_ms": f_mean,
                "gflops": fr["performance"]["throughput"]["gflops"],
                "impl_source": fr.get("impl_info", {}).get("source", ""),
            },
            "speedup": round(speedup, 4),
        })

    return {
        "metadata": {
            "platform": baseline_meta.get("platform", "unknown"),
            "baseline_provider": baseline_meta.get("provider", "unknown"),
            "flagos_provider": flagos_meta.get("provider", "flagos"),
        },
        "environment": baseline_env,
        "comparisons": comparisons,
    }


def load_compare_docs(args: argparse.Namespace) -> List[dict]:
    docs: List[dict] = []
    if args.compare:
        docs.append(load_json(Path(args.compare)))
    if args.compare_dir:
        root = Path(args.compare_dir)
        for path in sorted(root.rglob("*_compare_*.json")):
            docs.append(load_json(path))
    if args.baseline and args.flagos:
        docs.append(compare_from_pair(Path(args.baseline), Path(args.flagos)))
    return docs


def build_rows(docs: List[dict]):
    detail_rows = []
    # op -> list of speedups / times
    agg: Dict[str, Dict[str, Any]] = {}

    for doc in docs:
        meta = doc.get("metadata", {})
        env_str = format_env(doc.get("environment", {}))
        baseline_impl = meta.get("baseline_provider", "baseline")
        flagos_impl = meta.get("flagos_provider", "flagos")
        platform = meta.get("platform", "")

        for c in doc.get("comparisons", []):
            op = c["operator"]
            b = c["baseline"]
            f = c["flagos"]
            detail_rows.append({
                "operator": op,
                "workload": c.get("workload", ""),
                "parameters": json.dumps(c.get("parameters", {}), ensure_ascii=False, sort_keys=True),
                "flagos_time_ms": f.get("mean_ms"),
                "baseline_time_ms": b.get("mean_ms"),
                "speedup": c.get("speedup"),
                "flagos_gflops": f.get("gflops"),
                "baseline_gflops": b.get("gflops"),
                "flagos_impl": f.get("impl_source") or flagos_impl,
                "baseline_impl": b.get("impl_source") or baseline_impl,
                "environment": env_str,
                "platform": platform,
            })
            bucket = agg.setdefault(op, {
                "flagos_times": [],
                "baseline_times": [],
                "speedups": [],
                "flagos_impl": f.get("impl_source") or flagos_impl,
                "baseline_impl": b.get("impl_source") or baseline_impl,
                "environment": env_str,
                "platform": platform,
            })
            if f.get("mean_ms") is not None:
                bucket["flagos_times"].append(f["mean_ms"])
            if b.get("mean_ms") is not None:
                bucket["baseline_times"].append(b["mean_ms"])
            if c.get("speedup") is not None:
                bucket["speedups"].append(c["speedup"])

    summary_rows = []
    for op, bucket in sorted(agg.items()):
        ft = bucket["flagos_times"]
        bt = bucket["baseline_times"]
        summary_rows.append({
            "operator": op,
            "flagos_impl": bucket["flagos_impl"],
            "baseline_impl": bucket["baseline_impl"],
            "flagos_avg_time_ms": round(sum(ft) / len(ft), 4) if ft else None,
            "baseline_avg_time_ms": round(sum(bt) / len(bt), 4) if bt else None,
            "avg_speedup": round(geo_mean(bucket["speedups"]), 4) if bucket["speedups"] else None,
            "num_workloads": len(bucket["speedups"]),
            "environment": bucket["environment"],
            "platform": bucket["platform"],
        })
    return summary_rows, detail_rows


def write_xlsx(summary_rows, detail_rows, output: Path) -> None:
    try:
        from openpyxl import Workbook
    except ImportError:
        # fallback CSV pair
        import csv
        output.parent.mkdir(parents=True, exist_ok=True)
        summary_csv = output.with_suffix("").as_posix() + "_summary.csv"
        detail_csv = output.with_suffix("").as_posix() + "_detail.csv"
        if summary_rows:
            with open(summary_csv, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
                w.writeheader()
                w.writerows(summary_rows)
        if detail_rows:
            with open(detail_csv, "w", newline="", encoding="utf-8-sig") as f:
                w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
                w.writeheader()
                w.writerows(detail_rows)
        print(f"  openpyxl missing; wrote CSV:\n    {summary_csv}\n    {detail_csv}")
        return

    wb = Workbook()
    ws_sum = wb.active
    ws_sum.title = "summary"
    sum_headers = [
        "operator", "flagos_impl", "baseline_impl",
        "flagos_avg_time_ms", "baseline_avg_time_ms", "avg_speedup",
        "num_workloads", "environment", "platform",
    ]
    ws_sum.append(sum_headers)
    for row in summary_rows:
        ws_sum.append([row.get(h) for h in sum_headers])

    ws_det = wb.create_sheet("detail")
    det_headers = [
        "operator", "workload", "parameters",
        "flagos_time_ms", "baseline_time_ms", "speedup",
        "flagos_gflops", "baseline_gflops",
        "flagos_impl", "baseline_impl", "environment", "platform",
    ]
    ws_det.append(det_headers)
    for row in detail_rows:
        ws_det.append([row.get(h) for h in det_headers])

    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)
    print(f"  Wrote xlsx: {output}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate NV-style FlagOS vs baseline xlsx")
    p.add_argument("--compare", type=str, help="Single *_compare_*.json")
    p.add_argument("--compare-dir", type=str, help="Scan directory for *_compare_*.json")
    p.add_argument("--baseline", type=str, help="Baseline JSON (pair with --flagos)")
    p.add_argument("--flagos", type=str, help="FlagOS JSON (pair with --baseline)")
    p.add_argument(
        "--output", "-o", type=str, default="results/国产平台_测试结果.xlsx",
        help="Output xlsx path",
    )
    return p.parse_args()


def main() -> int:
    args = parse_args()
    if not (args.compare or args.compare_dir or (args.baseline and args.flagos)):
        print("Need --compare / --compare-dir / (--baseline + --flagos)", file=sys.stderr)
        return 1

    docs = load_compare_docs(args)
    if not docs:
        print("No compare documents loaded", file=sys.stderr)
        return 1

    summary_rows, detail_rows = build_rows(docs)
    if not detail_rows:
        print("No comparison rows", file=sys.stderr)
        return 1

    write_xlsx(summary_rows, detail_rows, Path(args.output))
    print(f"  summary ops={len(summary_rows)}, detail rows={len(detail_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
