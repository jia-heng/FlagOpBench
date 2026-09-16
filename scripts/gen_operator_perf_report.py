#!/usr/bin/env python3
"""
Generate a detailed per-workload comparison CSV between FlagOS and baseline.

Output columns:
    operator, workload, parameters, flagos_time_ms, baseline_time_ms, speedup,
    flagos_gflops, baseline_gflops

Scans results/ recursively for *_flagos_{platform}.json files and pairs them
with *_{platform}.json baseline files by operator name and workload.

Supports both flat and nested directory layouts:
  - results/{op}/{op}_nvidia.json
  - results/{op}/{op}/{op}_nvidia.json

Usage:
    python scripts/gen_operator_perf_report.py
    python scripts/gen_operator_perf_report.py --input-dir results/ --output detail.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    default_input = Path(__file__).resolve().parent.parent / "results"

    parser = argparse.ArgumentParser(
        description="Generate detailed per-workload comparison CSV."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=default_input,
        help="Root directory containing result JSON files.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path (default: <input-dir>/operator_detail_comparison.csv).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def format_params(params: dict) -> str:
    if not params:
        return ""
    return ", ".join(f"{k}={v}" for k, v in params.items())


def discover_pairs(input_dir: Path) -> list[tuple[str, Path, Path | None]]:
    """Find all flagos result files and their baseline counterparts.

    Returns list of (op_name, flagos_path, baseline_path_or_None).
    """
    flagos_files = sorted(input_dir.rglob("*_flagos_*.json"))
    pairs = []

    for flagos_path in flagos_files:
        # skip compare files
        if "_compare_" in flagos_path.name:
            continue

        # Derive baseline name: {op}_flagos_{platform}.json -> {op}_{platform}.json
        baseline_name = flagos_path.name.replace("_flagos_", "_")
        baseline_path = flagos_path.parent / baseline_name

        # Extract op name from metadata or filename
        op_name = flagos_path.stem  # e.g. swiglu_flagos_nvidia
        # Remove _flagos_{platform} suffix to get op name
        parts = op_name.split("_flagos_")
        if parts:
            op_name = parts[0]

        if baseline_path.exists():
            pairs.append((op_name, flagos_path, baseline_path))
        else:
            pairs.append((op_name, flagos_path, None))

    return pairs


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir

    if not input_dir.is_dir():
        print(f"Error: directory not found: {input_dir}", file=sys.stderr)
        return 1

    pairs = discover_pairs(input_dir)
    if not pairs:
        print(f"No *_flagos_*.json files found in {input_dir}", file=sys.stderr)
        return 1

    rows = []

    for op_name, flagos_path, baseline_path in pairs:
        flagos_data = load_json(flagos_path)
        if flagos_data is None:
            continue

        # Build baseline lookup: workload -> result dict
        baseline_map = {}
        if baseline_path:
            baseline_data = load_json(baseline_path)
            if baseline_data:
                for r in baseline_data.get("results", []):
                    wl = r.get("workload", "")
                    if wl:
                        baseline_map[wl] = r

        for result in flagos_data.get("results", []):
            workload = result.get("workload", "")
            params = result.get("parameters", {})
            perf = result.get("performance", {})
            flagos_time = perf.get("device_time", {}).get("mean_ms")
            flagos_gflops = perf.get("throughput", {}).get("gflops")

            # Match baseline result
            bl_result = baseline_map.get(workload)
            bl_time = None
            bl_gflops = None
            if bl_result:
                bl_perf = bl_result.get("performance", {})
                bl_time = bl_perf.get("device_time", {}).get("mean_ms")
                bl_gflops = bl_perf.get("throughput", {}).get("gflops")

            # Compute speedup (baseline / flagos, >1 means flagos is faster)
            speedup = ""
            if flagos_time and bl_time and flagos_time > 0:
                speedup = f"{bl_time / flagos_time:.4f}"

            rows.append({
                "operator": op_name,
                "workload": workload,
                "parameters": format_params(params),
                "flagos_time_ms": f"{flagos_time:.4f}" if flagos_time is not None else "N/A",
                "baseline_time_ms": f"{bl_time:.4f}" if bl_time is not None else "N/A",
                "speedup": speedup if speedup else "N/A",
                "flagos_gflops": f"{flagos_gflops:.2f}" if flagos_gflops is not None else "N/A",
                "baseline_gflops": f"{bl_gflops:.2f}" if bl_gflops is not None else "N/A",
            })

    output_path = args.output or (input_dir / "operator_detail_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "operator", "workload", "parameters",
        "flagos_time_ms", "baseline_time_ms", "speedup",
        "flagos_gflops", "baseline_gflops",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    operators = sorted({r["operator"] for r in rows})
    print(f"Generated: {output_path}", file=sys.stderr)
    print(f"Total rows: {len(rows)}, Operators: {len(operators)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
