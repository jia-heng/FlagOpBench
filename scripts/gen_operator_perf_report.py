#!/usr/bin/env python3
"""
Aggregate FlagOpBench operator performance reports into a CSV.

Reads all per-operator JSON reports under the results directory (e.g. mm.json,
grouped_matmul.json, add_rmsnorm_bias.json) and flattens them into a single CSV
with operator name, scenario, parameters, accuracy status, and timing metrics.

Usage:
    python gen_operator_perf_report.py
    python gen_operator_perf_report.py --input-dir ../results --output ../results/operator_performance_report.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    default_script_dir = Path(__file__).resolve().parent
    default_root = default_script_dir.parent
    default_input = default_root / "results"
    default_output = default_root / "results" / "operator_performance_report.csv"

    parser = argparse.ArgumentParser(
        description="Aggregate FlagOpBench baseline regression JSON reports into CSV."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=default_input,
        help="Directory containing per-operator *.json files (default: ../results).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=default_output,
        help="Output CSV path (default: ../results/operator_performance_report.csv).",
    )
    return parser.parse_args()


def flatten_regression(data: dict, result: dict) -> dict:
    """Flatten one JSON result object into a CSV row."""
    params = result.get("params", {})
    perf = result.get("performance", {})
    device_time = perf.get("device_time", {})
    wall_time = perf.get("wall_time", {})
    accuracy = result.get("accuracy", {})

    return {
        "platform": data.get("platform", ""),
        "backend": data.get("backend", ""),
        "timestamp": data.get("timestamp", ""),
        "operator": result.get("operator", ""),
        "scenario": result.get("scenario", ""),
        "phase": params.get("phase", ""),
        "M": params.get("M", ""),
        "N": params.get("N", ""),
        "K": params.get("K", ""),
        "num_tokens": params.get("num_tokens", ""),
        "hidden_size": params.get("hidden_size", ""),
        "eps": params.get("eps", ""),
        "dtype": params.get("dtype", ""),
        "source": params.get("source", ""),
        "device_time_mean_ms": device_time.get("mean_ms", ""),
        "device_time_median_ms": device_time.get("median_ms", ""),
        "device_time_min_ms": device_time.get("min_ms", ""),
        "device_time_max_ms": device_time.get("max_ms", ""),
        "device_time_p99_ms": device_time.get("p99_ms", ""),
        "device_time_std_ms": device_time.get("std_ms", ""),
        "wall_time_mean_ms": wall_time.get("mean_ms", ""),
        "accuracy_passed": accuracy.get("passed", ""),
        "error": result.get("error", ""),
    }


def main() -> int:
    args = parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: input directory not found: {args.input_dir}", file=sys.stderr)
        return 1

    # Collect per-operator report files. We accept any *.json that has a top-level
    # "results" list where each entry contains an "operator" and "scenario".
    # Known non-operator files (nvidia_*.json, test_*.json) are skipped.
    skipped_prefixes = ("nvidia_", "test_")
    report_files = []
    for f in sorted(args.input_dir.glob("*.json")):
        if f.name.startswith(skipped_prefixes):
            continue
        try:
            with f.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(data.get("results"), list) and data["results"]:
            first = data["results"][0]
            if isinstance(first, dict) and "operator" in first and "scenario" in first:
                report_files.append(f)

    if not report_files:
        print(f"Warning: no valid operator report *.json files found in {args.input_dir}", file=sys.stderr)
        return 0

    rows = []
    skipped_files = []
    for f in report_files:
        try:
            with f.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            for r in data.get("results", []):
                rows.append(flatten_regression(data, r))
        except (json.JSONDecodeError, OSError) as e:
            skipped_files.append((f.name, str(e)))
            continue

    if skipped_files:
        print("Warning: skipped files due to errors:", file=sys.stderr)
        for name, err in skipped_files:
            print(f"  {name}: {err}", file=sys.stderr)

    fieldnames = [
        "platform", "backend", "timestamp", "operator", "scenario",
        "phase", "M", "N", "K", "num_tokens", "hidden_size", "eps", "dtype", "source",
        "device_time_mean_ms", "device_time_median_ms", "device_time_min_ms",
        "device_time_max_ms", "device_time_p99_ms", "device_time_std_ms",
        "wall_time_mean_ms", "accuracy_passed", "error",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    operators = sorted({r["operator"] for r in rows})
    print(f"Generated: {args.output}", file=sys.stderr)
    print(f"Total rows: {len(rows)}", file=sys.stderr)
    print(f"Operators ({len(operators)}): {operators}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
