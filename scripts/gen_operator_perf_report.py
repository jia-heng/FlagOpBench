#!/usr/bin/env python3
"""
Aggregate FlagOpBench operator performance reports into a CSV.

Reads all per-operator JSON reports under the results directory (e.g. mm.json,
grouped_matmul.json, add_rmsnorm_bias.json) and flattens them into a single CSV
with operator name, scenario, timing metrics, parameters, source and accuracy.

The output filename includes platform and backend by default, e.g.:
    operator_report_nvidia_h20.csv

Usage:
    python gen_operator_perf_report.py
    python gen_operator_perf_report.py --input-dir ../results --output ../results/my_report.csv
"""

import argparse
import csv
import json
import re
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    default_script_dir = Path(__file__).resolve().parent
    default_root = default_script_dir.parent
    default_input = default_root / "results"

    parser = argparse.ArgumentParser(
        description="Aggregate FlagOpBench operator performance JSON reports into CSV."
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
        default=None,
        help="Output CSV path (default: ../results/operator_report_<platform>_<backend>.csv).",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    """Convert a platform/backend string to a safe filename token."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def extract_gpu_name(platform: str, backend: str) -> str:
    """Extract a short GPU identifier from the platform string.

    Examples:
        platform='NVIDIA NVIDIA H20', backend='nvidia' -> 'H20'
        platform='NVIDIA H100', backend='nvidia'       -> 'H100'
    """
    clean = platform
    if platform.lower().startswith(backend.lower()):
        clean = platform[len(backend):].strip()
    parts = clean.split()
    return parts[-1] if parts else clean


def flatten_regression(result: dict) -> dict:
    """Flatten one JSON result object into a CSV row."""
    params = result.get("params", {})
    perf = result.get("performance", {})
    device_time = perf.get("device_time", {})
    wall_time = perf.get("wall_time", {})
    accuracy = result.get("accuracy", {})

    return {
        "operator": result.get("operator", ""),
        "scenario": result.get("scenario", ""),
        "device_time_mean_ms": device_time.get("mean_ms", ""),
        "wall_time_mean_ms": wall_time.get("mean_ms", ""),
        "params_json": json.dumps(params, ensure_ascii=False, separators=(",", ":")),
        "source": params.get("source", ""),
        "accuracy_passed": accuracy.get("passed", ""),
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
    platform_backend_pairs = set()
    for f in report_files:
        try:
            with f.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
            platform_backend_pairs.add((data.get("platform", "unknown"), data.get("backend", "unknown")))
            for r in data.get("results", []):
                rows.append(flatten_regression(r))
        except (json.JSONDecodeError, OSError) as e:
            skipped_files.append((f.name, str(e)))
            continue

    if skipped_files:
        print("Warning: skipped files due to errors:", file=sys.stderr)
        for name, err in skipped_files:
            print(f"  {name}: {err}", file=sys.stderr)

    fieldnames = [
        "operator", "scenario", "device_time_mean_ms", "wall_time_mean_ms",
        "params_json", "source", "accuracy_passed",
    ]

    if args.output is None:
        # Derive default filename from backend and GPU model.
        if len(platform_backend_pairs) == 1:
            platform, backend = platform_backend_pairs.pop()
        else:
            platform, backend = "unknown", "unknown"
        gpu = extract_gpu_name(platform, backend)
        filename = f"operator_report_{slugify(backend)}_{slugify(gpu)}.csv"
        args.output = args.input_dir / filename

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
