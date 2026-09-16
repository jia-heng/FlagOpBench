#!/usr/bin/env python3
"""
Generate an operator-level summary CSV comparing FlagOS vs baseline performance.

Output columns:
    operator, flagos_impl, baseline_impl, flagos_avg_time_ms, baseline_avg_time_ms,
    avg_speedup, num_workloads, environment

Scans results/ recursively for *_flagos_{platform}.json files and pairs them
with *_{platform}.json baseline files. Computes per-workload speedup and
reports the geometric mean speedup for each operator.

Supports both flat and nested directory layouts:
  - results/{op}/{op}_nvidia.json
  - results/{op}/{op}/{op}_nvidia.json

Usage:
    python scripts/gen_operator_summary.py
    python scripts/gen_operator_summary.py --input-dir results/ --output summary.csv
"""

import argparse
import csv
import json
import math
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    default_input = Path(__file__).resolve().parent.parent / "results"

    parser = argparse.ArgumentParser(
        description="Generate operator summary CSV (FlagOS vs baseline)."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=default_input,
        help="Root directory containing result JSON files.",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="Output CSV path (default: <input-dir>/operator_summary.csv).",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict | None:
    try:
        with path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def build_workload_map(data: dict) -> dict:
    """Map workload name -> device_time mean_ms."""
    mapping = {}
    for r in data.get("results", []):
        wl = r.get("workload", "")
        mean_ms = r.get("performance", {}).get("device_time", {}).get("mean_ms")
        if wl and mean_ms is not None:
            mapping[wl] = mean_ms
    return mapping


def geo_mean(values: list[float]) -> float:
    """Geometric mean of positive values."""
    if not values:
        return float("nan")
    log_sum = sum(math.log(v) for v in values if v > 0)
    n = sum(1 for v in values if v > 0)
    if n == 0:
        return float("nan")
    return math.exp(log_sum / n)


def format_env(data: dict) -> str:
    env = data.get("environment", {})
    device = env.get("device_name", env.get("gpu_name", "Unknown"))
    mem = env.get("device_memory_gb", env.get("gpu_memory_gb", ""))
    cuda = env.get("cuda_version", "")
    torch_ver = env.get("torch_version", "")
    parts = [device]
    if mem:
        parts[0] += f" ({mem}GB)"
    if torch_ver:
        parts.append(f"PyTorch {torch_ver}")
    if cuda:
        parts.append(f"CUDA {cuda}")
    return "; ".join(parts)


def discover_pairs(input_dir: Path) -> list[tuple[str, Path, Path | None]]:
    """Find flagos result files and their baseline counterparts.

    Returns list of (op_name, flagos_path, baseline_path_or_None).
    """
    flagos_files = sorted(input_dir.rglob("*_flagos_*.json"))
    pairs = []

    for flagos_path in flagos_files:
        if "_compare_" in flagos_path.name:
            continue

        baseline_name = flagos_path.name.replace("_flagos_", "_")
        baseline_path = flagos_path.parent / baseline_name

        op_name = flagos_path.stem.split("_flagos_")[0]

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

    env_str = ""
    rows = []

    for op_name, flagos_path, baseline_path in pairs:
        flagos_data = load_json(flagos_path)
        if flagos_data is None:
            continue

        if not env_str:
            env_str = format_env(flagos_data)

        flagos_map = build_workload_map(flagos_data)

        # Get impl source from first result
        flagos_impl = ""
        if flagos_data.get("results"):
            flagos_impl = (flagos_data["results"][0]
                           .get("impl_info", {}).get("source", ""))

        # Load baseline if available
        baseline_data = load_json(baseline_path) if baseline_path else None
        baseline_map = build_workload_map(baseline_data) if baseline_data else {}

        baseline_impl = ""
        if baseline_data and baseline_data.get("results"):
            baseline_impl = (baseline_data["results"][0]
                             .get("impl_info", {}).get("source", ""))

        speedups = []
        flagos_times = []
        baseline_times = []

        for wl, flagos_t in flagos_map.items():
            flagos_times.append(flagos_t)
            if wl in baseline_map:
                bl_t = baseline_map[wl]
                baseline_times.append(bl_t)
                if flagos_t > 0:
                    speedups.append(bl_t / flagos_t)

        avg_flagos = (sum(flagos_times) / len(flagos_times)
                      if flagos_times else float("nan"))
        avg_baseline = (sum(baseline_times) / len(baseline_times)
                        if baseline_times else float("nan"))
        avg_speedup = geo_mean(speedups) if speedups else float("nan")

        rows.append({
            "operator": op_name,
            "flagos_impl": flagos_impl,
            "baseline_impl": baseline_impl,
            "flagos_avg_time_ms": f"{avg_flagos:.4f}" if not math.isnan(avg_flagos) else "N/A",
            "baseline_avg_time_ms": f"{avg_baseline:.4f}" if not math.isnan(avg_baseline) else "N/A",
            "avg_speedup": f"{avg_speedup:.4f}" if not math.isnan(avg_speedup) else "N/A",
            "num_workloads": len(flagos_times),
            "environment": env_str,
        })

    # Sort by speedup descending (N/A last)
    def sort_key(r):
        try:
            return -float(r["avg_speedup"])
        except (ValueError, TypeError):
            return 0.0
    rows.sort(key=sort_key)

    output_path = args.output or (input_dir / "operator_summary.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "operator", "flagos_impl", "baseline_impl",
        "flagos_avg_time_ms", "baseline_avg_time_ms",
        "avg_speedup", "num_workloads", "environment",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated: {output_path}", file=sys.stderr)
    print(f"Operators: {len(rows)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
