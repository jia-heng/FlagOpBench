#!/usr/bin/env python3
"""
Generate an operator-level summary CSV comparing FlagOS vs vLLM performance.

Output columns:
    算子名, 来源(impl), FlagOS平均耗时(ms), vLLM平均耗时(ms), 平均加速比, 用例数, 环境

The script pairs *_flagos.json and *_vllm.json files by operator name,
computes per-workload speedup (vllm_time / flagos_time), and reports
the geometric mean speedup for each operator.

Usage:
    python gen_operator_summary.py
    python gen_operator_summary.py --input-dir ../results --output summary.csv
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
        description="Generate operator summary CSV (FlagOS vs vLLM)."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=default_input,
        help="Directory containing *_flagos.json / *_vllm.json files.",
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
    gpu = env.get("gpu_name", "Unknown")
    mem = env.get("gpu_memory_gb", "")
    cuda = env.get("cuda_version", "")
    torch_ver = env.get("torch_version", "")
    parts = [gpu]
    if mem:
        parts[0] += f" ({mem}GB)"
    if torch_ver:
        parts.append(f"PyTorch {torch_ver}")
    if cuda:
        parts.append(f"CUDA {cuda}")
    return "; ".join(parts)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir

    if not input_dir.is_dir():
        print(f"Error: directory not found: {input_dir}", file=sys.stderr)
        return 1

    # Discover operator names that have flagos results
    flagos_files = sorted(input_dir.glob("*_flagos.json"))
    if not flagos_files:
        print(f"No *_flagos.json files found in {input_dir}", file=sys.stderr)
        return 1

    env_str = ""
    rows = []

    for flagos_path in flagos_files:
        op_name = flagos_path.stem.replace("_flagos", "")
        vllm_path = input_dir / f"{op_name}_vllm.json"

        flagos_data = load_json(flagos_path)
        if flagos_data is None:
            continue

        if not env_str:
            env_str = format_env(flagos_data)

        flagos_map = build_workload_map(flagos_data)

        # Get impl source from first result
        flagos_impl = ""
        if flagos_data.get("results"):
            flagos_impl = flagos_data["results"][0].get("impl_info", {}).get("source", "")

        # If vllm counterpart exists, compute speedup
        vllm_data = load_json(vllm_path) if vllm_path.exists() else None
        vllm_map = build_workload_map(vllm_data) if vllm_data else {}

        vllm_impl = ""
        if vllm_data and vllm_data.get("results"):
            vllm_impl = vllm_data["results"][0].get("impl_info", {}).get("source", "")

        speedups = []
        flagos_times = []
        vllm_times = []

        for wl, flagos_t in flagos_map.items():
            flagos_times.append(flagos_t)
            if wl in vllm_map:
                vllm_t = vllm_map[wl]
                vllm_times.append(vllm_t)
                if flagos_t > 0:
                    speedups.append(vllm_t / flagos_t)

        avg_flagos = sum(flagos_times) / len(flagos_times) if flagos_times else float("nan")
        avg_vllm = sum(vllm_times) / len(vllm_times) if vllm_times else float("nan")
        avg_speedup = geo_mean(speedups) if speedups else float("nan")

        rows.append({
            "算子名": op_name,
            "FlagOS来源": flagos_impl,
            "vLLM来源": vllm_impl,
            "FlagOS平均耗时(ms)": f"{avg_flagos:.4f}" if not math.isnan(avg_flagos) else "N/A",
            "vLLM平均耗时(ms)": f"{avg_vllm:.4f}" if not math.isnan(avg_vllm) else "N/A",
            "平均加速比": f"{avg_speedup:.4f}" if not math.isnan(avg_speedup) else "N/A",
            "用例数": len(flagos_times),
            "环境": env_str,
        })

    # Sort by speedup descending (N/A last)
    def sort_key(r):
        try:
            return -float(r["平均加速比"])
        except (ValueError, TypeError):
            return 0.0
    rows.sort(key=sort_key)

    output_path = args.output or (input_dir / "operator_summary.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["算子名", "FlagOS来源", "vLLM来源", "FlagOS平均耗时(ms)", "vLLM平均耗时(ms)", "平均加速比", "用例数", "环境"]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated: {output_path}", file=sys.stderr)
    print(f"Operators: {len(rows)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
