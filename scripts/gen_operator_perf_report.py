#!/usr/bin/env python3
"""
Generate a detailed per-workload comparison CSV between FlagOS and vLLM.

Output columns:
    算子名, workload, parameters, FlagOS耗时(ms), vLLM耗时(ms), 加速比, FlagOS吞吐(GFLOPS), vLLM吞吐(GFLOPS)

Pairs *_flagos.json with *_vllm.json by operator and workload name,
producing one row per workload with side-by-side timing.

Usage:
    python gen_operator_perf_report.py
    python gen_operator_perf_report.py --input-dir ../results --output detail.csv
"""

import argparse
import csv
import json
import sys
from pathlib import Path


def parse_args() -> argparse.Namespace:
    default_input = Path(__file__).resolve().parent.parent / "results"

    parser = argparse.ArgumentParser(
        description="Generate detailed per-workload comparison CSV (FlagOS vs vLLM)."
    )
    parser.add_argument(
        "--input-dir", type=Path, default=default_input,
        help="Directory containing *_flagos.json / *_vllm.json files.",
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
    """Format parameters dict into a compact readable string."""
    if not params:
        return ""
    parts = [f"{k}={v}" for k, v in params.items()]
    return ", ".join(parts)


def main() -> int:
    args = parse_args()
    input_dir = args.input_dir

    if not input_dir.is_dir():
        print(f"Error: directory not found: {input_dir}", file=sys.stderr)
        return 1

    flagos_files = sorted(input_dir.glob("*_flagos.json"))
    if not flagos_files:
        print(f"No *_flagos.json files found in {input_dir}", file=sys.stderr)
        return 1

    rows = []

    for flagos_path in flagos_files:
        op_name = flagos_path.stem.replace("_flagos", "")
        vllm_path = input_dir / f"{op_name}_vllm.json"

        flagos_data = load_json(flagos_path)
        if flagos_data is None:
            continue

        # Build vllm lookup: workload -> result dict
        vllm_data = load_json(vllm_path) if vllm_path.exists() else None
        vllm_map = {}
        if vllm_data:
            for r in vllm_data.get("results", []):
                wl = r.get("workload", "")
                if wl:
                    vllm_map[wl] = r

        for result in flagos_data.get("results", []):
            workload = result.get("workload", "")
            params = result.get("parameters", {})
            perf = result.get("performance", {})
            flagos_time = perf.get("device_time", {}).get("mean_ms")
            flagos_gflops = perf.get("throughput", {}).get("gflops")

            # Match vllm result
            vllm_result = vllm_map.get(workload)
            vllm_time = None
            vllm_gflops = None
            if vllm_result:
                vllm_perf = vllm_result.get("performance", {})
                vllm_time = vllm_perf.get("device_time", {}).get("mean_ms")
                vllm_gflops = vllm_perf.get("throughput", {}).get("gflops")

            # Compute speedup
            speedup = ""
            if flagos_time and vllm_time and flagos_time > 0:
                speedup = f"{vllm_time / flagos_time:.4f}"

            rows.append({
                "算子名": op_name,
                "workload": workload,
                "parameters": format_params(params),
                "FlagOS耗时(ms)": f"{flagos_time:.4f}" if flagos_time is not None else "N/A",
                "vLLM耗时(ms)": f"{vllm_time:.4f}" if vllm_time is not None else "N/A",
                "加速比": speedup if speedup else "N/A",
                "FlagOS吞吐(GFLOPS)": f"{flagos_gflops:.2f}" if flagos_gflops is not None else "N/A",
                "vLLM吞吐(GFLOPS)": f"{vllm_gflops:.2f}" if vllm_gflops is not None else "N/A",
            })

    output_path = args.output or (input_dir / "operator_detail_comparison.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "算子名", "workload", "parameters",
        "FlagOS耗时(ms)", "vLLM耗时(ms)", "加速比",
        "FlagOS吞吐(GFLOPS)", "vLLM吞吐(GFLOPS)",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    operators = sorted({r["算子名"] for r in rows})
    print(f"Generated: {output_path}", file=sys.stderr)
    print(f"Total rows: {len(rows)}, Operators: {len(operators)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
