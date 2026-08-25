#!/usr/bin/env python3
"""
Generate a high-level operator summary CSV from FlagOpBench JSON reports.

Output columns:
    算子名, 框架, 厂商软硬件信息, 测试用例来源

Example row:
    add_rmsnorm_bias (Fused Add+RMSNorm), vllm_ops,
    NVIDIA H20 (95.1 GB, Driver 610.43.02); PyTorch 2.11.0+cu130; CUDA 13.0, cuDNN 91900,
    DeepSeek-V3 (14 workloads)

Usage:
    python gen_operator_summary.py
    python gen_operator_summary.py --input-dir ../results --output ../results/operator_summary_nvidia_h20.csv
"""

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from pathlib import Path


# Known operator metadata. Used for framework and description labels.
OPERATOR_META = {
    "mm": {"framework": "torch.mm", "description": "GEMM NT/NN"},
    "grouped_matmul": {"framework": "torch.bmm", "description": "Grouped GEMM (GQA)"},
    "rms_norm": {"framework": "F.rms_norm", "description": "Standard normalization"},
    "add_rmsnorm_bias": {"framework": "vllm_ops", "description": "Fused Add+RMSNorm"},
    "rope": {"framework": "vllm_ops.rotary_embedding", "description": "Positional encoding"},
    "fused_q_kv_rmsnorm": {"framework": "PyTorch (custom)", "description": "Q/KV RMSNorm"},
    "persistent_topk": {"framework": "torch.topk", "description": "Sampling operator"},
    "topk_softplus_sqrt": {"framework": "PyTorch (custom)", "description": "Top-P renorm"},
    "topk_selector": {"framework": "torch.topk + gather", "description": "Top-K mask logits"},
    "top_k_per_row_decode": {"framework": "PyTorch (custom)", "description": "Per-row sampling decode"},
    "top_k_per_row_prefill": {"framework": "PyTorch (custom)", "description": "Per-row sampling prefill"},
    "router_gemm_bf16_fp32": {"framework": "torch.mm (bf16→fp32)", "description": "Router GEMM"},
    "flashattention": {"framework": "F.scaled_dot_product_attention", "description": "FlashAttention GQA"},
    "sparse_attention": {"framework": "F.scaled_dot_product_attention", "description": "Block-sparse attention"},
    "flash_mla": {"framework": "PyTorch SDPA + Low-rank", "description": "Flash MLA"},
    "flash_linear_attention": {"framework": "vLLM Triton kernel", "description": "Gated Delta Rule"},
    "fused_moe": {"framework": "PyTorch (custom)", "description": "FP8 MoE with routing"},
    "bmm": {"framework": "torch.bmm", "description": "Batch matrix multiply"},
    "gemma_rms_norm": {"framework": "F.rms_norm", "description": "Gemma RMSNorm"},
    "layernorm": {"framework": "F.layer_norm", "description": "Layer normalization"},
    "silu_and_mul": {"framework": "PyTorch (custom)", "description": "SiLU + Mul FFN"},
    "silu_and_mul_with_clamp": {"framework": "PyTorch (custom)", "description": "SiLU + Mul FFN with clamp"},
    "swiglu": {"framework": "PyTorch (custom)", "description": "SwiGLU FFN"},
    "gemm_w8a8": {"framework": "torch.mm", "description": "W8A8 GEMM"},
    "fused_marlin_moe": {"framework": "vllm_ops", "description": "Marlin MoE"},
    "moe_align_block_size": {"framework": "vllm_ops", "description": "MoE token alignment"},
    "moe_sum": {"framework": "PyTorch (custom)", "description": "MoE sum"},
    "per_token_group_fp8_quant": {"framework": "vllm_ops", "description": "Per-token group FP8 quant"},
    "fused_inv_rope_fp8_quant": {"framework": "vllm_ops", "description": "Inverse RoPE + FP8 quant"},
    "fp8_einsum": {"framework": "torch.einsum", "description": "FP8 einsum"},
    "kv_rms_norm_rope_cache": {"framework": "vllm_ops", "description": "KV cache RMSNorm + RoPE"},
    "causal_conv1d_decode": {"framework": "PyTorch (custom)", "description": "Causal 1D conv decode"},
    "causal_conv1d_prefill": {"framework": "PyTorch (custom)", "description": "Causal 1D conv prefill"},
}

# Models whose names may appear in the source string.
MODEL_PATTERNS = [
    r"Llama-3\.1-8B",
    r"DeepSeek-V3",
    r"Qwen3-30B-A3B",
    r"Gemma-2-9B",
    r"Mixtral",
    r"Mamba-2\.8B",
    r"Falcon-H1",
    r"GLA",
]


def parse_args() -> argparse.Namespace:
    default_script_dir = Path(__file__).resolve().parent
    default_root = default_script_dir.parent
    default_input = default_root / "results"

    parser = argparse.ArgumentParser(
        description="Generate a high-level operator summary CSV from FlagOpBench JSON reports."
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
        help="Output CSV path (default: ../results/operator_summary_<backend>_<gpu>.csv).",
    )
    return parser.parse_args()


def slugify(text: str) -> str:
    """Convert a string to a safe filename token."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9]+", "_", text)
    text = re.sub(r"_+", "_", text)
    return text.strip("_")


def extract_gpu_name(platform: str, backend: str) -> str:
    """Extract a short GPU identifier from the platform string."""
    clean = platform
    if platform.lower().startswith(backend.lower()):
        clean = platform[len(backend):].strip()
    parts = clean.split()
    return parts[-1] if parts else clean


def format_env_info(data: dict) -> str:
    """Format platform/backend/env into a single hardware/software string."""
    env = data.get("env", {})
    gpu = data.get("platform", env.get("gpu_model", "Unknown GPU"))
    gpu_mem = env.get("gpu_memory_gb", "")
    driver = env.get("driver", "")
    pytorch = env.get("pytorch", "")
    cuda = env.get("cuda_version", "")
    cudnn = env.get("cudnn_version", "")

    parts = [f"{gpu} ({gpu_mem} GB, Driver {driver})"]
    if pytorch:
        parts.append(f"PyTorch {pytorch}")
    if cuda:
        parts.append(f"CUDA {cuda}")
    if cudnn:
        parts.append(f"cuDNN {cudnn}")

    return "; ".join(parts)


def extract_models(source: str) -> list:
    """Extract model names from a source string."""
    if not source:
        return []
    found = []
    for pattern in MODEL_PATTERNS:
        for match in re.finditer(pattern, source):
            if match.group(0) not in found:
                found.append(match.group(0))
    return found


def infer_framework(operator: str, source: str) -> str:
    """Infer framework label for unknown operators."""
    if "vllm" in operator.lower() or "vllm" in source.lower():
        return "vllm_ops"
    return "torch"


def infer_description(operator: str) -> str:
    """Infer a short description for unknown operators."""
    return operator.replace("_", " ").title()


def main() -> int:
    args = parse_args()

    if not args.input_dir.is_dir():
        print(f"Error: input directory not found: {args.input_dir}", file=sys.stderr)
        return 1

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
                report_files.append((f, data))

    if not report_files:
        print(f"Warning: no valid operator report *.json files found in {args.input_dir}", file=sys.stderr)
        return 0

    # Per-operator aggregation.
    op_info = {}
    op_models = defaultdict(list)
    op_workload_count = defaultdict(int)
    env_info_str = ""
    platform_backend_pairs = set()

    for f, data in report_files:
        platform_backend_pairs.add((data.get("platform", "unknown"), data.get("backend", "unknown")))
        if not env_info_str:
            env_info_str = format_env_info(data)

        for r in data.get("results", []):
            op = r.get("operator", "")
            if not op:
                continue
            if op not in op_info:
                meta = OPERATOR_META.get(op, {})
                framework = meta.get("framework", infer_framework(op, r.get("params", {}).get("source", "")))
                description = meta.get("description", infer_description(op))
                op_info[op] = {"framework": framework, "description": description}

            op_workload_count[op] += 1
            for model in extract_models(r.get("params", {}).get("source", "")):
                if model not in op_models[op]:
                    op_models[op].append(model)

    rows = []
    for op in sorted(op_info.keys()):
        info = op_info[op]
        models = op_models.get(op, [])
        count = op_workload_count[op]
        if models:
            source_str = f"{', '.join(models)} ({count} workloads)"
        else:
            source_str = f"({count} workloads)"
        rows.append({
            "算子名": f"{op} ({info['description']})" if info["description"] else op,
            "框架": info["framework"],
            "厂商软硬件信息": env_info_str,
            "测试用例来源": source_str,
        })

    if args.output is None:
        if len(platform_backend_pairs) == 1:
            platform, backend = platform_backend_pairs.pop()
        else:
            platform, backend = "unknown", "unknown"
        gpu = extract_gpu_name(platform, backend)
        filename = f"operator_summary_{slugify(backend)}_{slugify(gpu)}.csv"
        args.output = args.input_dir / filename

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=["算子名", "框架", "厂商软硬件信息", "测试用例来源"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Generated: {args.output}", file=sys.stderr)
    print(f"Total operators: {len(rows)}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
