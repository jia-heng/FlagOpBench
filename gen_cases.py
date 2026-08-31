#!/usr/bin/env python3
"""
用例生成器 CLI — 从模型 config + 部署场景自动生成 benchmark YAML case 文件

Usage:
    # 为所有模型生成 online_serving 场景的用例
    python gen_cases.py

    # 指定模型和算子
    python gen_cases.py --models deepseek_v4_pro,kimi_k3 --operators fused_moe,group_gemm

    # 指定 profile
    python gen_cases.py --profile offline_batch

    # 列出可用的模型和算子
    python gen_cases.py --list
"""

import argparse
import sys
from pathlib import Path

# 允许直接运行（不需要 package install）
sys.path.insert(0, str(Path(__file__).resolve().parent))

from casegen.model_parser import load_all_models
from casegen.param_mapping import load_operator_registry, get_applicable_operators
from casegen.profile_loader import load_all_profiles
from casegen.generator import generate_all, generate_merged


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent

    parser = argparse.ArgumentParser(description="Auto-generate benchmark case YAML files.")
    parser.add_argument(
        "--models", type=str, default="all",
        help="Comma-separated model names (stem of JSON files), or 'all'.",
    )
    parser.add_argument(
        "--operators", type=str, default="all",
        help="Comma-separated operator names, or 'all'.",
    )
    parser.add_argument(
        "--profile", type=str, default="online_serving",
        help="Profile name to use (default: online_serving).",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=root / "model_configs",
        help="Directory containing model config JSON files.",
    )
    parser.add_argument(
        "--profile-dir", type=Path, default=root / "profiles",
        help="Directory containing profile YAML files.",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=root / "cases" / "generated",
        help="Output directory for generated YAML files.",
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available models, operators, and profiles.",
    )
    parser.add_argument(
        "--no-dedup", action="store_true",
        help="Disable cross-model deduplication (generate all even if const_axes identical).",
    )
    parser.add_argument(
        "--merged", action="store_true",
        help="Generate merged case files (one per operator, cross-model aggregated).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Load models
    all_models = load_all_models(args.model_dir)
    if not all_models:
        print(f"Error: no model configs found in {args.model_dir}", file=sys.stderr)
        return 1

    # Load profiles
    all_profiles = load_all_profiles(args.profile_dir)
    if not all_profiles:
        print(f"Error: no profiles found in {args.profile_dir}", file=sys.stderr)
        return 1

    # --list mode
    if args.list:
        registry = load_operator_registry()
        print("Models:")
        for m in all_models:
            ops = get_applicable_operators(m)
            print(f"  {m.name:30s} arch={m.architecture:20s} operators={len(ops)}")
        print(f"\nOperators ({len(registry)}):")
        for name, spec in sorted(registry.items()):
            print(f"  {name:45s} [{spec['library']}]")
        print(f"\nProfiles ({len(all_profiles)}):")
        for name, p in all_profiles.items():
            print(f"  {name:20s} {p.description}")
        return 0

    # Select models
    if args.models == "all":
        models = all_models
    else:
        model_names = [n.strip() for n in args.models.split(",")]
        model_map = {m.name: m for m in all_models}
        models = []
        for n in model_names:
            if n in model_map:
                models.append(model_map[n])
            else:
                print(f"Warning: model '{n}' not found, skipping.", file=sys.stderr)
        if not models:
            print("Error: no valid models selected.", file=sys.stderr)
            return 1

    # Select operators
    if args.operators == "all":
        operators = None  # generate_all handles None as "all"
    else:
        operators = [n.strip() for n in args.operators.split(",")]

    # Select profile
    if args.profile not in all_profiles:
        print(f"Error: profile '{args.profile}' not found. Available: {list(all_profiles.keys())}", file=sys.stderr)
        return 1
    profile = all_profiles[args.profile]

    # Generate
    if args.merged:
        merged_dir = args.output_dir / "merged"
        generated = generate_merged(models, operators, profile, merged_dir)
        print(f"Generated {len(generated)} merged case files in {merged_dir}", file=sys.stderr)
    else:
        generated = generate_all(models, operators, profile, args.output_dir, dedup=not args.no_dedup)
        print(f"Generated {len(generated)} case files in {args.output_dir}", file=sys.stderr)

    for p in generated:
        print(f"  {p.relative_to(args.output_dir)}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    sys.exit(main())
