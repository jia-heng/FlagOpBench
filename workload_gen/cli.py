#!/usr/bin/env python3
"""
Workload Generator CLI

从模型配置推导真实推理 workload，生成 FlagOpBench YAML case 文件。
"""

import argparse
import sys
from pathlib import Path

from .generator import WorkloadGenerator
from .exporter import YAMLExporter
from .config import InferenceScenario


def cmd_generate(args):
    """生成 workload 并导出为 YAML"""
    print(f"\n📦 Generating workloads from {args.config}")
    print("=" * 60)

    # 创建生成器
    generator = WorkloadGenerator.from_config_file(args.config)
    config = generator.config

    print(f"  Model: {config.model_name}")
    print(f"  Type: {config.model_type}")
    print(f"  Hidden: {config.hidden_size}, Heads: {config.num_attention_heads}, "
          f"FFN: {config.intermediate_size}")

    # 生成场景
    if args.scenarios == "standard":
        scenarios = InferenceScenario.standard_scenarios()
    else:
        # 自定义场景：格式 "decode:1,4,8;prefill:128,512"
        scenarios = parse_custom_scenarios(args.scenarios)

    print(f"  Scenarios: {len(scenarios)} (decode + prefill)")

    # 生成 workload
    workload_sets = generator.generate(scenarios)

    print(f"\n✓ Generated {len(workload_sets)} operator workload sets")
    print(f"  Operators: {', '.join(ws.op_name for ws in workload_sets)}")

    # 导出 YAML
    print(f"\n📝 Exporting to {args.output}/")
    print("=" * 60)

    exporter = YAMLExporter(args.output)
    exported_files = exporter.export(workload_sets)

    print(f"\n✅ Done! Exported {len(exported_files)} files to {args.output}/")
    print(f"   Run with: python baseline/run.py run --case-dir {args.output}/")


def cmd_list_architectures(args):
    """列出支持的架构类型"""
    from .architectures import ARCHITECTURE_REGISTRY

    print("\n📋 Supported Architectures:")
    print("=" * 60)

    for arch_type, arch_class in ARCHITECTURE_REGISTRY.items():
        print(f"  {arch_type:15s} → {arch_class.__name__}")
        if arch_class.__doc__:
            # 打印前3行文档
            lines = [l.strip() for l in arch_class.__doc__.strip().split('\n') if l.strip()]
            for line in lines[:3]:
                print(f"                     {line}")

    print(f"\nTotal: {len(ARCHITECTURE_REGISTRY)} architectures")


def parse_custom_scenarios(spec: str) -> list[InferenceScenario]:
    """解析自定义场景字符串

    格式: "decode:1,4,8;prefill:128,512"
    """
    scenarios = []

    for part in spec.split(';'):
        phase, values = part.split(':')
        phase = phase.strip()

        for val in values.split(','):
            val = int(val.strip())

            if phase == "decode":
                scenarios.append(InferenceScenario(
                    phase="decode",
                    batch_size=val,
                    seq_len=1,
                    kv_len=2048
                ))
            elif phase == "prefill":
                scenarios.append(InferenceScenario(
                    phase="prefill",
                    batch_size=1,
                    seq_len=val,
                    kv_len=0
                ))

    return scenarios


def main():
    parser = argparse.ArgumentParser(
        description="FlagOpBench Workload Generator - 从模型配置推导真实推理 workload"
    )

    subparsers = parser.add_subparsers(dest="command", help="子命令")

    # generate 子命令
    gen_parser = subparsers.add_parser("generate", help="生成 workload YAML")
    gen_parser.add_argument(
        "--config", "-c",
        required=True,
        help="模型配置文件路径（JSON 格式）"
    )
    gen_parser.add_argument(
        "--output", "-o",
        default="baseline/cases/traced/",
        help="输出目录（默认: baseline/cases/traced/）"
    )
    gen_parser.add_argument(
        "--scenarios",
        default="standard",
        help="推理场景 (standard | 自定义，如 'decode:1,4;prefill:128,512')"
    )
    gen_parser.set_defaults(func=cmd_generate)

    # list 子命令
    list_parser = subparsers.add_parser("list", help="列出支持的架构类型")
    list_parser.set_defaults(func=cmd_list_architectures)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
