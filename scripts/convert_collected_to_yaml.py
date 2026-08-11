#!/usr/bin/env python3
"""
转换采集的 shape JSON 为 FlagOpBench YAML 格式

使用方法:
    python scripts/convert_collected_to_yaml.py \
      --input collected_shapes \
      --output baseline/cases/basic
"""

import json
import yaml
import argparse
from pathlib import Path
from typing import Dict, List
from framework.filters import create_filter


def convert_shapes_to_yaml(shapes_file: Path, operator: str, output_dir: Path, apply_dedup: bool = True):
    """转换单个 shape JSON 文件为 YAML"""

    with open(shapes_file) as f:
        data = json.load(f)

    operator_name = data.get("operator", operator)
    shapes = data.get("shapes", [])

    if not shapes:
        print(f"Warning: No shapes found in {shapes_file}")
        return None

    # 应用去重过滤
    if apply_dedup:
        filter = create_filter(operator_name, keep_first_k=1)
        original_count = len(shapes)
        shapes = filter.filter(shapes)
        print(f"{operator_name}: {original_count} → {len(shapes)} shapes after deduplication")

    # 转换为 scenarios
    scenarios = []
    for idx, shape_info in enumerate(shapes):
        scenario = {
            "name": f"collected_{idx}",
            **{k: v for k, v in shape_info.items() if k != 'operator'}
        }
        scenarios.append(scenario)

    # 构建 YAML 数据
    yaml_data = {
        "operator": operator_name,
        "level": "basic",
        "source": "collected",
        "warmup": 10,
        "iters": 100,
        "scenarios": scenarios,
    }

    # 写入文件
    output_file = output_dir / f"{operator_name}_collected.yaml"
    with open(output_file, 'w') as f:
        yaml.dump(yaml_data, f, sort_keys=False, allow_unicode=True)

    print(f"Written {output_file}: {len(scenarios)} scenarios")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Convert collected shapes to YAML")
    parser.add_argument("--input", type=str, required=True,
                        help="Input directory with collected shape JSON files")
    parser.add_argument("--output", type=str, required=True,
                        help="Output directory for YAML files")
    parser.add_argument("--no-dedup", action="store_true",
                        help="Disable deduplication")

    args = parser.parse_args()

    input_dir = Path(args.input)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 查找所有 JSON 文件
    json_files = list(input_dir.glob("*_shapes.json"))

    if not json_files:
        print(f"No *_shapes.json files found in {input_dir}")
        return 1

    print(f"Found {len(json_files)} shape files")
    print()

    # 转换每个文件
    for json_file in json_files:
        # 从文件名提取算子名称（例如 mm_shapes.json → mm）
        operator = json_file.stem.replace("_shapes", "")
        convert_shapes_to_yaml(
            json_file,
            operator,
            output_dir,
            apply_dedup=not args.no_dedup
        )

    print()
    print(f"Conversion complete! Check {output_dir}")

    return 0


if __name__ == "__main__":
    exit(main())
