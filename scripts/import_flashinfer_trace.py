#!/usr/bin/env python3
"""
从 flashinfer-trace 数据集导入真实推理 shape 到 FlagOpBench

flashinfer-trace 数据集：
- HuggingFace: flashinfer/flashinfer-trace
- 包含 190 个 Definition，111 个 Workload 文件
- 来源：Llama-3.1-8B, DeepSeek, Qwen 等真实推理

使用方法:
    python scripts/import_flashinfer_trace.py --dataset-path /path/to/flashinfer-trace
"""

import json
import yaml
import argparse
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


class FlashinferTraceImporter:
    """flashinfer-trace 数据集导入器"""

    def __init__(self, dataset_path: Path, output_dir: Path):
        self.dataset_path = dataset_path
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 算子映射：flashinfer 名称 → FlagOpBench 名称
        self.operator_mapping = {
            # GEMM 类
            "gemm": "mm",
            "batch_gemm": "bmm",

            # Attention 类
            "batch_decode_with_paged_kv": "paged_attention_decode",
            "batch_prefill_with_paged_kv": "paged_attention_prefill",
            "single_decode": "attention_decode",
            "single_prefill": "attention_prefill",

            # Normalization
            "rms_norm": "rms_norm",
            "layer_norm": "layernorm",

            # MoE
            "moe_align_block_size": "moe_align",

            # Sampling
            "top_k_sampling": "topk",
            "top_p_sampling": "topp",
        }

    def load_definitions(self) -> Dict[str, Dict]:
        """加载所有 Definition 文件"""
        definitions = {}
        definition_dir = self.dataset_path / "definitions"

        if not definition_dir.exists():
            print(f"Warning: {definition_dir} not found")
            return definitions

        for def_file in definition_dir.glob("*.json"):
            with open(def_file) as f:
                data = json.load(f)
                definitions[data["name"]] = data

        print(f"Loaded {len(definitions)} definitions")
        return definitions

    def load_workloads(self, definition_name: str) -> List[Dict]:
        """加载指定 Definition 的所有 Workload"""
        workloads = []
        workload_dir = self.dataset_path / "workloads" / definition_name

        if not workload_dir.exists():
            return workloads

        for wl_file in workload_dir.glob("*.json"):
            with open(wl_file) as f:
                workloads.append(json.load(f))

        return workloads

    def map_operator_name(self, flashinfer_op: str) -> str:
        """映射算子名称"""
        return self.operator_mapping.get(flashinfer_op, flashinfer_op)

    def convert_gemm_case(self, definition: Dict, workloads: List[Dict]) -> Dict:
        """转换 GEMM 类算子"""
        # 提取 const 轴（从 definition name 解析）
        # 例：gemm_n4096_k4096 → N=4096, K=4096
        def_name = definition["name"]
        const_axes = {}

        # 解析 definition name
        parts = def_name.split("_")
        for part in parts[1:]:  # 跳过 "gemm"
            if part.startswith("n"):
                const_axes["N"] = int(part[1:])
            elif part.startswith("k"):
                const_axes["K"] = int(part[1:])

        # 转换 workloads
        scenarios = []
        for wl in workloads:
            axes = wl.get("axes", {})
            scenario = {
                "name": wl.get("name", f"m{axes.get('M', 0)}"),
                "M": axes.get("M", 0),
                "N": const_axes.get("N", axes.get("N", 0)),
                "K": const_axes.get("K", axes.get("K", 0)),
                "dtype": wl.get("dtype", "fp16"),
            }
            scenarios.append(scenario)

        return {
            "operator": "mm",
            "level": "basic",
            "source": "flashinfer-trace",
            "definition": def_name,
            "warmup": 10,
            "iters": 100,
            "const_axes": const_axes,
            "scenarios": scenarios,
        }

    def convert_attention_case(self, definition: Dict, workloads: List[Dict]) -> Dict:
        """转换 Attention 类算子"""
        def_name = definition["name"]

        # 解析 const 轴
        # 例：gqa_paged_decode_h32_kv8_d128_ps1
        const_axes = {}
        parts = def_name.split("_")
        for part in parts:
            if part.startswith("h") and part[1:].isdigit():
                const_axes["num_qo_heads"] = int(part[1:])
            elif part.startswith("kv") and part[2:].isdigit():
                const_axes["num_kv_heads"] = int(part[2:])
            elif part.startswith("d") and part[1:].isdigit():
                const_axes["head_dim"] = int(part[1:])
            elif part.startswith("ps") and part[2:].isdigit():
                const_axes["page_size"] = int(part[2:])

        # 转换 workloads
        scenarios = []
        for wl in workloads:
            axes = wl.get("axes", {})
            scenario = {
                "name": wl.get("name", f"bs{axes.get('batch_size', 1)}"),
                **axes,  # 保留所有原始 axes
                "dtype": wl.get("dtype", "fp16"),
            }
            scenarios.append(scenario)

        op_name = "paged_attention_decode" if "decode" in def_name else "paged_attention_prefill"

        return {
            "operator": op_name,
            "level": "model",
            "source": "flashinfer-trace",
            "definition": def_name,
            "warmup": 10,
            "iters": 50,
            "const_axes": const_axes,
            "scenarios": scenarios,
        }

    def convert_norm_case(self, definition: Dict, workloads: List[Dict]) -> Dict:
        """转换 Normalization 算子"""
        def_name = definition["name"]

        # 解析 hidden_size
        const_axes = {}
        parts = def_name.split("_")
        for part in parts:
            if part.startswith("h") and part[1:].isdigit():
                const_axes["hidden_size"] = int(part[1:])

        scenarios = []
        for wl in workloads:
            axes = wl.get("axes", {})
            scenario = {
                "name": wl.get("name", f"bs{axes.get('batch_size', 1)}"),
                "batch_size": axes.get("batch_size", 1),
                "hidden_size": const_axes.get("hidden_size", axes.get("hidden_size", 4096)),
                "dtype": wl.get("dtype", "bf16"),
            }
            scenarios.append(scenario)

        op_name = self.map_operator_name(definition.get("operator", "rms_norm"))

        return {
            "operator": op_name,
            "level": "basic",
            "source": "flashinfer-trace",
            "definition": def_name,
            "warmup": 10,
            "iters": 100,
            "const_axes": const_axes,
            "scenarios": scenarios,
        }

    def convert_definition(self, definition: Dict, workloads: List[Dict]) -> Dict:
        """根据算子类型转换 definition"""
        def_name = definition["name"]
        op_type = definition.get("operator", "")

        # 根据算子类型选择转换器
        if "gemm" in def_name or op_type == "gemm":
            return self.convert_gemm_case(definition, workloads)
        elif "decode" in def_name or "prefill" in def_name or "attention" in op_type:
            return self.convert_attention_case(definition, workloads)
        elif "norm" in def_name or "norm" in op_type:
            return self.convert_norm_case(definition, workloads)
        else:
            # 默认转换
            return self.convert_generic_case(definition, workloads)

    def convert_generic_case(self, definition: Dict, workloads: List[Dict]) -> Dict:
        """通用转换"""
        scenarios = []
        for wl in workloads:
            scenario = {
                "name": wl.get("name", "unnamed"),
                **wl.get("axes", {}),
                "dtype": wl.get("dtype", "fp16"),
            }
            scenarios.append(scenario)

        return {
            "operator": self.map_operator_name(definition.get("operator", "unknown")),
            "level": "basic",
            "source": "flashinfer-trace",
            "definition": definition["name"],
            "warmup": 10,
            "iters": 100,
            "scenarios": scenarios,
        }

    def import_all(self, operator_filter: List[str] = None):
        """导入所有 definitions"""
        definitions = self.load_definitions()

        # 按算子分组
        by_operator = defaultdict(list)

        for def_name, definition in definitions.items():
            workloads = self.load_workloads(def_name)

            if not workloads:
                print(f"Skip {def_name}: no workloads")
                continue

            # 过滤
            op_type = definition.get("operator", "")
            if operator_filter and op_type not in operator_filter:
                continue

            # 转换
            case_data = self.convert_definition(definition, workloads)
            op_name = case_data["operator"]
            by_operator[op_name].append((def_name, case_data))

        # 输出 YAML 文件
        for op_name, cases in by_operator.items():
            self.write_operator_cases(op_name, cases)

        print(f"\nImport complete!")
        print(f"Generated {len(by_operator)} operator case files")
        print(f"Total definitions: {sum(len(cases) for cases in by_operator.values())}")

    def write_operator_cases(self, op_name: str, cases: List[tuple]):
        """为每个算子生成 YAML 文件

        支持两种模式：
        1. definition_mode=True: 一个 Definition 一个文件（推荐）
        2. definition_mode=False: 一个算子一个文件（兼容模式）
        """
        definition_mode = True  # 可通过参数控制

        if definition_mode:
            # 模式 1: 每个 Definition 单独输出
            for def_name, case_data in cases:
                self._write_definition_file(op_name, def_name, case_data)
        else:
            # 模式 2: 合并到一个文件（兼容旧版）
            self._write_merged_file(op_name, cases)

    def _write_definition_file(self, op_name: str, def_name: str, case_data: Dict):
        """为单个 Definition 生成 YAML 文件"""
        level = case_data["level"]
        scenarios = case_data["scenarios"]
        const_axes = case_data.get("const_axes", {})

        # 构建输出目录：basic/gemm/ 或 model/attention/
        level_dir = self.output_dir / level
        if op_name in ['mm', 'bmm', 'addmm']:
            op_dir = level_dir / "gemm"
        elif 'norm' in op_name:
            op_dir = level_dir / "norm"
        elif op_name in ['softmax', 'gelu', 'silu', 'silu_and_mul']:
            op_dir = level_dir / "activation"
        elif 'attention' in op_name or 'decode' in op_name or 'prefill' in op_name:
            op_dir = level_dir / "attention"
        elif 'moe' in op_name:
            op_dir = level_dir / "moe"
        else:
            op_dir = level_dir / "other"

        op_dir.mkdir(parents=True, exist_ok=True)

        # 构建 YAML 数据
        output_data = {
            "definition": def_name,
            "operator": op_name,
            "level": level,
            "source": "flashinfer-trace",
            "warmup": 10,
            "iters": case_data["iters"],
        }

        if const_axes:
            output_data["const_axes"] = const_axes

        output_data["workloads"] = scenarios

        # 写入文件
        output_file = op_dir / f"{def_name}.yaml"
        with open(output_file, "w") as f:
            f.write(f"# Definition: {def_name}\n")
            f.write(f"# Operator: {op_name}\n")
            f.write(f"# Source: flashinfer-trace\n")
            f.write(f"# Workloads: {len(scenarios)}\n\n")
            yaml.dump(output_data, f, sort_keys=False, allow_unicode=True)

        print(f"Written {output_file}: {len(scenarios)} workloads")

    def _write_merged_file(self, op_name: str, cases: List[tuple]):
        """合并多个 Definition 到一个文件（兼容模式）"""
        # 合并同一算子的多个 definition
        merged_scenarios = []
        const_axes_list = []
        definitions = []

        for def_name, case_data in cases:
            merged_scenarios.extend(case_data["scenarios"])
            const_axes_list.append(case_data.get("const_axes", {}))
            definitions.append(def_name)

        # 选择最常见的 const_axes
        const_axes = const_axes_list[0] if const_axes_list else {}

        output_data = {
            "operator": op_name,
            "level": cases[0][1]["level"],
            "source": "flashinfer-trace",
            "definitions": definitions,  # 记录所有来源 definition
            "warmup": 10,
            "iters": cases[0][1]["iters"],
        }

        if const_axes:
            output_data["const_axes"] = const_axes

        output_data["scenarios"] = merged_scenarios

        # 写入文件
        level_dir = self.output_dir / cases[0][1]["level"]
        level_dir.mkdir(exist_ok=True)

        output_file = level_dir / f"{op_name}_flashinfer.yaml"
        with open(output_file, "w") as f:
            yaml.dump(output_data, f, sort_keys=False, allow_unicode=True)

        print(f"Written {output_file}: {len(merged_scenarios)} scenarios from {len(cases)} definitions")


def main():
    parser = argparse.ArgumentParser(description="Import flashinfer-trace dataset")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to flashinfer-trace dataset")
    parser.add_argument("--output-dir", type=str, default="baseline/cases",
                        help="Output directory for case files")
    parser.add_argument("--operator", type=str, nargs="*",
                        help="Filter by operator types (e.g., gemm, attention)")

    args = parser.parse_args()

    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path {dataset_path} does not exist")
        print("\nTo download flashinfer-trace:")
        print("  git clone https://huggingface.co/datasets/flashinfer/flashinfer-trace")
        return 1

    importer = FlashinferTraceImporter(
        dataset_path=dataset_path,
        output_dir=Path(args.output_dir)
    )

    importer.import_all(operator_filter=args.operator)

    return 0


if __name__ == "__main__":
    exit(main())
