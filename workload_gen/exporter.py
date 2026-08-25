"""
Workload 导出器 - 生成 YAML case 文件
"""

import yaml
from pathlib import Path
from typing import List

from .config import WorkloadSet


class YAMLExporter:
    """导出为 FlagOpBench YAML case 格式"""

    def __init__(self, output_dir: str):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def export(self, workload_sets: List[WorkloadSet]):
        """导出所有 WorkloadSet 为 YAML 文件

        每个 WorkloadSet 对应一个 YAML 文件，文件名为 {op_name}_{model_name}.yaml
        """
        exported_files = []

        for ws in workload_sets:
            # 生成文件名
            filename = f"{ws.op_name}_{ws.model_name}.yaml"
            filepath = self.output_dir / filename

            # 转换为 YAML 字典
            yaml_dict = ws.to_yaml_dict()

            # 写入文件
            with open(filepath, "w") as f:
                yaml.dump(
                    yaml_dict,
                    f,
                    sort_keys=False,
                    allow_unicode=True,
                    default_flow_style=False,
                )

            exported_files.append(filepath)
            print(f"  ✓ {filepath} ({len(ws.workloads)} workloads)")

        return exported_files

    def export_single(self, workload_set: WorkloadSet, filename: str = None):
        """导出单个 WorkloadSet"""
        if filename is None:
            filename = f"{workload_set.op_name}_{workload_set.model_name}.yaml"

        filepath = self.output_dir / filename
        yaml_dict = workload_set.to_yaml_dict()

        with open(filepath, "w") as f:
            yaml.dump(
                yaml_dict,
                f,
                sort_keys=False,
                allow_unicode=True,
                default_flow_style=False,
            )

        print(f"  ✓ {filepath} ({len(workload_set.workloads)} workloads)")
        return filepath
