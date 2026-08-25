"""
Workload 生成器 - 核心协调逻辑
"""

from collections import defaultdict
from typing import List
from pathlib import Path

from .config import ModelConfig, InferenceScenario, OperatorWorkload, WorkloadSet
from .architectures import get_architecture


class WorkloadGenerator:
    """Workload 生成器

    将 (ModelConfig, InferenceScenario[]) 映射为按算子聚合的 WorkloadSet。
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        # 获取对应的架构类
        arch_class = get_architecture(config.model_type)
        self.architecture = arch_class()

    def generate(
        self,
        scenarios: List[InferenceScenario] = None
    ) -> List[WorkloadSet]:
        """生成所有 scenario 的 workload，按算子聚合

        Args:
            scenarios: 推理场景列表，默认使用标准场景

        Returns:
            按算子分组的 WorkloadSet 列表
        """
        if scenarios is None:
            scenarios = InferenceScenario.standard_scenarios()

        # 收集所有 workload
        all_workloads = []
        for scenario in scenarios:
            workloads = self.architecture.generate_all_workloads(self.config, scenario)
            all_workloads.extend(workloads)

        # 按算子分组
        grouped = defaultdict(list)
        for wl in all_workloads:
            grouped[wl.op_name].append(wl)

        # 构建 WorkloadSet
        workload_sets = []
        for op_name, workloads in grouped.items():
            ws = WorkloadSet(
                op_name=op_name,
                model_name=self.config.model_name,
                description=f"{op_name.upper()} - Traced from {self.config.model_name}",
                workloads=workloads,
            )
            workload_sets.append(ws)

        return workload_sets

    @classmethod
    def from_config_file(cls, config_path: str) -> "WorkloadGenerator":
        """从配置文件创建生成器"""
        config = ModelConfig.from_json(config_path)
        return cls(config)
