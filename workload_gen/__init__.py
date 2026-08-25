"""
FlagOpBench Workload Generator

从模型配置推导真实推理 workload，无需部署模型。
"""

from .config import ModelConfig, InferenceScenario
from .generator import WorkloadGenerator

__all__ = ["ModelConfig", "InferenceScenario", "WorkloadGenerator"]
