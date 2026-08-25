"""
模型架构定义

每个架构类负责将 (ModelConfig, InferenceScenario) 映射为算子 workload 列表。
"""

from .base import BaseArchitecture
from .llama import LlamaArchitecture

# 架构注册表
ARCHITECTURE_REGISTRY = {
    "llama": LlamaArchitecture,
    # 后续添加：
    # "mixtral": MixtralArchitecture,
    # "qwen_moe": QwenMoEArchitecture,
    # "deepseek_v3": DeepSeekV3Architecture,
}


def get_architecture(model_type: str) -> type[BaseArchitecture]:
    """根据 model_type 获取对应的架构类"""
    if model_type not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available: {list(ARCHITECTURE_REGISTRY.keys())}"
        )
    return ARCHITECTURE_REGISTRY[model_type]


__all__ = ["BaseArchitecture", "LlamaArchitecture", "get_architecture"]
