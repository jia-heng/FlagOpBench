"""
模型架构定义

每个架构类负责将 (ModelConfig, InferenceScenario) 映射为算子 workload 列表。
"""

from .base import BaseArchitecture
from .llama import LlamaArchitecture
from .deepseek_v4 import DeepSeekV4Architecture
from .qwen_hybrid import QwenHybridArchitecture

# 架构注册表
ARCHITECTURE_REGISTRY = {
    # 标准 Transformer (GQA + SwiGLU)
    "llama": LlamaArchitecture,
    "qwen2": LlamaArchitecture,       # Qwen2/2.5 Dense = Llama 架构

    # MLA + MoE
    "deepseek_v4": DeepSeekV4Architecture,
    "deepseek_v32": DeepSeekV4Architecture,  # HF model_type (DeepSeek-V3.2)
    "glm_moe_dsa": DeepSeekV4Architecture,   # GLM-5.2
    "glm4_moe_lite": DeepSeekV4Architecture, # GLM-4.7 Flash
    "kimi_k3": DeepSeekV4Architecture,       # Kimi-K3
    "kimi_k25": DeepSeekV4Architecture,      # Kimi-K2.6

    # Hybrid Attention (Linear + Full)
    "qwen3_5": QwenHybridArchitecture,          # Qwen3.5/3.6 Dense
    "qwen3_5_moe": QwenHybridArchitecture,      # Qwen3.6 MoE
    "qwen3_5_moe_text": QwenHybridArchitecture, # Qwen3.8 MoE
}


def get_architecture(model_type: str) -> type[BaseArchitecture]:
    """根据 model_type 获取对应的架构类"""
    if model_type not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unknown model_type: {model_type}. "
            f"Available: {list(ARCHITECTURE_REGISTRY.keys())}"
        )
    return ARCHITECTURE_REGISTRY[model_type]


__all__ = [
    "BaseArchitecture",
    "LlamaArchitecture",
    "DeepSeekV4Architecture",
    "QwenHybridArchitecture",
    "get_architecture",
]
