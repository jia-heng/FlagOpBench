"""
架构基类 - 定义算子推导接口
"""

from abc import ABC, abstractmethod
from typing import List
from ..config import ModelConfig, InferenceScenario, OperatorWorkload


class BaseArchitecture(ABC):
    """模型架构基类

    每个子类负责将 (ModelConfig, InferenceScenario) 映射为该场景下的所有算子 workload。
    """

    @abstractmethod
    def generate_layer_workloads(
        self,
        config: ModelConfig,
        scenario: InferenceScenario
    ) -> List[OperatorWorkload]:
        """生成单个 Transformer 层的所有算子 workload

        Args:
            config: 模型配置
            scenario: 推理场景

        Returns:
            该层所有算子的 workload 列表
        """
        pass

    def generate_all_workloads(
        self,
        config: ModelConfig,
        scenario: InferenceScenario
    ) -> List[OperatorWorkload]:
        """生成整个模型的所有算子 workload

        默认实现：单层 × num_hidden_layers
        子类可覆盖以添加 embedding/lm_head 等非层级算子。
        """
        # 生成单层 workload
        layer_workloads = self.generate_layer_workloads(config, scenario)

        # 由于每层重复，我们只保留一份（代表性 workload）
        # 如果需要考虑层间差异（如首尾层），可在此扩展
        return layer_workloads

    # 通用辅助方法

    def _norm_ops(
        self,
        config: ModelConfig,
        scenario: InferenceScenario,
        position: str
    ) -> List[OperatorWorkload]:
        """生成归一化算子 workload（RMSNorm）

        Args:
            position: 位置标识（"input" / "post_attn"）
        """
        return [OperatorWorkload(
            op_name="rms_norm",
            axes={
                "num_tokens": scenario.num_tokens,
                "hidden_size": config.hidden_size,
            },
            const_params={
                "dtype": "bf16",
                "eps": config.rms_norm_eps,
            },
            source=f"{config.model_name}/layer_{position}/{scenario.phase}",
            phase=scenario.phase,
        )]

    def _rope_ops(
        self,
        config: ModelConfig,
        scenario: InferenceScenario
    ) -> List[OperatorWorkload]:
        """生成 RoPE 算子 workload"""
        return [OperatorWorkload(
            op_name="rope",
            axes={
                "batch": scenario.batch_size,
                "seq_len": scenario.seq_len,
                "num_heads": config.num_attention_heads,
                "head_dim": config.head_dim,
            },
            const_params={
                "dtype": "bf16",
                "rope_theta": config.rope_theta,
            },
            source=f"{config.model_name}/rope/{scenario.phase}",
            phase=scenario.phase,
        )]
