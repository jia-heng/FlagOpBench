"""
Llama 架构实现

覆盖模型：
- Meta Llama (3.1/3.3)
- Qwen Dense (2.5/3)
- Yi (1.5)
- Baichuan (2)
- InternLM (2)
- Mistral (7B Dense)

架构特征：
- Attention: GQA (Grouped Query Attention)
- FFN: SwiGLU (gate_proj + up_proj → silu_and_mul → down_proj)
- Norm: RMSNorm
- RoPE: Standard rotary position embedding
"""

from typing import List
from .base import BaseArchitecture
from ..config import ModelConfig, InferenceScenario, OperatorWorkload


class LlamaArchitecture(BaseArchitecture):
    """Llama 系列架构（标准 Transformer）"""

    def generate_layer_workloads(
        self,
        config: ModelConfig,
        scenario: InferenceScenario
    ) -> List[OperatorWorkload]:
        """生成单层所有算子 workload"""
        workloads = []

        # 1. Input RMSNorm
        workloads.extend(self._norm_ops(config, scenario, "input"))

        # 2. Attention Block
        workloads.extend(self._attention_ops(config, scenario))

        # 3. Post-Attention RMSNorm
        workloads.extend(self._norm_ops(config, scenario, "post_attn"))

        # 4. FFN Block
        workloads.extend(self._ffn_ops(config, scenario))

        return workloads

    def _attention_ops(
        self,
        config: ModelConfig,
        scenario: InferenceScenario
    ) -> List[OperatorWorkload]:
        """生成 Attention 相关算子 workload"""
        workloads = []
        num_tokens = scenario.num_tokens
        head_dim = config.head_dim

        # QKV Projection (单个 GEMM，输出包含 Q + K + V)
        # Q: num_attention_heads * head_dim
        # K: num_key_value_heads * head_dim
        # V: num_key_value_heads * head_dim
        kv_size = config.num_key_value_heads * head_dim
        qkv_size = config.num_attention_heads * head_dim + 2 * kv_size

        workloads.append(OperatorWorkload(
            op_name="mm",
            axes={
                "M": num_tokens,
                "K": config.hidden_size,
                "N": qkv_size,
            },
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/qkv_proj/{scenario.phase}",
            phase=scenario.phase,
        ))

        # RoPE (应用到 Q 和 K)
        workloads.extend(self._rope_ops(config, scenario))

        # Attention Core
        if scenario.phase == "prefill":
            # Prefill: Flash Attention (causal)
            workloads.append(OperatorWorkload(
                op_name="flash_attention",
                axes={
                    "batch": scenario.batch_size,
                    "seq_len": scenario.seq_len,
                    "num_heads": config.num_attention_heads,
                    "head_dim": head_dim,
                },
                const_params={
                    "dtype": "bf16",
                    "causal": True,
                },
                source=f"{config.model_name}/attention_prefill",
                phase="prefill",
            ))
        else:
            # Decode: Paged Attention (single token attending to KV cache)
            workloads.append(OperatorWorkload(
                op_name="paged_attention_decode",
                axes={
                    "batch": scenario.batch_size,
                    "num_qo_heads": config.num_attention_heads,
                    "num_kv_heads": config.num_key_value_heads,
                    "head_dim": head_dim,
                    "kv_len": scenario.kv_len,
                },
                const_params={
                    "dtype": "bf16",
                    "page_size": 16,  # vLLM 默认 page_size
                },
                source=f"{config.model_name}/attention_decode",
                phase="decode",
            ))

        # O Projection (Attention output projection)
        workloads.append(OperatorWorkload(
            op_name="mm",
            axes={
                "M": num_tokens,
                "K": config.hidden_size,
                "N": config.hidden_size,
            },
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/o_proj/{scenario.phase}",
            phase=scenario.phase,
        ))

        return workloads

    def _ffn_ops(
        self,
        config: ModelConfig,
        scenario: InferenceScenario
    ) -> List[OperatorWorkload]:
        """生成 FFN 相关算子 workload (SwiGLU)

        SwiGLU FFN:
        1. gate_proj: Linear(hidden → intermediate)
        2. up_proj:   Linear(hidden → intermediate)
        3. silu_and_mul: SiLU(gate) * up
        4. down_proj: Linear(intermediate → hidden)
        """
        workloads = []
        num_tokens = scenario.num_tokens

        # Gate + Up Projection (可以合并为单个 GEMM，输出 2 * intermediate_size)
        # vLLM 实现中通常分开，这里先分开实现

        # Gate Projection
        workloads.append(OperatorWorkload(
            op_name="mm",
            axes={
                "M": num_tokens,
                "K": config.hidden_size,
                "N": config.intermediate_size,
            },
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/gate_proj/{scenario.phase}",
            phase=scenario.phase,
        ))

        # Up Projection
        workloads.append(OperatorWorkload(
            op_name="mm",
            axes={
                "M": num_tokens,
                "K": config.hidden_size,
                "N": config.intermediate_size,
            },
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/up_proj/{scenario.phase}",
            phase=scenario.phase,
        ))

        # SiLU and Mul (fused activation)
        workloads.append(OperatorWorkload(
            op_name="silu_and_mul",
            axes={
                "M": num_tokens,
                "N": config.intermediate_size,
            },
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/silu_and_mul/{scenario.phase}",
            phase=scenario.phase,
        ))

        # Down Projection
        workloads.append(OperatorWorkload(
            op_name="mm",
            axes={
                "M": num_tokens,
                "K": config.intermediate_size,
                "N": config.hidden_size,
            },
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/down_proj/{scenario.phase}",
            phase=scenario.phase,
        ))

        return workloads
