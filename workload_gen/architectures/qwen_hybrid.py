"""
Qwen3.5/3.6/3.8 Hybrid Attention 架构实现

覆盖模型：
- Qwen3.5-9B (4096h, GQA 16/4, 32 layers, Linear+Full 1:4)
- Qwen3.6-27B (5120h, GQA 24/4, 64 layers, Linear+Full 1:4)
- Qwen3.6-35B-A3B (2048h, GQA 16/2, 40 layers, MoE 256e/top-8, Linear+Full 1:4)
- Qwen3.8-2.4T-A95B (8192h, GQA 64/4, 92 layers, MoE 512e/top-10, Linear+Full 1:4)
- Qwen3.8-27B (5120h, GQA 24/4, 64 layers, Linear+Full 1:4)

架构特征：
- Attention 混合：
  - 3/4 层使用 Linear Attention (类似 GLA/Mamba-style，O(n) 复杂度)
  - 1/4 层使用 Full Attention (标准 Softmax Attention with GQA)
  - 比例由 full_attention_interval 控制
- FFN: SwiGLU (Dense) 或 MoE (部分模型)
- Norm: RMSNorm
- head_dim: 256 (比标准 128 更大)
"""

from typing import List
from .base import BaseArchitecture
from ..config import ModelConfig, InferenceScenario, OperatorWorkload


class QwenHybridArchitecture(BaseArchitecture):
    """Qwen3.5/3.6/3.8 Hybrid Attention 架构"""

    def generate_layer_workloads(
        self,
        config: ModelConfig,
        scenario: InferenceScenario
    ) -> List[OperatorWorkload]:
        """生成两种层类型的 workload（按比例）

        对于 full_attention_interval=4:
        - 3/4 层为 Linear Attention 层
        - 1/4 层为 Full Attention 层
        都会生成，标注不同 source
        """
        workloads = []

        # Full Attention 层的 workload
        workloads.extend(self._full_attention_layer(config, scenario))

        # Linear Attention 层的 workload
        workloads.extend(self._linear_attention_layer(config, scenario))

        return workloads

    def _full_attention_layer(
        self,
        config: ModelConfig,
        scenario: InferenceScenario
    ) -> List[OperatorWorkload]:
        """Full Attention 层（标准 GQA + SwiGLU/MoE）"""
        workloads = []
        num_tokens = scenario.num_tokens
        head_dim = config.head_dim

        # 1. Input RMSNorm
        workloads.extend(self._norm_ops(config, scenario, "full_attn_input"))

        # 2. QKV Projection (GQA)
        kv_size = config.num_key_value_heads * head_dim
        qkv_size = config.num_attention_heads * head_dim + 2 * kv_size
        workloads.append(OperatorWorkload(
            op_name="mm",
            axes={"M": num_tokens, "K": config.hidden_size, "N": qkv_size},
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/full_attn/qkv_proj/{scenario.phase}",
            phase=scenario.phase,
        ))

        # 3. RoPE
        workloads.append(OperatorWorkload(
            op_name="rope",
            axes={
                "batch": scenario.batch_size,
                "seq_len": scenario.seq_len,
                "num_heads": config.num_attention_heads,
                "head_dim": head_dim,
            },
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/full_attn/rope/{scenario.phase}",
            phase=scenario.phase,
        ))

        # 4. Attention Core
        if scenario.phase == "prefill":
            workloads.append(OperatorWorkload(
                op_name="flash_attention",
                axes={
                    "batch": scenario.batch_size,
                    "seq_len": scenario.seq_len,
                    "num_heads": config.num_attention_heads,
                    "head_dim": head_dim,
                },
                const_params={"dtype": "bf16", "causal": True},
                source=f"{config.model_name}/full_attn/attention_prefill",
                phase="prefill",
            ))
        else:
            workloads.append(OperatorWorkload(
                op_name="paged_attention_decode",
                axes={
                    "batch": scenario.batch_size,
                    "num_qo_heads": config.num_attention_heads,
                    "num_kv_heads": config.num_key_value_heads,
                    "head_dim": head_dim,
                    "kv_len": scenario.kv_len,
                },
                const_params={"dtype": "bf16"},
                source=f"{config.model_name}/full_attn/attention_decode",
                phase="decode",
            ))

        # 5. O Projection
        workloads.append(OperatorWorkload(
            op_name="mm",
            axes={"M": num_tokens, "K": config.num_attention_heads * head_dim, "N": config.hidden_size},
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/full_attn/o_proj/{scenario.phase}",
            phase=scenario.phase,
        ))

        # 6. Post-Attention RMSNorm
        workloads.extend(self._norm_ops(config, scenario, "full_attn_post"))

        # 7. FFN
        workloads.extend(self._ffn_ops(config, scenario, "full_attn"))

        return workloads

    def _linear_attention_layer(
        self,
        config: ModelConfig,
        scenario: InferenceScenario
    ) -> List[OperatorWorkload]:
        """Linear Attention 层

        Linear Attention (GLA-style):
        - QKV Projection: 与 Full Attention 相同
        - 无 Softmax，使用线性 recurrence（chunk-wise 或 fused_recurrent）
        - Prefill: chunk kernel (O(n*chunk_size))
        - Decode: recurrent state update (O(1) per token)
        """
        workloads = []
        num_tokens = scenario.num_tokens
        head_dim = config.head_dim

        # 1. Input RMSNorm
        workloads.extend(self._norm_ops(config, scenario, "linear_attn_input"))

        # 2. QKV Projection (same shape as full attention)
        kv_size = config.num_key_value_heads * head_dim
        qkv_size = config.num_attention_heads * head_dim + 2 * kv_size
        workloads.append(OperatorWorkload(
            op_name="mm",
            axes={"M": num_tokens, "K": config.hidden_size, "N": qkv_size},
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/linear_attn/qkv_proj/{scenario.phase}",
            phase=scenario.phase,
        ))

        # 3. Linear Attention Core (不同于 softmax attention)
        # Prefill: chunk-wise parallel computation
        # Decode: recurrent state update
        workloads.append(OperatorWorkload(
            op_name="flash_linear_attention",
            axes={
                "batch": scenario.batch_size,
                "seq_len": scenario.seq_len,
                "num_heads": config.num_attention_heads,
                "head_dim": head_dim,
            },
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/linear_attn/core/{scenario.phase}",
            phase=scenario.phase,
        ))

        # 4. O Projection
        workloads.append(OperatorWorkload(
            op_name="mm",
            axes={"M": num_tokens, "K": config.num_attention_heads * head_dim, "N": config.hidden_size},
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/linear_attn/o_proj/{scenario.phase}",
            phase=scenario.phase,
        ))

        # 5. Post-Attention RMSNorm
        workloads.extend(self._norm_ops(config, scenario, "linear_attn_post"))

        # 6. FFN
        workloads.extend(self._ffn_ops(config, scenario, "linear_attn"))

        return workloads

    def _ffn_ops(
        self,
        config: ModelConfig,
        scenario: InferenceScenario,
        layer_type: str
    ) -> List[OperatorWorkload]:
        """FFN 算子（Dense SwiGLU 或 MoE）"""
        num_tokens = scenario.num_tokens

        # 判断是否 MoE
        if config.is_moe:
            return self._moe_ffn_ops(config, scenario, layer_type)
        else:
            return self._dense_ffn_ops(config, scenario, layer_type)

    def _dense_ffn_ops(
        self,
        config: ModelConfig,
        scenario: InferenceScenario,
        layer_type: str
    ) -> List[OperatorWorkload]:
        """Dense SwiGLU FFN"""
        num_tokens = scenario.num_tokens

        return [
            # Gate Projection
            OperatorWorkload(
                op_name="mm",
                axes={"M": num_tokens, "K": config.hidden_size, "N": config.intermediate_size},
                const_params={"dtype": "bf16"},
                source=f"{config.model_name}/{layer_type}/gate_proj/{scenario.phase}",
                phase=scenario.phase,
            ),
            # Up Projection
            OperatorWorkload(
                op_name="mm",
                axes={"M": num_tokens, "K": config.hidden_size, "N": config.intermediate_size},
                const_params={"dtype": "bf16"},
                source=f"{config.model_name}/{layer_type}/up_proj/{scenario.phase}",
                phase=scenario.phase,
            ),
            # SiLU and Mul
            OperatorWorkload(
                op_name="silu_and_mul",
                axes={"M": num_tokens, "N": config.intermediate_size},
                const_params={"dtype": "bf16"},
                source=f"{config.model_name}/{layer_type}/silu_and_mul/{scenario.phase}",
                phase=scenario.phase,
            ),
            # Down Projection
            OperatorWorkload(
                op_name="mm",
                axes={"M": num_tokens, "K": config.intermediate_size, "N": config.hidden_size},
                const_params={"dtype": "bf16"},
                source=f"{config.model_name}/{layer_type}/down_proj/{scenario.phase}",
                phase=scenario.phase,
            ),
        ]

    def _moe_ffn_ops(
        self,
        config: ModelConfig,
        scenario: InferenceScenario,
        layer_type: str
    ) -> List[OperatorWorkload]:
        """MoE FFN"""
        num_tokens = scenario.num_tokens
        num_experts = config.num_experts or 256
        top_k = config.num_experts_per_tok or 8
        moe_ff = config.moe_intermediate_size or (config.intermediate_size // 4)

        workloads = []

        # Router: hidden → num_experts
        workloads.append(OperatorWorkload(
            op_name="mm",
            axes={"M": num_tokens, "K": config.hidden_size, "N": num_experts},
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/{layer_type}/router/{scenario.phase}",
            phase=scenario.phase,
        ))

        # TopK routing + softmax
        workloads.append(OperatorWorkload(
            op_name="topk_selector",
            axes={"M": num_tokens, "N": num_experts, "k": top_k},
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/{layer_type}/topk/{scenario.phase}",
            phase=scenario.phase,
        ))

        # MoE Align Block Size
        workloads.append(OperatorWorkload(
            op_name="moe_align_block_size",
            axes={
                "num_tokens": num_tokens,
                "num_experts": num_experts,
                "top_k": top_k,
                "block_size": 64,
            },
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/{layer_type}/moe_align/{scenario.phase}",
            phase=scenario.phase,
        ))

        # Fused MoE (experts computation)
        workloads.append(OperatorWorkload(
            op_name="fused_moe",
            axes={
                "num_tokens": num_tokens,
                "hidden_size": config.hidden_size,
                "expert_size": moe_ff,
                "num_experts": num_experts,
                "top_k": top_k,
            },
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/{layer_type}/fused_moe/{scenario.phase}",
            phase=scenario.phase,
        ))

        # Shared Expert (if present)
        shared_ff = getattr(config, 'shared_expert_intermediate_size', None) or config.intermediate_size
        if shared_ff and shared_ff > 0:
            workloads.append(OperatorWorkload(
                op_name="mm",
                axes={"M": num_tokens, "K": config.hidden_size, "N": shared_ff * 2},
                const_params={"dtype": "bf16"},
                source=f"{config.model_name}/{layer_type}/shared_expert_gate_up/{scenario.phase}",
                phase=scenario.phase,
            ))
            workloads.append(OperatorWorkload(
                op_name="silu_and_mul",
                axes={"M": num_tokens, "N": shared_ff},
                const_params={"dtype": "bf16"},
                source=f"{config.model_name}/{layer_type}/shared_expert_act/{scenario.phase}",
                phase=scenario.phase,
            ))
            workloads.append(OperatorWorkload(
                op_name="mm",
                axes={"M": num_tokens, "K": shared_ff, "N": config.hidden_size},
                const_params={"dtype": "bf16"},
                source=f"{config.model_name}/{layer_type}/shared_expert_down/{scenario.phase}",
                phase=scenario.phase,
            ))

        return workloads
