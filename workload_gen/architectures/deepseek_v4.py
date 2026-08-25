"""
DeepSeek-V4 / MLA + MoE 架构实现

覆盖模型：
- DeepSeek-V4 Flash (4096h, 256e, top-6)
- DeepSeek-V4 Pro (7168h, 384e, top-6)
- GLM-4.7 Flash (2048h, 64e, top-4)
- GLM-5.2 (6144h, 256e, top-8)
- Kimi-K2.6 (7168h, 384e, top-8)
- Kimi-K3 (7168h, 896e, top-?)

架构特征：
- Attention: MLA (Multi-head Latent Attention)
  - Q: x → q_lora_down(hidden→q_lora_rank) → q_lora_up(q_lora_rank→num_heads*head_dim)
  - KV: x → kv_lora_down(hidden→kv_lora_rank) → kv_lora_up → split K,V
  - RoPE 只应用在 qk_rope_head_dim 维度
- FFN: MoE (n_routed_experts, top-k routing) + shared expert
- Norm: RMSNorm
"""

from typing import List
from .base import BaseArchitecture
from ..config import ModelConfig, InferenceScenario, OperatorWorkload


class DeepSeekV4Architecture(BaseArchitecture):
    """DeepSeek-V4 / MLA + MoE 架构"""

    def generate_layer_workloads(
        self,
        config: ModelConfig,
        scenario: InferenceScenario
    ) -> List[OperatorWorkload]:
        """生成单层所有算子 workload"""
        workloads = []

        # 1. Input RMSNorm
        workloads.extend(self._norm_ops(config, scenario, "input"))

        # 2. MLA Attention Block
        workloads.extend(self._mla_attention_ops(config, scenario))

        # 3. Post-Attention RMSNorm
        workloads.extend(self._norm_ops(config, scenario, "post_attn"))

        # 4. MoE FFN Block
        workloads.extend(self._moe_ffn_ops(config, scenario))

        return workloads

    def _mla_attention_ops(
        self,
        config: ModelConfig,
        scenario: InferenceScenario
    ) -> List[OperatorWorkload]:
        """MLA Attention 算子

        MLA 流程：
        1. Q LoRA: x(hidden) → q_lora_down(q_lora_rank) → RMSNorm → q_lora_up(num_heads * head_dim)
        2. KV LoRA: x(hidden) → kv_lora_down(kv_lora_rank) → RMSNorm → kv_lora_up(K + V)
        3. RoPE (仅 qk_rope_head_dim 维度)
        4. Attention Core
        5. O Projection: attn_out → o_proj(hidden)
        """
        workloads = []
        num_tokens = scenario.num_tokens
        q_lora_rank = config.q_lora_rank or 1536
        kv_lora_rank = config.kv_lora_rank or 512
        qk_rope_head_dim = config.qk_rope_head_dim or 64
        head_dim = config.head_dim  # 通常是 v_head_dim (128)

        # --- Q Path ---
        # Q LoRA Down: hidden → q_lora_rank
        workloads.append(OperatorWorkload(
            op_name="mm",
            axes={"M": num_tokens, "K": config.hidden_size, "N": q_lora_rank},
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/q_lora_down/{scenario.phase}",
            phase=scenario.phase,
        ))

        # Q LoRA RMSNorm
        workloads.append(OperatorWorkload(
            op_name="rms_norm",
            axes={"num_tokens": num_tokens, "hidden_size": q_lora_rank},
            const_params={"dtype": "bf16", "eps": config.rms_norm_eps},
            source=f"{config.model_name}/q_lora_norm/{scenario.phase}",
            phase=scenario.phase,
        ))

        # Q LoRA Up: q_lora_rank → num_heads * (head_dim + qk_rope_head_dim)
        # 实际 Q 输出包括 nope 部分 + rope 部分
        q_proj_size = config.num_attention_heads * (head_dim + qk_rope_head_dim)
        workloads.append(OperatorWorkload(
            op_name="mm",
            axes={"M": num_tokens, "K": q_lora_rank, "N": q_proj_size},
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/q_lora_up/{scenario.phase}",
            phase=scenario.phase,
        ))

        # --- KV Path ---
        # KV LoRA Down: hidden → kv_lora_rank + qk_rope_head_dim
        kv_down_out = kv_lora_rank + qk_rope_head_dim
        workloads.append(OperatorWorkload(
            op_name="mm",
            axes={"M": num_tokens, "K": config.hidden_size, "N": kv_down_out},
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/kv_lora_down/{scenario.phase}",
            phase=scenario.phase,
        ))

        # KV LoRA RMSNorm (on kv_lora_rank part)
        workloads.append(OperatorWorkload(
            op_name="rms_norm",
            axes={"num_tokens": num_tokens, "hidden_size": kv_lora_rank},
            const_params={"dtype": "bf16", "eps": config.rms_norm_eps},
            source=f"{config.model_name}/kv_lora_norm/{scenario.phase}",
            phase=scenario.phase,
        ))

        # KV LoRA Up: kv_lora_rank → num_heads * (head_dim + head_dim)
        # 输出 K_nope + V，每个 head: head_dim for K_nope, head_dim for V
        kv_up_out = config.num_attention_heads * head_dim * 2
        workloads.append(OperatorWorkload(
            op_name="mm",
            axes={"M": num_tokens, "K": kv_lora_rank, "N": kv_up_out},
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/kv_lora_up/{scenario.phase}",
            phase=scenario.phase,
        ))

        # --- RoPE (仅 qk_rope_head_dim 维度) ---
        workloads.append(OperatorWorkload(
            op_name="rope",
            axes={
                "batch": scenario.batch_size,
                "seq_len": scenario.seq_len,
                "num_heads": config.num_attention_heads,
                "head_dim": qk_rope_head_dim,
            },
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/rope/{scenario.phase}",
            phase=scenario.phase,
        ))

        # --- Attention Core ---
        # MLA attention: head_dim = v_head_dim + qk_rope_head_dim (for Q/K)
        # 但 attention score 计算用完整 head_dim
        attn_head_dim = head_dim + qk_rope_head_dim
        if scenario.phase == "prefill":
            workloads.append(OperatorWorkload(
                op_name="flash_attention",
                axes={
                    "batch": scenario.batch_size,
                    "seq_len": scenario.seq_len,
                    "num_heads": config.num_attention_heads,
                    "head_dim": attn_head_dim,
                },
                const_params={"dtype": "bf16", "causal": True},
                source=f"{config.model_name}/mla_attention_prefill",
                phase="prefill",
            ))
        else:
            workloads.append(OperatorWorkload(
                op_name="flash_mla",
                axes={
                    "batch": scenario.batch_size,
                    "num_heads": config.num_attention_heads,
                    "head_dim": attn_head_dim,
                    "kv_len": scenario.kv_len,
                    "kv_lora_rank": kv_lora_rank,
                },
                const_params={"dtype": "bf16"},
                source=f"{config.model_name}/mla_attention_decode",
                phase="decode",
            ))

        # --- O Projection ---
        # o_proj: num_heads * v_head_dim → hidden_size
        # 部分模型有 o_lora_rank，但大多数直接投影
        o_proj_in = config.num_attention_heads * head_dim
        workloads.append(OperatorWorkload(
            op_name="mm",
            axes={"M": num_tokens, "K": o_proj_in, "N": config.hidden_size},
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/o_proj/{scenario.phase}",
            phase=scenario.phase,
        ))

        return workloads

    def _moe_ffn_ops(
        self,
        config: ModelConfig,
        scenario: InferenceScenario
    ) -> List[OperatorWorkload]:
        """MoE FFN 算子

        MoE 流程：
        1. Router: x → router_logits (num_experts)
        2. TopK routing
        3. MoE 计算 (fused_moe 或 grouped_matmul)
        4. Shared Expert (如果有)
        """
        workloads = []
        num_tokens = scenario.num_tokens
        num_experts = config.num_experts or 256
        top_k = config.num_experts_per_tok or 8
        moe_ff = config.moe_intermediate_size or 2048

        # Router GEMM: hidden → num_experts
        workloads.append(OperatorWorkload(
            op_name="mm",
            axes={"M": num_tokens, "K": config.hidden_size, "N": num_experts},
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/moe_router/{scenario.phase}",
            phase=scenario.phase,
        ))

        # TopK + Softmax (router_gemm / topk_softplus_sqrt)
        workloads.append(OperatorWorkload(
            op_name="softmax",
            axes={"M": num_tokens, "N": num_experts},
            const_params={"dtype": "bf16"},
            source=f"{config.model_name}/moe_softmax/{scenario.phase}",
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
            source=f"{config.model_name}/moe_align/{scenario.phase}",
            phase=scenario.phase,
        ))

        # Fused MoE (核心算子)
        # 每个 expert: gate+up → silu_and_mul → down
        # 等效: x(hidden) → expert_weights(hidden, moe_ff*2) → silu → down(moe_ff, hidden)
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
            source=f"{config.model_name}/fused_moe/{scenario.phase}",
            phase=scenario.phase,
        ))

        # Shared Expert (如果存在)
        n_shared = getattr(config, 'n_shared_experts', None)
        if n_shared and n_shared > 0:
            # Shared expert 是标准 SwiGLU FFN
            shared_ff = config.intermediate_size or (moe_ff * 4)
            workloads.append(OperatorWorkload(
                op_name="mm",
                axes={"M": num_tokens, "K": config.hidden_size, "N": shared_ff},
                const_params={"dtype": "bf16"},
                source=f"{config.model_name}/shared_expert_gate_up/{scenario.phase}",
                phase=scenario.phase,
            ))
            workloads.append(OperatorWorkload(
                op_name="silu_and_mul",
                axes={"M": num_tokens, "N": shared_ff},
                const_params={"dtype": "bf16"},
                source=f"{config.model_name}/shared_expert_act/{scenario.phase}",
                phase=scenario.phase,
            ))
            workloads.append(OperatorWorkload(
                op_name="mm",
                axes={"M": num_tokens, "K": shared_ff, "N": config.hidden_size},
                const_params={"dtype": "bf16"},
                source=f"{config.model_name}/shared_expert_down/{scenario.phase}",
                phase=scenario.phase,
            ))

        return workloads
