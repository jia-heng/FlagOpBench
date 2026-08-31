"""TopK Softplus Sqrt算子

融合的 TopK选择 + Softplus激活 + Sqrt归一化，用于MoE门控。

功能:
    从gating_output中选出topk个专家，计算softplus权重并做sqrt归一化。

签名:
    topk_softplus_sqrt(topk_weights, topk_indices, token_expert_indices,
                       gating_output, renormalize, routed_scaling_factor,
                       correction_bias=None, input_ids=None, tid2eid=None)

输入:
    gating_output: (num_tokens, num_experts) — 门控logits (float32)
    topk_weights: (num_tokens, topk)         — 输出权重 (float32, 预分配)
    topk_indices: (num_tokens, topk)         — 输出专家索引 (int32, 预分配)
    token_expert_indices: (num_tokens, topk) — 输出token-expert索引 (int32, 预分配)
    renormalize: bool                        — 是否重新归一化权重
    routed_scaling_factor: float             — 缩放因子

输出:
    写入topk_weights, topk_indices, token_expert_indices (in-place)
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("topk_softplus_sqrt")
class TopkSoftplusSqrtOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "topk_softplus_sqrt"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            num_tokens: token数
            num_experts: 专家数（DeepSeek-V3为256）
            topk: 每个token选择的专家数（DeepSeek-V3为8）
            renormalize: 是否重新归一化
            routed_scaling_factor: 缩放因子
        """
        num_tokens = params["num_tokens"]
        num_experts = params.get("num_experts", 256)
        topk = params.get("topk", 8)
        renormalize = params.get("renormalize", True)
        routed_scaling_factor = params.get("routed_scaling_factor", 1.0)

        gating_output = torch.randn(
            num_tokens, num_experts, dtype=torch.float32, device="cuda"
        )
        topk_weights = torch.empty(
            num_tokens, topk, dtype=torch.float32, device="cuda"
        )
        topk_indices = torch.empty(
            num_tokens, topk, dtype=torch.int32, device="cuda"
        )
        token_expert_indices = torch.empty(
            num_tokens, topk, dtype=torch.int32, device="cuda"
        )

        return {
            "topk_weights": topk_weights,
            "topk_indices": topk_indices,
            "token_expert_indices": token_expert_indices,
            "gating_output": gating_output,
            "renormalize": renormalize,
            "routed_scaling_factor": routed_scaling_factor,
        }

    def compute_flops(self, **params):
        """理论FLOPs

        对于每个token:
          - softplus(x) = log(1+exp(x)): num_experts * ~4 ops
          - topk选择: num_experts * ~2 (比较+交换)
          - sqrt归一化: topk * 3 (sum, sqrt, div)
        总: num_tokens * (num_experts * 6 + topk * 3)
        """
        num_tokens = params["num_tokens"]
        num_experts = params.get("num_experts", 256)
        topk = params.get("topk", 8)

        return num_tokens * (num_experts * 6 + topk * 3)

    def compute_bytes(self, **params):
        """理论访存量

        读:
          gating_output: num_tokens * num_experts * 4 (float32)
        写:
          topk_weights: num_tokens * topk * 4 (float32)
          topk_indices: num_tokens * topk * 4 (int32)
          token_expert_indices: num_tokens * topk * 4 (int32)
        """
        num_tokens = params["num_tokens"]
        num_experts = params.get("num_experts", 256)
        topk = params.get("topk", 8)

        read_bytes = num_tokens * num_experts * 4
        write_bytes = num_tokens * topk * 4 * 3  # 3 output tensors

        return int(read_bytes + write_bytes)
