"""Grouped TopK 算子

MoE 路由中的分组 TopK 选择：先在 expert groups 内选出 top group，
再从选中的 group 内选出 topk 个 experts。

签名:
    grouped_topk(
        scores, n_group, topk_group, topk,
        renormalize, routed_scaling_factor, bias,
        scoring_func=0
    ) -> (topk_weights, topk_ids)

输入:
    scores: (num_tokens, num_experts) float32   — router 输出的 logits/scores
    n_group: int                                — expert 分组数
    topk_group: int                             — 选择的 group 数
    topk: int                                   — 每 token 选择的 expert 数
    renormalize: bool                           — 是否重新归一化权重
    routed_scaling_factor: float                — routing 缩放因子
    bias: (num_experts,) float32                — e_score_correction_bias
    scoring_func: int                           — 0=none, 1=sigmoid

输出:
    topk_weights: (num_tokens, topk) float32
    topk_ids: (num_tokens, topk) int32
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("grouped_topk")
class GroupedTopkOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "grouped_topk"

    @property
    def library(self) -> str:
        return "flag_gems"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            num_tokens: token 数
            num_experts: expert 总数
            n_group: expert 分组数
            topk_group: 选择的 group 数
            topk: 每 token 选择的 expert 数
            routed_scaling_factor: routing 缩放因子
            scoring_func: 0=none, 1=sigmoid
            dtype: scores 数据类型
        """
        num_tokens = params["num_tokens"]
        num_experts = params["num_experts"]
        n_group = params["n_group"]
        topk_group = params["topk_group"]
        topk = params["topk"]
        routed_scaling_factor = params.get("routed_scaling_factor", 2.5)
        scoring_func = params.get("scoring_func", 1)  # 1=sigmoid for DeepSeek
        dtype = self.get_dtype(params.get("dtype", "fp32"))

        # scores: (num_tokens, num_experts) — router logits
        scores = torch.randn(num_tokens, num_experts, dtype=dtype, device="cuda")

        # bias: (num_experts,) — e_score_correction_bias
        bias = torch.randn(num_experts, dtype=dtype, device="cuda")

        return {
            "scores": scores,
            "n_group": n_group,
            "topk_group": topk_group,
            "topk": topk,
            "renormalize": True,
            "routed_scaling_factor": routed_scaling_factor,
            "bias": bias,
            "scoring_func": scoring_func,
        }

    def compute_flops(self, **params):
        """理论FLOPs

        主要是比较操作（sorting/selection），不是 FP FLOPS 密集型。
        粗略估算: num_tokens * num_experts * log(num_experts)
        """
        num_tokens = params["num_tokens"]
        num_experts = params["num_experts"]
        import math
        return int(num_tokens * num_experts * math.log2(num_experts))

    def compute_bytes(self, **params):
        """理论访存量

        读:
          scores: num_tokens * num_experts * elem_bytes
          bias: num_experts * elem_bytes
        写:
          topk_weights: num_tokens * topk * 4
          topk_ids: num_tokens * topk * 4
        """
        num_tokens = params["num_tokens"]
        num_experts = params["num_experts"]
        topk = params["topk"]
        dtype_str = params.get("dtype", "fp32")
        elem_bytes = self.dtype_bytes(dtype_str)

        read_bytes = (
            num_tokens * num_experts * elem_bytes  # scores
            + num_experts * elem_bytes             # bias
        )
        write_bytes = (
            num_tokens * topk * 4   # topk_weights (fp32)
            + num_tokens * topk * 4  # topk_ids (int32)
        )

        return int(read_bytes + write_bytes)
