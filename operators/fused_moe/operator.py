"""Fused MoE算子 (Mixture of Experts)

基于Triton的融合MoE GEMM实现，支持:
  - TopK routing + expert GEMM (gate_up * w1 -> activation -> w2)
  - BF16/FP8/INT8权重
  - SiLU/GeGLU激活

签名:
    outplace_fused_experts(
        hidden_states, w1, w2, topk_weights, topk_ids,
        activation="silu", ...
    ) -> output

输入:
    hidden_states: (num_tokens, hidden_size) bf16     — 输入hidden states
    w1: (num_experts, intermediate_size*2, hidden_size) bf16 — gate+up权重
    w2: (num_experts, hidden_size, intermediate_size) bf16  — down权重
    topk_weights: (num_tokens, topk) fp32             — routing权重
    topk_ids: (num_tokens, topk) int32                — expert索引

输出:
    output: (num_tokens, hidden_size) bf16
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("fused_moe")
class FusedMoeOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "fused_moe"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    @property
    def impl_name(self) -> str:
        """实际函数名，与注册名不同"""
        return "outplace_fused_experts"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            num_tokens: token数
            hidden_size: 隐藏层大小
            intermediate_size: MLP中间层大小
            num_experts: expert总数
            topk: 每个token选择的expert数
            dtype: 数据类型
        """
        num_tokens = params["num_tokens"]
        hidden_size = params.get("hidden_size", 7168)
        intermediate_size = params.get("intermediate_size", 2048)
        num_experts = params.get("num_experts", 256)
        topk = params.get("topk", 8)
        dtype_str = params.get("dtype", "bf16")
        dtype = self.get_dtype(dtype_str)

        # hidden_states: (num_tokens, hidden_size)
        hidden_states = torch.randn(
            num_tokens, hidden_size,
            dtype=dtype, device="cuda"
        )

        # w1: (num_experts, intermediate_size*2, hidden_size) — gate+up fused
        w1 = torch.randn(
            num_experts, intermediate_size * 2, hidden_size,
            dtype=dtype, device="cuda"
        )

        # w2: (num_experts, hidden_size, intermediate_size)
        w2 = torch.randn(
            num_experts, hidden_size, intermediate_size,
            dtype=dtype, device="cuda"
        )

        # topk_weights: (num_tokens, topk) fp32
        topk_weights = torch.rand(
            num_tokens, topk,
            dtype=torch.float32, device="cuda"
        )
        # 归一化
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # topk_ids: (num_tokens, topk) int32
        topk_ids = torch.randint(
            0, num_experts, (num_tokens, topk),
            dtype=torch.int32, device="cuda"
        )

        return {
            "hidden_states": hidden_states,
            "w1": w1,
            "w2": w2,
            "topk_weights": topk_weights,
            "topk_ids": topk_ids,
            "activation": "silu",
        }

    def compute_flops(self, **params):
        """理论FLOPs

        每个token被路由到topk个expert:
          - GEMM1 (hidden -> intermediate*2): 2 * hidden_size * intermediate_size * 2
          - Activation (SiLU + gate mul): ~intermediate_size * 2
          - GEMM2 (intermediate -> hidden): 2 * intermediate_size * hidden_size
        Total per token: topk * (4 * hidden_size * intermediate_size + 2 * intermediate_size * hidden_size)
             = topk * 6 * hidden_size * intermediate_size
        """
        num_tokens = params["num_tokens"]
        hidden_size = params.get("hidden_size", 7168)
        intermediate_size = params.get("intermediate_size", 2048)
        topk = params.get("topk", 8)

        # GEMM1: (M, K) x (K, N) → 2*M*K*N where K=hidden, N=inter*2
        # GEMM2: (M, K) x (K, N) → 2*M*K*N where K=inter, N=hidden
        flops_per_token = topk * (
            2 * hidden_size * intermediate_size * 2  # w1: gate+up
            + 2 * intermediate_size * hidden_size    # w2: down
        )
        return num_tokens * flops_per_token

    def compute_bytes(self, **params):
        """理论访存量

        读:
          hidden_states: num_tokens * hidden_size * elem_bytes
          w1 (active experts): topk * intermediate_size*2 * hidden_size * elem_bytes
          w2 (active experts): topk * hidden_size * intermediate_size * elem_bytes
          topk_weights: num_tokens * topk * 4
          topk_ids: num_tokens * topk * 4
        写:
          output: num_tokens * hidden_size * elem_bytes
          intermediate: num_tokens * topk * intermediate_size * elem_bytes (内部)
        """
        num_tokens = params["num_tokens"]
        hidden_size = params.get("hidden_size", 7168)
        intermediate_size = params.get("intermediate_size", 2048)
        num_experts = params.get("num_experts", 256)
        topk = params.get("topk", 8)
        dtype_str = params.get("dtype", "bf16")
        elem_bytes = self.dtype_bytes(dtype_str)

        # 实际访问的expert权重 (假设topk个不同expert)
        # 但多个token可能复用同一expert，这里按最坏情况估算
        # 更现实的估算: min(num_tokens * topk, num_experts) 个expert被激活
        active_experts = min(num_tokens * topk, num_experts)

        read_bytes = (
            num_tokens * hidden_size * elem_bytes               # hidden_states
            + active_experts * intermediate_size * 2 * hidden_size * elem_bytes  # w1
            + active_experts * hidden_size * intermediate_size * elem_bytes      # w2
            + num_tokens * topk * 4                             # topk_weights
            + num_tokens * topk * 4                             # topk_ids
        )
        write_bytes = (
            num_tokens * hidden_size * elem_bytes               # output
        )

        return int(read_bytes + write_bytes)
