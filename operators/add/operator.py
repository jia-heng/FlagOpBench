"""Add算子

逐元素加法: output = A + alpha * B

输入:
    A: (M, N) 张量
    B: (M, N) 张量
    alpha: 缩放因子 (默认 1)

输出:
    (M, N) — A + alpha * B
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("add")
class AddOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "add"

    @property
    def library(self) -> str:
        return "flag_gems"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            M/num_tokens: 行数
            N/hidden_size: 列数
            alpha: 缩放因子 (默认 1)
            dtype: 数据类型
        """
        M = params.get("M") or params.get("num_tokens")
        N = params.get("N") or params.get("hidden_size")
        alpha = params.get("alpha", 1)
        dtype = self.get_dtype(params.get("dtype", "bfloat16"))

        A = torch.randn(M, N, dtype=dtype, device="cuda")
        B = torch.randn(M, N, dtype=dtype, device="cuda")

        return {"A": A, "B": B, "alpha": alpha}

    def compute_flops(self, **params):
        """理论FLOPs

        每个元素: alpha * B (1 mul) + A + ... (1 add) = 2 ops
        alpha=1时只有1 add
        """
        M = params.get("M") or params.get("num_tokens")
        N = params.get("N") or params.get("hidden_size")
        alpha = params.get("alpha", 1)
        ops_per_elem = 1 if alpha == 1 else 2
        return M * N * ops_per_elem

    def compute_bytes(self, **params):
        """理论访存量

        读: A(M*N) + B(M*N)
        写: output(M*N)
        """
        M = params.get("M") or params.get("num_tokens")
        N = params.get("N") or params.get("hidden_size")
        elem_bytes = self.dtype_bytes(params.get("dtype", "bfloat16"))
        return int(3 * M * N * elem_bytes)
