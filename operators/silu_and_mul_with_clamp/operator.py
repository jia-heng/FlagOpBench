"""SiLU and Mul with Clamp算子

融合的 SiLU激活 + 逐元素乘法 + Clamp操作。

计算:
    out = clamp(silu(x) * y, -limit, limit)

其中 silu(x) = x * sigmoid(x)

输入:
    x: (M, N)     — 门控输入
    y: (M, N)     — 值输入
    limit: float  — clamp的绝对值上限

输出:
    out: (M, N)
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("silu_and_mul_with_clamp")
class SiluAndMulWithClampOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "silu_and_mul_with_clamp"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            M: 第一维大小（tokens数）
            N: 第二维大小（hidden_size）
            limit: clamp范围 [-limit, limit]
            dtype: 数据类型
        """
        M = params["M"]
        N = params["N"]
        limit = params.get("limit", 10.0)
        dtype = self.get_dtype(params.get("dtype", "bfloat16"))

        x = torch.randn(M, N, dtype=dtype, device="cuda")
        y = torch.randn(M, N, dtype=dtype, device="cuda")

        return {
            "x": x,
            "y": y,
            "limit": limit,
        }

    def compute_flops(self, **params):
        """理论FLOPs

        对于每个元素:
          - sigmoid(x): ~4 ops (exp, add, div, neg)
          - x * sigmoid(x): 1 mul
          - silu(x) * y: 1 mul
          - clamp: 2 comparisons
        总: M * N * 8
        """
        M = params["M"]
        N = params["N"]
        return M * N * 8

    def compute_bytes(self, **params):
        """理论访存量

        读:
          x: M * N (dtype)
          y: M * N (dtype)
        写:
          out: M * N (dtype)
        总: 3 * M * N * elem_bytes
        """
        M = params["M"]
        N = params["N"]
        elem_bytes = self.dtype_bytes(params.get("dtype", "bfloat16"))

        return 3 * M * N * elem_bytes
