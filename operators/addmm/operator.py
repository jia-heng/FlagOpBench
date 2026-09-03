"""Addmm算子

矩阵乘加: output = beta * bias + alpha * (mat1 @ mat2)

输入:
    bias: (N,) 或 (M, N) 张量 (可广播)
    mat1: (M, K) 张量
    mat2: (K, N) 张量
    beta: 标量, 默认 1
    alpha: 标量, 默认 1

输出:
    (M, N) — beta * bias + alpha * (mat1 @ mat2)
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("addmm")
class AddmmOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "addmm"

    @property
    def library(self) -> str:
        return "flag_gems"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            M/num_tokens: 行数 (mat1 的行)
            K/hidden_size: 内部维度
            N: 列数 (mat2 的列, 默认等于 K)
            dtype: 数据类型
        """
        M = params.get("M") or params.get("num_tokens")
        K = params.get("K") or params.get("hidden_size")
        N = params.get("N") or K
        dtype = self.get_dtype(params.get("dtype", "bfloat16"))

        bias = torch.randn(N, dtype=dtype, device="cuda")
        mat1 = torch.randn(M, K, dtype=dtype, device="cuda")
        mat2 = torch.randn(K, N, dtype=dtype, device="cuda")

        return {"bias": bias, "mat1": mat1, "mat2": mat2}

    def compute_flops(self, **params):
        """理论FLOPs

        矩阵乘法 (M,K) x (K,N): 2*M*K*N
        加偏置: M*N (忽略不计)
        """
        M = params.get("M") or params.get("num_tokens")
        K = params.get("K") or params.get("hidden_size")
        N = params.get("N") or K
        return 2 * M * K * N

    def compute_bytes(self, **params):
        """理论访存量

        读: bias(N) + mat1(M*K) + mat2(K*N)
        写: output(M*N)
        """
        M = params.get("M") or params.get("num_tokens")
        K = params.get("K") or params.get("hidden_size")
        N = params.get("N") or K
        elem_bytes = self.dtype_bytes(params.get("dtype", "bfloat16"))
        return int((N + M * K + K * N + M * N) * elem_bytes)
