"""MM算子

矩阵乘法: output = A @ B

输入:
    a: (M, K) 张量
    b: (K, N) 张量

输出:
    (M, N) — a @ b
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("mm")
class MmOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "mm"

    @property
    def library(self) -> str:
        return "flag_gems"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            M/num_tokens: 行数 (第一个矩阵的行)
            K/hidden_size: 内部维度
            N: 列数 (第二个矩阵的列, 默认等于 K)
            dtype: 数据类型
        """
        M = params.get("M") or params.get("num_tokens")
        K = params.get("K") or params.get("hidden_size")
        N = params.get("N") or K
        dtype = self.get_dtype(params.get("dtype", "bfloat16"))

        a = torch.randn(M, K, dtype=dtype, device="cuda")
        b = torch.randn(K, N, dtype=dtype, device="cuda")

        return {"a": a, "b": b}

    def compute_flops(self, **params):
        """理论FLOPs

        矩阵乘法 (M,K) x (K,N): 2*M*K*N (每个输出元素 K 次乘加)
        """
        M = params.get("M") or params.get("num_tokens")
        K = params.get("K") or params.get("hidden_size")
        N = params.get("N") or K
        return 2 * M * K * N

    def compute_bytes(self, **params):
        """理论访存量

        读: a(M*K) + b(K*N)
        写: output(M*N)
        """
        M = params.get("M") or params.get("num_tokens")
        K = params.get("K") or params.get("hidden_size")
        N = params.get("N") or K
        elem_bytes = self.dtype_bytes(params.get("dtype", "bfloat16"))
        return int((M * K + K * N + M * N) * elem_bytes)
