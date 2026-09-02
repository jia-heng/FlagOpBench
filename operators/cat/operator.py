"""Cat算子

张量拼接: output = torch.cat([A, B], dim=dim)

典型场景: 推理中 KV cache 拼接、隐藏层拼接等。
沿 dim=0 拼接两个 (M, N) 张量 → (2M, N)。

输入:
    A: List[Tensor]  — 待拼接的张量列表
    dim: int         — 拼接维度 (默认 0)

输出:
    拼接后的张量
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("cat")
class CatOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "cat"

    @property
    def library(self) -> str:
        return "flag_gems"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            M/num_tokens: 每个张量的行数
            N/hidden_size: 列数
            num_tensors: 拼接的张量数量 (默认 2)
            dim: 拼接维度 (默认 0)
            dtype: 数据类型
        """
        M = params.get("M") or params.get("num_tokens")
        N = params.get("N") or params.get("hidden_size")
        num_tensors = params.get("num_tensors", 2)
        dim = params.get("dim", 0)
        dtype = self.get_dtype(params.get("dtype", "bfloat16"))

        tensors = [torch.randn(M, N, dtype=dtype, device="cuda")
                   for _ in range(num_tensors)]

        return {"A": tensors, "dim": dim}

    def compute_flops(self, **params):
        """理论FLOPs

        cat 是纯数据搬运，无计算
        """
        return 0

    def compute_bytes(self, **params):
        """理论访存量

        读: num_tensors * M * N
        写: num_tensors * M * N (拼接后的输出)
        """
        M = params.get("M") or params.get("num_tokens")
        N = params.get("N") or params.get("hidden_size")
        num_tensors = params.get("num_tensors", 2)
        elem_bytes = self.dtype_bytes(params.get("dtype", "bfloat16"))
        total_elems = num_tensors * M * N
        return int(2 * total_elems * elem_bytes)
