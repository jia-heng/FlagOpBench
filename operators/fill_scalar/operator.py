"""Fill_scalar算子

用标量值填充张量: output = flag_gems.fill_scalar(input, value)

典型场景: 初始化张量、清零操作。

输入:
    input: Tensor   — 输入张量（作为模板）
    value: float    — 填充值

输出:
    填充后的张量
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("fill_scalar")
class FillScalarOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "fill_scalar"

    @property
    def library(self) -> str:
        return "flag_gems"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            M/num_tokens: 行数
            N/hidden_size: 列数
            value: 填充值 (默认 0.0)
            dtype: 数据类型
        """
        M = params.get("M") or params.get("num_tokens")
        N = params.get("N") or params.get("hidden_size")
        value = params.get("value", 0.0)
        dtype = self.get_dtype(params.get("dtype", "bfloat16"))

        input_tensor = torch.empty(M, N, dtype=dtype, device="cuda")

        return {"input": input_tensor, "value": value}

    def compute_flops(self, **params):
        """理论FLOPs

        fill_scalar 无算术运算
        """
        return 0

    def compute_bytes(self, **params):
        """理论访存量

        写: M * N (输出)
        读: 忽略（input 仅作为 shape 模板，实际可能不读）
        """
        M = params.get("M") or params.get("num_tokens")
        N = params.get("N") or params.get("hidden_size")
        elem_bytes = self.dtype_bytes(params.get("dtype", "bfloat16"))
        return int(M * N * elem_bytes)
