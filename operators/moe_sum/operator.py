"""MoE Sum算子

将topk个expert的输出沿topk维度求和。

输入:
    input: (num_tokens, topk, hidden_size)
    output: (num_tokens, hidden_size) — 预分配的输出tensor

输出:
    output（原地写入）
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("moe_sum")
class MoESumOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "moe_sum"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            num_tokens: token数
            topk: 选择的expert数
            hidden_size: 隐藏维度
            dtype: 数据类型
        """
        num_tokens = params["num_tokens"]
        topk = params["topk"]
        hidden_size = params["hidden_size"]
        dtype = self.get_dtype(params.get("dtype", "bfloat16"))

        input = torch.randn(num_tokens, topk, hidden_size, dtype=dtype, device="cuda")
        output = torch.empty(num_tokens, hidden_size, dtype=dtype, device="cuda")

        return {"input": input, "output": output}

    def compute_flops(self, **params):
        """理论FLOPs

        每个(token, hidden)位置做topk-1次加法
        """
        num_tokens = params["num_tokens"]
        topk = params["topk"]
        hidden_size = params["hidden_size"]
        return num_tokens * hidden_size * (topk - 1)

    def compute_bytes(self, **params):
        """理论访存量

        读: num_tokens * topk * hidden_size
        写: num_tokens * hidden_size
        """
        num_tokens = params["num_tokens"]
        topk = params["topk"]
        hidden_size = params["hidden_size"]
        elem_bytes = self.dtype_bytes(params.get("dtype", "bfloat16"))

        read_bytes = num_tokens * topk * hidden_size * elem_bytes
        write_bytes = num_tokens * hidden_size * elem_bytes
        return int(read_bytes + write_bytes)
