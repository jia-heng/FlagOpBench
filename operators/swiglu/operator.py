"""SwiGLU算子"""
from framework.registry import register_operator
from framework.base_operator import BaseOperator
import torch


@register_operator("swiglu")
class SwiGLUOperator(BaseOperator):
    """SwiGLU激活函数

    输入: x [M, N*2]
    输出: silu(x[:, :N]) * x[:, N:] -> [M, N]
    """

    @property
    def name(self) -> str:
        return "swiglu"

    @property
    def library(self) -> str:
        return "flaggems"  # swiglu在flag_gems中

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            M/num_tokens: batch size / seq_len
            N: hidden_dim
            dtype: 数据类型
        """
        M = params.get("M") or params.get("num_tokens")
        N = params["N"]
        dtype = self.get_dtype(params.get("dtype", "bfloat16"))

        input_tensor = torch.randn(M, N * 2, dtype=dtype, device="cuda")
        return {"input_tensor": input_tensor}

    def compute_flops(self, **params):
        """理论FLOPs

        silu(x1) = x1 * sigmoid(x1): ~4 ops per element
        multiply: 1 op per element
        total: ~5 ops per output element
        """
        M = params.get("M") or params.get("num_tokens")
        N = params["N"]
        return M * N * 5

    def compute_bytes(self, **params):
        """理论访存量

        input: M * N * 2 elements
        output: M * N elements
        """
        M = params.get("M") or params.get("num_tokens")
        N = params["N"]
        dtype_bytes = self.dtype_bytes(params.get("dtype", "bfloat16"))
        return (M * N * 2 + M * N) * dtype_bytes
