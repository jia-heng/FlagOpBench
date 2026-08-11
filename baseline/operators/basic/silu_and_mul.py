"""SiLU and Mul 算子 (对应算子列表 #2: silu_and_mul_with_clamp)"""

import torch
from baseline.operators.registry import BaseOperator, register_operator


@register_operator("silu_and_mul")
class SiLUAndMulOperator(BaseOperator):
    """SiLU and Mul: y = silu(x1) * x2

    常见于 LLM FFN 的 gate projection。
    Input: x of shape (M, 2*N)，前半是 gate，后半是 up。
    Output: (M, N)
    """

    @property
    def name(self) -> str:
        return "silu_and_mul"

    def forward(self, x: torch.Tensor, clamp_value: float = None, **kwargs) -> torch.Tensor:
        """SiLU and Mul 前向

        x: (M, 2*N) - 前半 gate, 后半 up
        """
        half = x.shape[-1] // 2
        gate = x[..., :half]
        up = x[..., half:]
        result = torch.nn.functional.silu(gate) * up
        if clamp_value is not None:
            result = result.clamp(-clamp_value, clamp_value)
        return result

    def compute_flops(self, M: int, N: int, **kwargs) -> int:
        """SiLU(x) * y FLOPs ≈ 5 * M * N (sigmoid: 4 ops, mul: 1 op for silu, + mul: 1 op)"""
        return 6 * M * N

    def compute_bytes(self, M: int, N: int, dtype: str = "bf16", **kwargs) -> int:
        """访存 = 读 x (2*N) + 写 output (N)"""
        elem_bytes = self.dtype_bytes(dtype)
        read_x = M * 2 * N * elem_bytes
        write_out = M * N * elem_bytes
        return read_x + write_out

    def prepare_inputs(self, M: int, N: int, dtype: str = "bf16",
                       clamp_value: float = None, **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(M, 2 * N, device=self.device, dtype=torch_dtype)
        inputs = {"x": x}
        if clamp_value is not None:
            inputs["clamp_value"] = clamp_value
        return inputs

    def compute_golden(self, x: torch.Tensor, clamp_value: float = None,
                       **kwargs) -> torch.Tensor:
        x_fp32 = x.float().cpu()
        half = x_fp32.shape[-1] // 2
        gate = x_fp32[..., :half]
        up = x_fp32[..., half:]
        result = torch.nn.functional.silu(gate) * up
        if clamp_value is not None:
            result = result.clamp(-clamp_value, clamp_value)
        return result.to(x.dtype).to(x.device)
