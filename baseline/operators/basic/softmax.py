"""Softmax 算子"""

import torch
from baseline.operators.registry import BaseOperator, register_operator


@register_operator("softmax")
class SoftmaxOperator(BaseOperator):
    """Softmax: y = exp(x) / sum(exp(x))

    Input: (M, N) -> Output: (M, N)
    """

    @property
    def name(self) -> str:
        return "softmax"

    def forward(self, x: torch.Tensor, dim: int = -1, **kwargs) -> torch.Tensor:
        return torch.nn.functional.softmax(x, dim=dim)

    def compute_flops(self, M: int, N: int, **kwargs) -> int:
        """Softmax FLOPs ≈ 5 * M * N (max, sub, exp, sum, div)"""
        return 5 * M * N

    def compute_bytes(self, M: int, N: int, dtype: str = "bf16", **kwargs) -> int:
        """访存 = 读 x + 写 output"""
        elem_bytes = self.dtype_bytes(dtype)
        return 2 * M * N * elem_bytes

    def prepare_inputs(self, M: int, N: int, dtype: str = "bf16",
                       dim: int = -1, **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(M, N, device=self.device, dtype=torch_dtype)
        return {"x": x, "dim": dim}

    def compute_golden(self, x: torch.Tensor, dim: int = -1, **kwargs) -> torch.Tensor:
        x_fp32 = x.float().cpu()
        result = torch.nn.functional.softmax(x_fp32, dim=dim)
        return result.to(x.dtype).to(x.device)
