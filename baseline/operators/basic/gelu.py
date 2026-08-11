"""GeLU 激活函数算子"""

import torch
from baseline.operators.registry import BaseOperator, register_operator


@register_operator("gelu")
class GeLUOperator(BaseOperator):
    """GeLU 激活函数

    Input: (M, N) -> Output: (M, N)
    """

    @property
    def name(self) -> str:
        return "gelu"

    def forward(self, x: torch.Tensor, approximate: str = "none", **kwargs) -> torch.Tensor:
        return torch.nn.functional.gelu(x, approximate=approximate)

    def compute_flops(self, M: int, N: int, **kwargs) -> int:
        """GeLU FLOPs ≈ 8 * M * N (使用 tanh 近似时)"""
        return 8 * M * N

    def compute_bytes(self, M: int, N: int, dtype: str = "bf16", **kwargs) -> int:
        """访存 = 读 x + 写 output"""
        elem_bytes = self.dtype_bytes(dtype)
        return 2 * M * N * elem_bytes

    def prepare_inputs(self, M: int, N: int, dtype: str = "bf16",
                       approximate: str = "none", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(M, N, device=self.device, dtype=torch_dtype)
        return {"x": x, "approximate": approximate}

    def compute_golden(self, x: torch.Tensor, approximate: str = "none",
                       **kwargs) -> torch.Tensor:
        x_fp32 = x.float().cpu()
        result = torch.nn.functional.gelu(x_fp32, approximate=approximate)
        return result.to(x.dtype).to(x.device)
