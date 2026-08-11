"""SwiGLU 算子 (对应算子列表 #7)"""

import torch
from baseline.operators.registry import BaseOperator, register_operator


@register_operator("swiglu")
class SwiGLUOperator(BaseOperator):
    """SwiGLU: y = silu(x @ W_gate) * (x @ W_up)

    简化版本（不含 matmul，只测 activation 部分）:
    Input: gate (M, N), up (M, N)
    Output: (M, N)
    """

    @property
    def name(self) -> str:
        return "swiglu"

    def forward(self, gate: torch.Tensor, up: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.nn.functional.silu(gate) * up

    def compute_flops(self, M: int, N: int, **kwargs) -> int:
        """SwiGLU FLOPs ≈ 5 * M * N"""
        return 5 * M * N

    def compute_bytes(self, M: int, N: int, dtype: str = "bf16", **kwargs) -> int:
        """访存 = 读 gate + 读 up + 写 output"""
        elem_bytes = self.dtype_bytes(dtype)
        return 3 * M * N * elem_bytes

    def prepare_inputs(self, M: int, N: int, dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        gate = torch.randn(M, N, device=self.device, dtype=torch_dtype)
        up = torch.randn(M, N, device=self.device, dtype=torch_dtype)
        return {"gate": gate, "up": up}

    def compute_golden(self, gate: torch.Tensor, up: torch.Tensor, **kwargs) -> torch.Tensor:
        g_fp32 = gate.float().cpu()
        u_fp32 = up.float().cpu()
        result = torch.nn.functional.silu(g_fp32) * u_fp32
        return result.to(gate.dtype).to(gate.device)
