"""矩阵乘法算子 (mm)"""

import torch
from baseline.operators.registry import BaseOperator, register_operator


@register_operator("mm")
class MMOperator(BaseOperator):
    """矩阵乘法: C = A @ B

    A: (M, K), B: (K, N) -> C: (M, N)
    """

    @property
    def name(self) -> str:
        return "mm"

    def forward(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.mm(A, B)

    def compute_flops(self, M: int, K: int, N: int, **kwargs) -> int:
        """GEMM FLOPs = 2 * M * N * K (乘加各算一次)"""
        return 2 * M * N * K

    def compute_bytes(self, M: int, K: int, N: int, dtype: str = "bf16", **kwargs) -> int:
        """访存量 = 读 A + 读 B + 写 C"""
        elem_bytes = self.dtype_bytes(dtype)
        read_a = M * K * elem_bytes
        read_b = K * N * elem_bytes
        write_c = M * N * elem_bytes
        return read_a + read_b + write_c

    def prepare_inputs(self, M: int, K: int, N: int, dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        A = torch.randn(M, K, device=self.device, dtype=torch_dtype)
        B = torch.randn(K, N, device=self.device, dtype=torch_dtype)
        return {"A": A, "B": B}

    def compute_golden(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        """CPU fp32 golden reference"""
        a_fp32 = A.float().cpu()
        b_fp32 = B.float().cpu()
        result = torch.mm(a_fp32, b_fp32)
        return result.to(A.dtype).to(A.device)
