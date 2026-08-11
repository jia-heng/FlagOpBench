"""批矩阵乘法算子 (bmm)"""

import torch
from baseline.operators.registry import BaseOperator, register_operator


@register_operator("bmm")
class BMMOperator(BaseOperator):
    """批矩阵乘法: C = A @ B (batched)

    A: (B, M, K), B_mat: (B, K, N) -> C: (B, M, N)
    """

    @property
    def name(self) -> str:
        return "bmm"

    def forward(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        return torch.bmm(A, B)

    def compute_flops(self, batch: int, M: int, K: int, N: int, **kwargs) -> int:
        """Batched GEMM FLOPs = batch * 2 * M * N * K"""
        return batch * 2 * M * N * K

    def compute_bytes(self, batch: int, M: int, K: int, N: int,
                      dtype: str = "bf16", **kwargs) -> int:
        elem_bytes = self.dtype_bytes(dtype)
        read_a = batch * M * K * elem_bytes
        read_b = batch * K * N * elem_bytes
        write_c = batch * M * N * elem_bytes
        return read_a + read_b + write_c

    def prepare_inputs(self, batch: int, M: int, K: int, N: int,
                       dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        A = torch.randn(batch, M, K, device=self.device, dtype=torch_dtype)
        B = torch.randn(batch, K, N, device=self.device, dtype=torch_dtype)
        return {"A": A, "B": B}

    def compute_golden(self, A: torch.Tensor, B: torch.Tensor, **kwargs) -> torch.Tensor:
        a_fp32 = A.float().cpu()
        b_fp32 = B.float().cpu()
        result = torch.bmm(a_fp32, b_fp32)
        return result.to(A.dtype).to(A.device)
