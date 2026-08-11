"""FP8 Einsum 算子 (对应算子列表 #30: fp8_einsum/w8a8_block_fp8_bmm)

使用 einsum 实现通用张量收缩，测试 bf16 baseline。
"""

import torch
from baseline.operators.registry import BaseOperator, register_operator


@register_operator("fp8_einsum")
class Fp8EinsumOperator(BaseOperator):
    """FP8 Einsum (bf16 baseline)

    使用 einsum 实现通用张量收缩，当前测试 bf16 性能。
    Future: 支持 fp8 量化版本。
    """

    @property
    def name(self) -> str:
        return "fp8_einsum"

    def forward(self, a: torch.Tensor, b: torch.Tensor,
                equation: str = "bik,bkj->bij", **kwargs) -> torch.Tensor:
        """Einsum operation with specified equation"""
        return torch.einsum(equation, a, b)

    def compute_flops(self, batch: int, M: int, K: int, N: int,
                      **kwargs) -> int:
        """FLOPs for bmm-like einsum"""
        return 2 * batch * M * N * K

    def compute_bytes(self, batch: int, M: int, K: int, N: int,
                      dtype: str = "bf16", **kwargs) -> int:
        elem_bytes = self.dtype_bytes(dtype)
        read_a = batch * M * K * elem_bytes
        read_b = batch * K * N * elem_bytes
        write_output = batch * M * N * elem_bytes
        return read_a + read_b + write_output

    def prepare_inputs(self, batch: int, M: int, K: int, N: int,
                       dtype: str = "bf16",
                       equation: str = "bik,bkj->bij", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        # 根据 equation 确定 shape (简化为 bmm)
        a = torch.randn(batch, M, K, device=self.device, dtype=torch_dtype)
        b = torch.randn(batch, K, N, device=self.device, dtype=torch_dtype)
        return {"a": a, "b": b, "equation": equation}

    def compute_golden(self, a: torch.Tensor, b: torch.Tensor,
                       equation: str = "bik,bkj->bij", **kwargs) -> torch.Tensor:
        a_fp32 = a.float().cpu()
        b_fp32 = b.float().cpu()
        result = torch.einsum(equation, a_fp32, b_fp32)
        return result.to(a.dtype).to(a.device)
