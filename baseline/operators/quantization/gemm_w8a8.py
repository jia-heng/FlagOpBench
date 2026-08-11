"""W8A8 GEMM 算子 (FP8 混合精度矩阵乘法)

可用实现:
1. PyTorch 官方 torch.mm (bf16 GEMM)
2. PyTorch torch._scaled_mm (FP8 scaled GEMM, PyTorch 2.1+)
3. CUTLASS FP8 kernel (未集成)

性能对比 (待实测后更新):
- torch.mm (bf16): cuBLAS GEMM，成熟稳定，throughput 高
- torch._scaled_mm (FP8): 理论吞吐量翻倍 (FP8 tensor core)，但有精度损失
- CUTLASS FP8: 定制 kernel，可能进一步优化

基线选择策略:
  bf16 torch.mm 作为当前基线（稳定且通用）。
  FP8 _scaled_mm 在 H100+ 硬件上理论性能翻倍。
  实测后如果 FP8 性能提升 > 20% 且精度可接受，则切换为 FP8 基线。
"""

import torch
from baseline.operators.registry import BaseOperator, register_operator


@register_operator("gemm_w8a8")
class GemmW8A8Operator(BaseOperator):
    """W8A8 混合精度 GEMM

    Input:
        x: (M, K) - activation (W8/FP8)
        weight: (K, N) - weight (A8/FP8)
        x_scale: scalar or (M,) - activation scale
        w_scale: scalar or (N,) - weight scale
    Output:
        (M, N)
    """

    @property
    def name(self) -> str:
        return "gemm_w8a8"

    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                x_scale: torch.Tensor = None, w_scale: torch.Tensor = None,
                use_fp8: bool = False, **kwargs) -> torch.Tensor:
        """W8A8 GEMM 前向

        可用实现:
        1. torch.mm (bf16) - cuBLAS GEMM，稳定通用 ⭐ 当前基线
        2. torch._scaled_mm (FP8) - FP8 tensor core，吞吐量翻倍
        3. CUTLASS FP8 kernel (未集成)

        基线选择: torch.mm (bf16)
        TODO: 实测 bf16 vs FP8 性能对比，H100 上 FP8 理论提升 ~2x
              如果实测提升 > 20% 且精度可接受，切换为 FP8 基线
        """
        if use_fp8 and hasattr(torch, '_scaled_mm') and x_scale is not None:
            # FP8 实现: 需要 PyTorch 2.1+ 且输入已量化为 FP8
            return torch._scaled_mm(x, weight, scale_a=x_scale, scale_b=w_scale)
        else:
            # 当前基线: bf16 GEMM (cuBLAS)
            return torch.mm(x, weight)

    def compute_flops(self, M: int, N: int, K: int, **kwargs) -> int:
        """GEMM FLOPs = 2 * M * N * K"""
        return 2 * M * N * K

    def compute_bytes(self, M: int, N: int, K: int,
                      dtype: str = "bf16", **kwargs) -> int:
        """访存 = 读 x + 读 weight + 写 output"""
        elem_bytes = self.dtype_bytes(dtype)
        read_x = M * K * elem_bytes
        read_w = K * N * elem_bytes
        write_out = M * N * elem_bytes
        return read_x + read_w + write_out

    def prepare_inputs(self, M: int, N: int, K: int,
                       dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(M, K, device=self.device, dtype=torch_dtype)
        weight = torch.randn(K, N, device=self.device, dtype=torch_dtype)

        # Placeholder scales (FP8 版本使用)
        x_scale = torch.tensor(1.0, device=self.device, dtype=torch.float32)
        w_scale = torch.tensor(1.0, device=self.device, dtype=torch.float32)

        return {
            "x": x,
            "weight": weight,
            "x_scale": x_scale,
            "w_scale": w_scale
        }

    def compute_golden(self, x: torch.Tensor, weight: torch.Tensor,
                       x_scale: torch.Tensor = None, w_scale: torch.Tensor = None,
                       **kwargs) -> torch.Tensor:
        x_fp32 = x.float().cpu()
        w_fp32 = weight.float().cpu()
        result = torch.mm(x_fp32, w_fp32)
        return result.to(x.dtype).to(x.device)
