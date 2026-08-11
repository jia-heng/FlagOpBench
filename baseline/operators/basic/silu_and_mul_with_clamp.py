"""SiLU + Mul + Clamp 融合算子

对应算子列表中的融合激活函数

优先级:
1. PyTorch 官方组合 (F.silu + torch.clamp)
2. vLLM CUDA kernel (torch.ops._C.silu_and_mul)
3. 手动实现
"""

import torch
import torch.nn.functional as F
from baseline.operators.registry import BaseOperator, register_operator

# 尝试导入 vLLM CUDA ops
try:
    import torch.ops._C as vllm_ops
    HAS_VLLM_SILU = hasattr(vllm_ops, 'silu_and_mul')
except (ImportError, AttributeError):
    HAS_VLLM_SILU = False


@register_operator("silu_and_mul_with_clamp")
class SiluAndMulWithClampOperator(BaseOperator):
    """SiLU + Mul + Clamp 融合算子

    Input: (M, 2*N) - 前半部分做 SiLU，后半部分直接相乘，最后 clamp
    Output: (M, N)
    """

    @property
    def name(self) -> str:
        return "silu_and_mul_with_clamp"

    def forward(self, x: torch.Tensor, clamp_min: float = -10.0,
                clamp_max: float = 10.0, **kwargs) -> torch.Tensor:
        """SiLU + Mul + Clamp 前向

        优先级:
        1. PyTorch 官方组合 (最推荐)
        2. vLLM CUDA kernel (可选优化)
        3. 手动实现 (fallback)
        """
        # 方法1: PyTorch 官方组合
        x1, x2 = x.chunk(2, dim=-1)
        out = F.silu(x1) * x2
        out = torch.clamp(out, clamp_min, clamp_max)
        return out

    def forward_vllm(self, x: torch.Tensor, clamp_min: float = -10.0,
                     clamp_max: float = 10.0, **kwargs) -> torch.Tensor:
        """vLLM CUDA kernel 实现（用于性能对比）"""
        if not HAS_VLLM_SILU:
            return self.forward(x, clamp_min, clamp_max, **kwargs)

        out = torch.empty(x.shape[0], x.shape[1] // 2,
                         device=x.device, dtype=x.dtype)
        vllm_ops.silu_and_mul(out, x)
        out = torch.clamp(out, clamp_min, clamp_max)
        return out

    def compute_flops(self, M: int, N: int, **kwargs) -> int:
        """FLOPs = SiLU(M*N) + Mul(M*N) + Clamp(M*N)"""
        return 3 * M * N

    def compute_bytes(self, M: int, N: int, dtype: str = "bf16", **kwargs) -> int:
        """访存 = 读 x(M*2N) + 写 output(M*N)"""
        elem_bytes = self.dtype_bytes(dtype)
        read_x = M * 2 * N * elem_bytes
        write_out = M * N * elem_bytes
        return read_x + write_out

    def prepare_inputs(self, M: int, N: int, dtype: str = "bf16",
                       clamp_min: float = -10.0, clamp_max: float = 10.0,
                       **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(M, 2 * N, device=self.device, dtype=torch_dtype)
        return {"x": x, "clamp_min": clamp_min, "clamp_max": clamp_max}

    def compute_golden(self, x: torch.Tensor, clamp_min: float = -10.0,
                       clamp_max: float = 10.0, **kwargs) -> torch.Tensor:
        x_fp32 = x.float().cpu()
        x1, x2 = x_fp32.chunk(2, dim=-1)
        out = F.silu(x1) * x2
        out = torch.clamp(out, clamp_min, clamp_max)
        return out.to(x.dtype).to(x.device)
