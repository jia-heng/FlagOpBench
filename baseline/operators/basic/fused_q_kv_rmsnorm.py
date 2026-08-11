"""Fused Q/KV RMSNorm 算子 (对应算子列表 #15: fused_q_kv_rmsnorm)

对 Q 和 KV 分别做 RMSNorm，常见于 MLA 架构。
使用 PyTorch 官方 rms_norm 或 vLLM CUDA kernel
"""

import torch
import torch.nn.functional as F
from baseline.operators.registry import BaseOperator, register_operator

# 尝试导入 vLLM CUDA ops
try:
    import torch.ops._C as vllm_ops
    HAS_VLLM_OPS = hasattr(vllm_ops, 'rms_norm')
except (ImportError, AttributeError):
    HAS_VLLM_OPS = False


@register_operator("fused_q_kv_rmsnorm")
class FusedQKVRmsNormOperator(BaseOperator):
    """Fused Q/KV RMSNorm

    对 query 和 key-value 分别执行 RMSNorm：
    - q_norm = rmsnorm(q, q_weight)
    - kv_norm = rmsnorm(kv, kv_weight)
    """

    @property
    def name(self) -> str:
        return "fused_q_kv_rmsnorm"

    def _rmsnorm(self, x: torch.Tensor, weight: torch.Tensor,
                 eps: float = 1e-6) -> torch.Tensor:
        """RMSNorm 实现 - 优先使用官方 Ops

        优先级:
        1. PyTorch 官方 F.rms_norm (最推荐)
        2. vLLM CUDA kernel (次选)
        3. 手动实现 (fallback)
        """
        if hasattr(F, 'rms_norm'):
            # 优先: PyTorch 官方 rms_norm (PyTorch 2.4+)
            return F.rms_norm(x, [x.shape[-1]], weight, eps)
        elif HAS_VLLM_OPS:
            # 次选: vLLM 官方 CUDA kernel
            out = torch.empty_like(x)
            vllm_ops.rms_norm(out, x, weight, eps)
            return out
        else:
            # Fallback: 手动实现
            input_dtype = x.dtype
            x = x.float()
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)
            return (x * weight.float()).to(input_dtype)

    def forward(self, q: torch.Tensor, kv: torch.Tensor,
                q_weight: torch.Tensor, kv_weight: torch.Tensor,
                eps: float = 1e-6, **kwargs) -> torch.Tensor:
        """分别对 Q 和 KV 做 RMSNorm，拼接返回"""
        q_normed = self._rmsnorm(q, q_weight, eps)
        kv_normed = self._rmsnorm(kv, kv_weight, eps)
        # 返回拼接结果用于校验
        return torch.cat([q_normed, kv_normed], dim=-1)

    def compute_flops(self, num_tokens: int, q_dim: int, kv_dim: int,
                      **kwargs) -> int:
        """两个 RMSNorm: 4 * tokens * dim each"""
        return 4 * num_tokens * (q_dim + kv_dim)

    def compute_bytes(self, num_tokens: int, q_dim: int, kv_dim: int,
                      dtype: str = "bf16", **kwargs) -> int:
        elem_bytes = self.dtype_bytes(dtype)
        # 读 q + kv + q_weight + kv_weight，写 q_out + kv_out
        read_q = num_tokens * q_dim * elem_bytes
        read_kv = num_tokens * kv_dim * elem_bytes
        read_weights = (q_dim + kv_dim) * elem_bytes
        write_output = num_tokens * (q_dim + kv_dim) * elem_bytes
        return read_q + read_kv + read_weights + write_output

    def prepare_inputs(self, num_tokens: int, q_dim: int, kv_dim: int,
                       dtype: str = "bf16", eps: float = 1e-6, **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        q = torch.randn(num_tokens, q_dim, device=self.device, dtype=torch_dtype)
        kv = torch.randn(num_tokens, kv_dim, device=self.device, dtype=torch_dtype)
        q_weight = torch.ones(q_dim, device=self.device, dtype=torch_dtype)
        kv_weight = torch.ones(kv_dim, device=self.device, dtype=torch_dtype)
        return {"q": q, "kv": kv, "q_weight": q_weight,
                "kv_weight": kv_weight, "eps": eps}

    def compute_golden(self, q: torch.Tensor, kv: torch.Tensor,
                       q_weight: torch.Tensor, kv_weight: torch.Tensor,
                       eps: float = 1e-6, **kwargs) -> torch.Tensor:
        q_fp32 = q.float().cpu()
        kv_fp32 = kv.float().cpu()
        qw_fp32 = q_weight.float().cpu()
        kvw_fp32 = kv_weight.float().cpu()

        q_var = q_fp32.pow(2).mean(-1, keepdim=True)
        q_norm = q_fp32 * torch.rsqrt(q_var + eps) * qw_fp32

        kv_var = kv_fp32.pow(2).mean(-1, keepdim=True)
        kv_norm = kv_fp32 * torch.rsqrt(kv_var + eps) * kvw_fp32

        result = torch.cat([q_norm, kv_norm], dim=-1)
        return result.to(q.dtype).to(q.device)
