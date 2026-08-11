"""Gemma RMS Norm 算子 (对应算子列表 #39: gemma_rms_norm)

Gemma 模型的 RMSNorm 变体: weight + 1
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


@register_operator("gemma_rms_norm")
class GemmaRmsNormOperator(BaseOperator):
    """Gemma RMSNorm: y = x / sqrt(mean(x^2) + eps) * (weight + 1)

    与标准 RMSNorm 的区别：weight 需要 +1
    """

    @property
    def name(self) -> str:
        return "gemma_rms_norm"

    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                eps: float = 1e-6, **kwargs) -> torch.Tensor:
        """Gemma RMSNorm 前向

        优先级:
        1. PyTorch 官方 F.rms_norm (最推荐)
        2. vLLM CUDA kernel (次选)
        3. 手动实现 (fallback)
        """
        # Gemma 特殊: weight + 1
        adjusted_weight = weight + 1.0

        if hasattr(F, 'rms_norm'):
            # 优先: PyTorch 官方 rms_norm (PyTorch 2.4+)
            return F.rms_norm(x, [x.shape[-1]], adjusted_weight, eps)
        elif HAS_VLLM_OPS:
            # 次选: vLLM 官方 CUDA kernel
            out = torch.empty_like(x)
            vllm_ops.rms_norm(out, x, adjusted_weight, eps)
            return out
        else:
            # Fallback: 手动实现
            input_dtype = x.dtype
            x = x.float()
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)
            return (x * adjusted_weight.float()).to(input_dtype)

    def compute_flops(self, M: int = None, num_tokens: int = None,
                      hidden_size: int = None, **kwargs) -> int:
        """RMSNorm FLOPs ≈ 4 * M * hidden_size"""
        batch = num_tokens if num_tokens is not None else M
        return 4 * batch * hidden_size

    def compute_bytes(self, M: int = None, num_tokens: int = None,
                      hidden_size: int = None, dtype: str = "bf16", **kwargs) -> int:
        batch = num_tokens if num_tokens is not None else M
        elem_bytes = self.dtype_bytes(dtype)
        read_x = batch * hidden_size * elem_bytes
        read_weight = hidden_size * elem_bytes
        write_output = batch * hidden_size * elem_bytes
        return read_x + read_weight + write_output

    def prepare_inputs(self, M: int = None, num_tokens: int = None,
                       hidden_size: int = None, dtype: str = "bf16",
                       eps: float = 1e-6, **kwargs) -> dict:
        # Support both M and num_tokens parameter naming
        batch = num_tokens if num_tokens is not None else M
        if batch is None:
            raise ValueError("Must specify either 'M' or 'num_tokens'")

        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(batch, hidden_size, device=self.device, dtype=torch_dtype)
        weight = torch.ones(hidden_size, device=self.device, dtype=torch_dtype)
        return {"x": x, "weight": weight, "eps": eps}

    def compute_golden(self, x: torch.Tensor, weight: torch.Tensor,
                       eps: float = 1e-6, **kwargs) -> torch.Tensor:
        x_fp32 = x.float().cpu()
        w_fp32 = weight.float().cpu()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + eps)
        result = x_norm * (w_fp32 + 1.0)
        return result.to(x.dtype).to(x.device)
