"""Add + RMSNorm + Bias 算子

对应算子列表 #21: AddRmsNormBias

优先级:
1. PyTorch 官方 F.rms_norm (PyTorch 2.4+)
2. vLLM CUDA kernel (torch.ops._C.fused_add_rms_norm)
3. 手动实现 (fallback)
"""

import torch
import torch.nn.functional as F
from baseline.operators.registry import BaseOperator, register_operator

# 尝试导入 vLLM CUDA ops
try:
    import torch.ops._C as vllm_ops
    HAS_VLLM_OPS = hasattr(vllm_ops, 'fused_add_rms_norm')
except (ImportError, AttributeError):
    HAS_VLLM_OPS = False


@register_operator("add_rmsnorm_bias")
class AddRmsNormBiasOperator(BaseOperator):
    """AddRmsNormBias: y = rmsnorm(x + residual) * weight + bias

    融合操作：
    1. residual add: h = x + residual
    2. RMSNorm: h_norm = h / sqrt(mean(h^2) + eps) * weight
    3. bias add: output = h_norm + bias
    """

    @property
    def name(self) -> str:
        return "add_rmsnorm_bias"

    def forward(self, x: torch.Tensor, residual: torch.Tensor,
                weight: torch.Tensor, bias: torch.Tensor = None,
                eps: float = 1e-6, **kwargs) -> torch.Tensor:
        """Add + RMSNorm + Bias 前向

        优先级:
        1. PyTorch 官方 F.rms_norm (最推荐，性能好且具有参考价值)
        2. vLLM CUDA kernel (次选，不支持 bias)
        3. 手动实现 (fallback)
        """
        # Step 1: Add
        h = x + residual

        # Step 2: RMSNorm
        if hasattr(F, 'rms_norm'):
            # 优先: PyTorch 官方 rms_norm (PyTorch 2.4+)
            h_norm = F.rms_norm(h, [h.shape[-1]], weight, eps)
        elif HAS_VLLM_OPS:
            # 次选: vLLM CUDA kernel (需要 in-place 操作)
            h_copy = h.clone()
            out = torch.empty_like(h)
            # vLLM rms_norm: (out, input, weight, eps)
            vllm_ops.rms_norm(out, h_copy, weight, eps)
            h_norm = out
        else:
            # Fallback: 手动实现
            input_dtype = h.dtype
            h_fp32 = h.float()
            variance = h_fp32.pow(2).mean(-1, keepdim=True)
            h_norm = (h_fp32 * torch.rsqrt(variance + eps) * weight.float()).to(input_dtype)

        # Step 3: Bias add (if exists)
        if bias is not None:
            h_norm = h_norm + bias

        return h_norm

    def compute_flops(self, M: int, hidden_size: int, **kwargs) -> int:
        """Add: M*H, RMSNorm: 4*M*H, Bias: M*H"""
        add_flops = M * hidden_size
        norm_flops = 4 * M * hidden_size
        bias_flops = M * hidden_size
        return add_flops + norm_flops + bias_flops

    def compute_bytes(self, M: int, hidden_size: int,
                      dtype: str = "bf16", **kwargs) -> int:
        elem_bytes = self.dtype_bytes(dtype)
        read_x = M * hidden_size * elem_bytes
        read_residual = M * hidden_size * elem_bytes
        read_weight = hidden_size * elem_bytes
        read_bias = hidden_size * elem_bytes
        write_output = M * hidden_size * elem_bytes
        return read_x + read_residual + read_weight + read_bias + write_output

    def prepare_inputs(self, M: int, hidden_size: int, dtype: str = "bf16",
                       has_bias: bool = True, eps: float = 1e-6, **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(M, hidden_size, device=self.device, dtype=torch_dtype)
        residual = torch.randn(M, hidden_size, device=self.device, dtype=torch_dtype)
        weight = torch.ones(hidden_size, device=self.device, dtype=torch_dtype)
        bias = torch.zeros(hidden_size, device=self.device, dtype=torch_dtype) if has_bias else None
        return {"x": x, "residual": residual, "weight": weight,
                "bias": bias, "eps": eps}

    def compute_golden(self, x: torch.Tensor, residual: torch.Tensor,
                       weight: torch.Tensor, bias: torch.Tensor = None,
                       eps: float = 1e-6, **kwargs) -> torch.Tensor:
        h = (x + residual).float().cpu()
        w = weight.float().cpu()
        variance = h.pow(2).mean(-1, keepdim=True)
        h_norm = h * torch.rsqrt(variance + eps)
        output = h_norm * w
        if bias is not None:
            output = output + bias.float().cpu()
        return output.to(x.dtype).to(x.device)
