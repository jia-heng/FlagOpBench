"""LayerNorm 算子"""

import torch
from baseline.operators.registry import BaseOperator, register_operator


@register_operator("layernorm")
class LayerNormOperator(BaseOperator):
    """LayerNorm: y = (x - mean) / sqrt(var + eps) * gamma + beta

    Input: (batch, seq_len, hidden_size) or (M, hidden_size)
    """

    @property
    def name(self) -> str:
        return "layernorm"

    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                bias: torch.Tensor = None, eps: float = 1e-5, **kwargs) -> torch.Tensor:
        normalized_shape = [x.shape[-1]]
        return torch.nn.functional.layer_norm(x, normalized_shape, weight, bias, eps)

    def compute_flops(self, M: int, hidden_size: int, **kwargs) -> int:
        """LayerNorm FLOPs ≈ 5 * M * hidden_size (mean, var, norm, scale, shift)"""
        return 5 * M * hidden_size

    def compute_bytes(self, M: int, hidden_size: int, dtype: str = "bf16", **kwargs) -> int:
        """访存 = 读 x + 读 weight + 读 bias + 写 output"""
        elem_bytes = self.dtype_bytes(dtype)
        read_x = M * hidden_size * elem_bytes
        read_weight = hidden_size * elem_bytes
        read_bias = hidden_size * elem_bytes
        write_out = M * hidden_size * elem_bytes
        return read_x + read_weight + read_bias + write_out

    def prepare_inputs(self, M: int, hidden_size: int, dtype: str = "bf16",
                       has_bias: bool = True, eps: float = 1e-5, **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(M, hidden_size, device=self.device, dtype=torch_dtype)
        weight = torch.ones(hidden_size, device=self.device, dtype=torch_dtype)
        bias = torch.zeros(hidden_size, device=self.device, dtype=torch_dtype) if has_bias else None
        return {"x": x, "weight": weight, "bias": bias, "eps": eps}

    def compute_golden(self, x: torch.Tensor, weight: torch.Tensor,
                       bias: torch.Tensor = None, eps: float = 1e-5, **kwargs) -> torch.Tensor:
        x_fp32 = x.float().cpu()
        w_fp32 = weight.float().cpu()
        b_fp32 = bias.float().cpu() if bias is not None else None
        normalized_shape = [x_fp32.shape[-1]]
        result = torch.nn.functional.layer_norm(x_fp32, normalized_shape, w_fp32, b_fp32, eps)
        return result.to(x.dtype).to(x.device)
