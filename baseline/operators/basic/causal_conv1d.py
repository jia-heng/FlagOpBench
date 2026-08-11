"""Causal Conv1D 算子 (对应算子列表 #22/#23)

因果卷积，用于 Mamba 等状态空间模型：
- CausalConv1DPrefill (#22): prefill 场景
- CausualConv1DDecode (#23): decode 场景（逐 token）
"""

import torch
import torch.nn.functional as F
from baseline.operators.registry import BaseOperator, register_operator


@register_operator("causal_conv1d_prefill")
class CausalConv1DPrefillOperator(BaseOperator):
    """Causal Conv1D - Prefill 场景

    在 prefill 时对整个序列做因果卷积。
    Input: (batch, hidden_size, seq_len)
    Output: (batch, hidden_size, seq_len)
    """

    @property
    def name(self) -> str:
        return "causal_conv1d_prefill"

    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                kernel_size: int = 4, **kwargs) -> torch.Tensor:
        """Causal conv1d with left padding"""
        # x: (batch, hidden_size, seq_len)
        # weight: (hidden_size, 1, kernel_size)
        padding = kernel_size - 1
        x_padded = F.pad(x, (padding, 0))
        return F.conv1d(x_padded, weight, groups=x.shape[1])

    def compute_flops(self, batch: int, hidden_size: int, seq_len: int,
                      kernel_size: int = 4, **kwargs) -> int:
        """Conv1d FLOPs = batch * hidden_size * seq_len * kernel_size * 2"""
        return batch * hidden_size * seq_len * kernel_size * 2

    def compute_bytes(self, batch: int, hidden_size: int, seq_len: int,
                      kernel_size: int = 4, dtype: str = "bf16", **kwargs) -> int:
        elem_bytes = self.dtype_bytes(dtype)
        read_x = batch * hidden_size * seq_len * elem_bytes
        read_weight = hidden_size * kernel_size * elem_bytes
        write_output = batch * hidden_size * seq_len * elem_bytes
        return read_x + read_weight + write_output

    def prepare_inputs(self, batch: int, hidden_size: int, seq_len: int,
                       kernel_size: int = 4, dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(batch, hidden_size, seq_len, device=self.device, dtype=torch_dtype)
        weight = torch.randn(hidden_size, 1, kernel_size, device=self.device, dtype=torch_dtype)
        return {"x": x, "weight": weight, "kernel_size": kernel_size}

    def compute_golden(self, x: torch.Tensor, weight: torch.Tensor,
                       kernel_size: int = 4, **kwargs) -> torch.Tensor:
        x_fp32 = x.float().cpu()
        w_fp32 = weight.float().cpu()
        padding = kernel_size - 1
        x_padded = F.pad(x_fp32, (padding, 0))
        result = F.conv1d(x_padded, w_fp32, groups=x_fp32.shape[1])
        return result.to(x.dtype).to(x.device)


@register_operator("causal_conv1d_decode")
class CausalConv1DDecodeOperator(BaseOperator):
    """Causal Conv1D - Decode 场景

    decode 时逐 token 步进，使用 sliding window。
    Input: (batch, hidden_size, 1)
    Output: (batch, hidden_size, 1)
    """

    @property
    def name(self) -> str:
        return "causal_conv1d_decode"

    def forward(self, x: torch.Tensor, history: torch.Tensor,
                weight: torch.Tensor, kernel_size: int = 4, **kwargs) -> torch.Tensor:
        """Single-token conv1d with history context"""
        # x: (batch, hidden_size, 1)
        # history: (batch, hidden_size, kernel_size-1)
        # weight: (hidden_size, 1, kernel_size)
        x_with_history = torch.cat([history, x], dim=-1)  # (batch, hidden_size, kernel_size)
        return F.conv1d(x_with_history, weight, groups=x.shape[1])

    def compute_flops(self, batch: int, hidden_size: int,
                      kernel_size: int = 4, **kwargs) -> int:
        """Single-token conv1d FLOPs"""
        return batch * hidden_size * kernel_size * 2

    def compute_bytes(self, batch: int, hidden_size: int,
                      kernel_size: int = 4, dtype: str = "bf16", **kwargs) -> int:
        elem_bytes = self.dtype_bytes(dtype)
        read_x = batch * hidden_size * 1 * elem_bytes
        read_history = batch * hidden_size * (kernel_size - 1) * elem_bytes
        read_weight = hidden_size * kernel_size * elem_bytes
        write_output = batch * hidden_size * 1 * elem_bytes
        return read_x + read_history + read_weight + write_output

    def prepare_inputs(self, batch: int, hidden_size: int,
                       kernel_size: int = 4, dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(batch, hidden_size, 1, device=self.device, dtype=torch_dtype)
        history = torch.randn(batch, hidden_size, kernel_size - 1,
                             device=self.device, dtype=torch_dtype)
        weight = torch.randn(hidden_size, 1, kernel_size,
                            device=self.device, dtype=torch_dtype)
        return {"x": x, "history": history, "weight": weight, "kernel_size": kernel_size}

    def compute_golden(self, x: torch.Tensor, history: torch.Tensor,
                       weight: torch.Tensor, kernel_size: int = 4, **kwargs) -> torch.Tensor:
        x_fp32 = x.float().cpu()
        h_fp32 = history.float().cpu()
        w_fp32 = weight.float().cpu()
        x_with_history = torch.cat([h_fp32, x_fp32], dim=-1)
        result = F.conv1d(x_with_history, w_fp32, groups=x_fp32.shape[1])
        return result.to(x.dtype).to(x.device)
