"""Per-token Group FP8 Quantization 算子

动态量化: 将输入按 group 量化为 FP8

可用实现:
1. PyTorch 手动实现 (absmax + scale + round)
2. vLLM dynamic quantization kernel (未集成)

性能对比 (待实测后更新):
- PyTorch 手动: 多步 ops (reshape, abs, max, div, round, clamp)，有中间 tensor 开销
- vLLM kernel: 单 kernel fused 完成所有步骤，无中间 tensor

基线选择策略:
  当前仅 PyTorch 手动实现可用，作为基线。
  vLLM kernel 集成后需进行性能对比，预期 fused kernel 性能提升 30%+。
  一旦 vLLM kernel 可用且经过验证，应切换为 vLLM 实现。
"""

import torch
from baseline.operators.registry import BaseOperator, register_operator


@register_operator("per_token_group_fp8_quant")
class PerTokenGroupFP8QuantOperator(BaseOperator):
    """Per-token Group FP8 动态量化

    Input:
        x: (num_tokens, hidden_size)
        group_size: int - 量化分组大小
    Output:
        x_quant: (num_tokens, hidden_size) - int8
        scales: (num_tokens, num_groups) - float32
    """

    @property
    def name(self) -> str:
        return "per_token_group_fp8_quant"

    def forward(self, x: torch.Tensor, group_size: int = 128,
                **kwargs) -> tuple:
        """Per-token Group FP8 量化前向

        可用实现:
        1. PyTorch 手动实现 (absmax + scale + round) ⭐ 当前基线
        2. vLLM dynamic quantization kernel (未集成)

        基线选择: PyTorch 手动实现（唯一可用实现）
        TODO: 集成 vLLM kernel 后进行性能对比，预期 fused 版本提升 30%+
        """
        num_tokens, hidden_size = x.shape
        num_groups = (hidden_size + group_size - 1) // group_size

        # Reshape 为 groups
        # Padding 到 group_size 的倍数
        padded_size = num_groups * group_size
        if padded_size > hidden_size:
            padding = torch.zeros(num_tokens, padded_size - hidden_size,
                                device=x.device, dtype=x.dtype)
            x_padded = torch.cat([x, padding], dim=-1)
        else:
            x_padded = x

        # Reshape: (num_tokens, num_groups, group_size)
        x_grouped = x_padded.reshape(num_tokens, num_groups, group_size)

        # 计算每组的 scale (使用 absmax)
        scales = x_grouped.abs().max(dim=-1, keepdim=True)[0] / 127.0  # (num_tokens, num_groups, 1)
        scales = scales.squeeze(-1)  # (num_tokens, num_groups)

        # 量化
        x_quant = (x_grouped / (scales.unsqueeze(-1) + 1e-8)).round().clamp(-127, 127).to(torch.int8)

        # Reshape back
        x_quant = x_quant.reshape(num_tokens, padded_size)
        if padded_size > hidden_size:
            x_quant = x_quant[:, :hidden_size]

        return x_quant, scales

    def compute_flops(self, num_tokens: int, hidden_size: int,
                      group_size: int = 128, **kwargs) -> int:
        """FLOPs ≈ num_tokens * hidden_size * 3 (absmax, div, round)"""
        return num_tokens * hidden_size * 3

    def compute_bytes(self, num_tokens: int, hidden_size: int,
                      group_size: int = 128, dtype: str = "bf16",
                      **kwargs) -> int:
        """访存 = 读 x + 写 x_quant + 写 scales"""
        elem_bytes = self.dtype_bytes(dtype)
        num_groups = (hidden_size + group_size - 1) // group_size

        read_x = num_tokens * hidden_size * elem_bytes
        write_quant = num_tokens * hidden_size * 1  # int8
        write_scales = num_tokens * num_groups * 4  # float32
        return read_x + write_quant + write_scales

    def prepare_inputs(self, num_tokens: int, hidden_size: int,
                       group_size: int = 128, dtype: str = "bf16",
                       **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(num_tokens, hidden_size, device=self.device, dtype=torch_dtype)
        return {"x": x, "group_size": group_size}

    def compute_golden(self, x: torch.Tensor, group_size: int = 128,
                       **kwargs) -> tuple:
        """Golden reference"""
        x_fp32 = x.float().cpu()
        num_tokens, hidden_size = x_fp32.shape
        num_groups = (hidden_size + group_size - 1) // group_size

        # Padding
        padded_size = num_groups * group_size
        if padded_size > hidden_size:
            padding = torch.zeros(num_tokens, padded_size - hidden_size)
            x_padded = torch.cat([x_fp32, padding], dim=-1)
        else:
            x_padded = x_fp32

        # Reshape
        x_grouped = x_padded.reshape(num_tokens, num_groups, group_size)

        # Scales
        scales = x_grouped.abs().max(dim=-1, keepdim=True)[0] / 127.0
        scales = scales.squeeze(-1)

        # Quantize
        x_quant = (x_grouped / (scales.unsqueeze(-1) + 1e-8)).round().clamp(-127, 127).to(torch.int8)
        x_quant = x_quant.reshape(num_tokens, padded_size)
        if padded_size > hidden_size:
            x_quant = x_quant[:, :hidden_size]

        return x_quant.to(x.device), scales.to(x.device)
