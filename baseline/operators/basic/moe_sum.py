"""MoE Sum 算子 (对应算子列表 #4: moe_sum)

MoE 中各 expert 输出的加权求和。
"""

import torch
from baseline.operators.registry import BaseOperator, register_operator


@register_operator("moe_sum")
class MoESumOperator(BaseOperator):
    """MoE Sum: 将各 expert 输出按权重加权求和

    Input:
        expert_outputs: (num_experts, num_tokens, hidden_size)
        weights: (num_tokens, num_experts)
    Output: (num_tokens, hidden_size)
    """

    @property
    def name(self) -> str:
        return "moe_sum"

    def forward(self, expert_outputs: torch.Tensor, weights: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """加权求和: output[t] = sum_e(weights[t,e] * expert_outputs[e,t])

        使用 einsum 代替手动 broadcast，更简洁且性能更好
        """
        # expert_outputs: (num_experts, num_tokens, hidden_size)
        # weights: (num_tokens, num_experts)
        # output: (num_tokens, hidden_size)
        return torch.einsum('enh,ne->nh', expert_outputs, weights)

    def compute_flops(self, num_tokens: int, num_experts: int,
                      hidden_size: int, **kwargs) -> int:
        """FLOPs = num_experts * num_tokens * hidden_size * 2 (mul + add)"""
        return 2 * num_experts * num_tokens * hidden_size

    def compute_bytes(self, num_tokens: int, num_experts: int,
                      hidden_size: int, dtype: str = "bf16", **kwargs) -> int:
        elem_bytes = self.dtype_bytes(dtype)
        read_experts = num_experts * num_tokens * hidden_size * elem_bytes
        read_weights = num_tokens * num_experts * elem_bytes
        write_output = num_tokens * hidden_size * elem_bytes
        return read_experts + read_weights + write_output

    def prepare_inputs(self, num_tokens: int, num_experts: int,
                       hidden_size: int, dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        expert_outputs = torch.randn(
            num_experts, num_tokens, hidden_size,
            device=self.device, dtype=torch_dtype
        )
        weights = torch.softmax(
            torch.randn(num_tokens, num_experts, device=self.device, dtype=torch_dtype),
            dim=-1
        )
        return {"expert_outputs": expert_outputs, "weights": weights}

    def compute_golden(self, expert_outputs: torch.Tensor,
                       weights: torch.Tensor, **kwargs) -> torch.Tensor:
        e_fp32 = expert_outputs.float().cpu()
        w_fp32 = weights.float().cpu()
        # 使用 einsum 保持一致
        result = torch.einsum('enh,ne->nh', e_fp32, w_fp32)
        return result.to(expert_outputs.dtype).to(expert_outputs.device)
