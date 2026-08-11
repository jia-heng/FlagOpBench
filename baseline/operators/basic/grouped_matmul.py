"""Grouped Matmul 算子 (对应算子列表 #41: GroupedMatmul)

分组矩阵乘法，MoE 中常见。
"""

import torch
from baseline.operators.registry import BaseOperator, register_operator


@register_operator("grouped_matmul")
class GroupedMatmulOperator(BaseOperator):
    """Grouped Matmul

    对多个 expert 的权重矩阵分组执行 matmul。
    Input:
        x: (num_tokens, hidden_size)
        weights: (num_experts, hidden_size, expert_size)
        indices: (num_tokens,) - 每个 token 对应的 expert id
    Output: (num_tokens, expert_size)
    """

    @property
    def name(self) -> str:
        return "grouped_matmul"

    def forward(self, x: torch.Tensor, weights: torch.Tensor,
                expert_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        """按 expert_ids 分组执行 matmul

        使用 index_select + bmm 批量操作，避免循环
        """
        # x: (num_tokens, hidden_size)
        # weights: (num_experts, hidden_size, expert_size)
        # expert_ids: (num_tokens,)

        # 使用 index_select 批量索引对应的 expert weights
        selected_weights = weights[expert_ids]  # (num_tokens, hidden_size, expert_size)

        # 使用 bmm 批量计算
        output = torch.bmm(
            x.unsqueeze(1),  # (num_tokens, 1, hidden_size)
            selected_weights  # (num_tokens, hidden_size, expert_size)
        ).squeeze(1)  # (num_tokens, expert_size)

        return output

    def compute_flops(self, num_tokens: int, hidden_size: int,
                      expert_size: int, **kwargs) -> int:
        """GEMM FLOPs for all tokens"""
        return 2 * num_tokens * hidden_size * expert_size

    def compute_bytes(self, num_tokens: int, hidden_size: int,
                      expert_size: int, num_experts: int = 8,
                      dtype: str = "bf16", **kwargs) -> int:
        elem_bytes = self.dtype_bytes(dtype)
        read_x = num_tokens * hidden_size * elem_bytes
        read_weights = num_experts * hidden_size * expert_size * elem_bytes
        read_indices = num_tokens * 4  # int32
        write_output = num_tokens * expert_size * elem_bytes
        return read_x + read_weights + read_indices + write_output

    def prepare_inputs(self, num_tokens: int, hidden_size: int,
                       expert_size: int, num_experts: int = 8,
                       dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(num_tokens, hidden_size, device=self.device, dtype=torch_dtype)
        weights = torch.randn(num_experts, hidden_size, expert_size,
                             device=self.device, dtype=torch_dtype)
        expert_ids = torch.randint(0, num_experts, (num_tokens,), device=self.device)
        return {"x": x, "weights": weights, "expert_ids": expert_ids}

    def compute_golden(self, x: torch.Tensor, weights: torch.Tensor,
                       expert_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        x_fp32 = x.float().cpu()
        w_fp32 = weights.float().cpu()
        ids_cpu = expert_ids.cpu()

        # 使用批量操作保持一致
        selected_weights = w_fp32[ids_cpu]  # (num_tokens, hidden_size, expert_size)
        output = torch.bmm(
            x_fp32.unsqueeze(1),  # (num_tokens, 1, hidden_size)
            selected_weights  # (num_tokens, hidden_size, expert_size)
        ).squeeze(1)  # (num_tokens, expert_size)

        return output.to(x.dtype).to(x.device)
