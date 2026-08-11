"""TopK + Softplus + Sqrt 算子 (对应算子列表 #6: topk_softplus_sqrt)

MoE routing 中的 topk 选择 + softplus 激活 + sqrt 归一化。
"""

import torch
import torch.nn.functional as F
from baseline.operators.registry import BaseOperator, register_operator


@register_operator("topk_softplus_sqrt")
class TopKSoftplusSqrtOperator(BaseOperator):
    """TopK + Softplus + Sqrt

    对 router logits 取 top-k，然后 softplus 激活，最后 sqrt 归一化。
    Input: logits (num_tokens, num_experts)
    Output: values (num_tokens, k), indices (num_tokens, k)
    """

    @property
    def name(self) -> str:
        return "topk_softplus_sqrt"

    def forward(self, logits: torch.Tensor, k: int = 8, **kwargs) -> torch.Tensor:
        """TopK -> Softplus -> Sqrt normalization"""
        topk_values, topk_indices = torch.topk(logits, k, dim=-1)
        # Softplus activation
        activated = F.softplus(topk_values)
        # Sqrt normalization
        norm = torch.sqrt(activated.sum(dim=-1, keepdim=True))
        normalized = activated / norm
        return normalized

    def compute_flops(self, num_tokens: int, num_experts: int,
                      k: int = 8, **kwargs) -> int:
        """TopK: O(N*logK), Softplus: ~4*N*K, Sqrt+Div: ~3*N*K"""
        topk_flops = num_tokens * num_experts  # 简化为线性扫描
        activation_flops = 4 * num_tokens * k  # exp + log + add
        norm_flops = 3 * num_tokens * k  # sum + sqrt + div
        return topk_flops + activation_flops + norm_flops

    def compute_bytes(self, num_tokens: int, num_experts: int,
                      k: int = 8, dtype: str = "bf16", **kwargs) -> int:
        elem_bytes = self.dtype_bytes(dtype)
        read_logits = num_tokens * num_experts * elem_bytes
        write_output = num_tokens * k * elem_bytes
        return read_logits + write_output

    def prepare_inputs(self, num_tokens: int, num_experts: int,
                       k: int = 8, dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        logits = torch.randn(
            num_tokens, num_experts, device=self.device, dtype=torch_dtype
        )
        return {"logits": logits, "k": k}

    def compute_golden(self, logits: torch.Tensor, k: int = 8,
                       **kwargs) -> torch.Tensor:
        l_fp32 = logits.float().cpu()
        topk_values, _ = torch.topk(l_fp32, k, dim=-1)
        activated = F.softplus(topk_values)
        norm = torch.sqrt(activated.sum(dim=-1, keepdim=True))
        normalized = activated / norm
        return normalized.to(logits.dtype).to(logits.device)
