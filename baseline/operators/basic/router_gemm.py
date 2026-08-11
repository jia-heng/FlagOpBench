"""Router GEMM 算子 (对应算子列表 #52: router_gemm_bf16_fp32)

MoE router 的 GEMM，bf16 输入，fp32 累加。
"""

import torch
from baseline.operators.registry import BaseOperator, register_operator


@register_operator("router_gemm_bf16_fp32")
class RouterGemmBf16Fp32Operator(BaseOperator):
    """Router GEMM: bf16 input, fp32 accumulate

    常见于 MoE router 计算 logits:
    logits = x @ router_weight (bf16 -> fp32 accumulate)
    """

    @property
    def name(self) -> str:
        return "router_gemm_bf16_fp32"

    def forward(self, x: torch.Tensor, router_weight: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """bf16 GEMM with fp32 accumulate"""
        # x: (num_tokens, hidden_size), bf16
        # router_weight: (num_experts, hidden_size), bf16
        # output: (num_tokens, num_experts), fp32
        x_fp32 = x.float()
        w_fp32 = router_weight.t().float()
        logits = torch.mm(x_fp32, w_fp32)
        return logits

    def compute_flops(self, num_tokens: int, hidden_size: int,
                      num_experts: int, **kwargs) -> int:
        """GEMM FLOPs = 2 * M * N * K"""
        return 2 * num_tokens * num_experts * hidden_size

    def compute_bytes(self, num_tokens: int, hidden_size: int,
                      num_experts: int, dtype: str = "bf16", **kwargs) -> int:
        elem_bytes = self.dtype_bytes(dtype)
        read_x = num_tokens * hidden_size * elem_bytes
        read_weight = num_experts * hidden_size * elem_bytes
        write_logits = num_tokens * num_experts * 4  # fp32 output
        return read_x + read_weight + write_logits

    def prepare_inputs(self, num_tokens: int, hidden_size: int,
                       num_experts: int, dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(num_tokens, hidden_size, device=self.device, dtype=torch_dtype)
        router_weight = torch.randn(num_experts, hidden_size,
                                    device=self.device, dtype=torch_dtype)
        return {"x": x, "router_weight": router_weight}

    def compute_golden(self, x: torch.Tensor, router_weight: torch.Tensor,
                       **kwargs) -> torch.Tensor:
        x_fp32 = x.float().cpu()
        w_fp32 = router_weight.t().float().cpu()
        logits = torch.mm(x_fp32, w_fp32)
        return logits.to(x.device)
