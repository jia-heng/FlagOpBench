"""Fused Marlin MoE 算子 (INT4/INT8 量化 MoE GEMM)

对应算子列表: fused_marlin_moe

可用实现:
1. PyTorch dequantize + mm (bf16 baseline)
2. vLLM Marlin CUDA kernel (未集成)

性能对比 (待实测后更新):
- PyTorch dequant + mm: 先反量化为 bf16 再 matmul，内存带宽利用低
- vLLM Marlin kernel: 直接在 INT4/INT8 上计算，理论吞吐量 2-4x

基线选择策略:
  当前使用 PyTorch dequant + mm 作为基线（通用且无依赖）。
  vLLM Marlin kernel 集成后需对比，预期 INT4 场景提升 2-4x。
  一旦可用且经过验证，应切换为 Marlin kernel。
"""

import torch
import torch.nn.functional as F
from baseline.operators.registry import BaseOperator, register_operator

# 尝试导入 vLLM Marlin ops
try:
    import torch.ops._C as vllm_ops
    HAS_VLLM_MARLIN = hasattr(vllm_ops, 'marlin_gemm')
except (ImportError, AttributeError):
    HAS_VLLM_MARLIN = False


@register_operator("fused_marlin_moe")
class FusedMarlinMoEOperator(BaseOperator):
    """Fused Marlin MoE: INT4/INT8 量化的 MoE GEMM

    Input:
        x: (num_tokens, hidden_size) - activation (bf16)
        expert_weights_int: (num_experts, hidden_size, expert_size) - 量化权重 (int8 模拟)
        scales: (num_experts, expert_size) - 反量化 scale
        router_logits: (num_tokens, num_experts) - routing scores
    Output:
        (num_tokens, expert_size)
    """

    @property
    def name(self) -> str:
        return "fused_marlin_moe"

    def forward(self, x: torch.Tensor, expert_weights_int: torch.Tensor,
                scales: torch.Tensor, router_logits: torch.Tensor,
                top_k: int = 2, **kwargs) -> torch.Tensor:
        """Fused Marlin MoE 前向

        可用实现:
        1. PyTorch dequant + mm ⭐ 当前基线
        2. vLLM Marlin CUDA kernel (未集成)

        基线选择: PyTorch dequant + mm（通用无依赖）
        TODO: 集成 Marlin kernel 后对比，预期 INT4 提升 2-4x
        """
        num_tokens = x.shape[0]
        num_experts = expert_weights_int.shape[0]
        expert_size = expert_weights_int.shape[2]

        # Step 1: TopK routing
        topk_weights, topk_ids = torch.topk(router_logits, top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32).to(x.dtype)

        # Step 2: 向量化 expert computation (dequant + mm)
        flat_topk_ids = topk_ids.reshape(-1)
        flat_weights = topk_weights.reshape(-1)
        x_expanded = x.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, x.shape[-1])

        output = torch.zeros(num_tokens * top_k, expert_size,
                           device=x.device, dtype=x.dtype)

        for expert_id in range(num_experts):
            mask = (flat_topk_ids == expert_id)
            if not mask.any():
                continue
            expert_input = x_expanded[mask]

            # Dequantize: int8 * scale -> bf16
            weight_dequant = expert_weights_int[expert_id].to(x.dtype) * scales[expert_id].to(x.dtype).unsqueeze(0)

            # GEMM
            expert_output = torch.mm(expert_input, weight_dequant)
            output[mask] = expert_output * flat_weights[mask].unsqueeze(-1)

        # Step 3: Sum over top_k
        output = output.reshape(num_tokens, top_k, -1).sum(dim=1)
        return output

    def compute_flops(self, num_tokens: int, hidden_size: int,
                      expert_size: int, num_experts: int,
                      top_k: int = 2, **kwargs) -> int:
        """FLOPs = TopK + Dequant + Matmul * top_k"""
        topk_flops = num_tokens * num_experts
        # Dequant: hidden_size * expert_size * num_experts (scale mul)
        dequant_flops = num_experts * hidden_size * expert_size
        # Matmul
        matmul_flops = num_tokens * top_k * hidden_size * expert_size * 2
        return topk_flops + dequant_flops + matmul_flops

    def compute_bytes(self, num_tokens: int, hidden_size: int,
                      expert_size: int, num_experts: int,
                      top_k: int = 2, dtype: str = "bf16", **kwargs) -> int:
        """访存 = 读 x + 读 int8 weights + 读 scales + 读 logits + 写 output"""
        elem_bytes = self.dtype_bytes(dtype)
        read_x = num_tokens * hidden_size * elem_bytes
        read_weights = num_experts * hidden_size * expert_size * 1  # int8 = 1 byte
        read_scales = num_experts * expert_size * 4  # float32
        read_logits = num_tokens * num_experts * elem_bytes
        write_out = num_tokens * expert_size * elem_bytes
        return read_x + read_weights + read_scales + read_logits + write_out

    def prepare_inputs(self, num_tokens: int, hidden_size: int,
                       expert_size: int, num_experts: int,
                       top_k: int = 2, dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(num_tokens, hidden_size, device=self.device, dtype=torch_dtype)
        # 模拟 INT8 量化权重
        expert_weights_int = torch.randint(-127, 127, (num_experts, hidden_size, expert_size),
                                          device=self.device, dtype=torch.int8)
        # Per-column scales
        scales = torch.rand(num_experts, expert_size, device=self.device, dtype=torch.float32) * 0.01
        router_logits = torch.randn(num_tokens, num_experts, device=self.device, dtype=torch_dtype)
        return {
            "x": x,
            "expert_weights_int": expert_weights_int,
            "scales": scales,
            "router_logits": router_logits,
            "top_k": top_k
        }

    def compute_golden(self, x: torch.Tensor, expert_weights_int: torch.Tensor,
                       scales: torch.Tensor, router_logits: torch.Tensor,
                       top_k: int = 2, **kwargs) -> torch.Tensor:
        """Golden reference (CPU FP32)"""
        x_fp32 = x.float().cpu()
        w_int = expert_weights_int.cpu()
        s_fp32 = scales.float().cpu()
        r_fp32 = router_logits.float().cpu()

        num_tokens = x_fp32.shape[0]
        expert_size = w_int.shape[2]

        topk_weights, topk_ids = torch.topk(r_fp32, top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)

        output = torch.zeros(num_tokens, expert_size)
        for token_id in range(num_tokens):
            for k_idx in range(top_k):
                expert_id = topk_ids[token_id, k_idx].item()
                weight = topk_weights[token_id, k_idx]
                # Dequantize
                w_dequant = w_int[expert_id].float() * s_fp32[expert_id].unsqueeze(0)
                expert_out = torch.mm(x_fp32[token_id:token_id+1], w_dequant)
                output[token_id] += weight * expert_out.squeeze(0)

        return output.to(x.dtype).to(x.device)
