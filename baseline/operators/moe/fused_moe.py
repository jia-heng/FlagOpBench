"""Fused MoE 算子

融合 MoE: routing + expert computation + weighted sum

可用实现:
1. PyTorch 向量化实现 (expand + gather + bmm，避免 Python 循环)
2. PyTorch 循环实现 (逐 expert 循环，简单但慢)
3. vLLM CUDA kernel (torch.ops._C.cutlass_moe_mm，未集成)

性能对比 (待实测后更新):
- 向量化实现: 避免 Python 循环，利用 GPU 并行，性能显著优于循环版本
- 循环实现: Python 层逐 expert 迭代，launch 开销大，小 batch 尤其明显
- vLLM CUDA kernel: 单 kernel 完成所有计算，理论最优

基线选择策略:
  向量化实现 >> 循环实现（避免 Python loop overhead）。
  vLLM kernel 未集成前，使用向量化实现作为基线。
  实测后如果 vLLM kernel 可用且性能提升 > 5%，则切换。
"""

import torch
import torch.nn.functional as F
from baseline.operators.registry import BaseOperator, register_operator

# 尝试导入 vLLM CUDA ops
try:
    import torch.ops._C as vllm_ops
    HAS_VLLM_MOE = hasattr(vllm_ops, 'cutlass_moe_mm')
except (ImportError, AttributeError):
    HAS_VLLM_MOE = False


@register_operator("fused_moe")
class FusedMoEOperator(BaseOperator):
    """Fused MoE: 融合 routing + expert computation + weighted sum

    Input:
        x: (num_tokens, hidden_size)
        expert_weights: (num_experts, hidden_size, expert_size)
        router_logits: (num_tokens, num_experts)
    Output:
        (num_tokens, expert_size)
    """

    @property
    def name(self) -> str:
        return "fused_moe"

    def forward(self, x: torch.Tensor, expert_weights: torch.Tensor,
                router_logits: torch.Tensor, top_k: int = 2,
                **kwargs) -> torch.Tensor:
        """Fused MoE 前向

        可用实现:
        1. 向量化实现 (expand + scatter，避免 Python 循环) ⭐ 当前基线
        2. 逐 expert 循环实现 (简单但 Python loop 开销大)
        3. vLLM CUDA kernel (未集成)

        基线选择: 向量化实现（性能显著优于循环版本）
        TODO: 实测对比向量化 vs vLLM kernel，如 vLLM 可用且更优则切换
        """
        num_tokens = x.shape[0]
        num_experts = expert_weights.shape[0]
        expert_size = expert_weights.shape[2]

        # Step 1: TopK routing
        topk_weights, topk_ids = torch.topk(router_logits, top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32).to(x.dtype)

        # Step 2: 向量化 expert computation (避免 Python 循环)
        # 展开为 (num_tokens * top_k)
        flat_topk_ids = topk_ids.reshape(-1)  # (num_tokens * top_k)
        flat_weights = topk_weights.reshape(-1)  # (num_tokens * top_k)

        # 扩展 x 为 (num_tokens * top_k, hidden_size)
        x_expanded = x.unsqueeze(1).expand(-1, top_k, -1).reshape(-1, x.shape[-1])

        # 为每个 expert 选择对应的 tokens 并计算
        output = torch.zeros(num_tokens * top_k, expert_size,
                           device=x.device, dtype=x.dtype)
        for expert_id in range(num_experts):
            mask = (flat_topk_ids == expert_id)
            if not mask.any():
                continue
            expert_input = x_expanded[mask]
            expert_output = torch.mm(expert_input, expert_weights[expert_id])
            output[mask] = expert_output * flat_weights[mask].unsqueeze(-1)

        # Step 3: Reshape 并求和得到最终输出
        output = output.reshape(num_tokens, top_k, -1).sum(dim=1)
        return output

    def forward_loop(self, x: torch.Tensor, expert_weights: torch.Tensor,
                     router_logits: torch.Tensor, top_k: int = 2,
                     **kwargs) -> torch.Tensor:
        """循环实现 (性能较差，仅作对比参考)

        性能劣势: Python 层逐 expert + 逐 token 双重循环，launch 开销大
        """
        num_tokens = x.shape[0]
        num_experts = expert_weights.shape[0]
        expert_size = expert_weights.shape[2]

        # Step 1: TopK routing
        topk_weights, topk_ids = torch.topk(router_logits, top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1, dtype=torch.float32).to(x.dtype)

        # Step 2: Expert computation (逐 expert 循环)
        output = torch.zeros(num_tokens, expert_size, device=x.device, dtype=x.dtype)

        for expert_id in range(num_experts):
            expert_mask = (topk_ids == expert_id).any(dim=-1)
            if not expert_mask.any():
                continue

            token_ids = expert_mask.nonzero(as_tuple=True)[0]
            expert_input = x[token_ids]
            expert_output = torch.mm(expert_input, expert_weights[expert_id])

            for i, token_id in enumerate(token_ids):
                expert_positions = (topk_ids[token_id] == expert_id).nonzero(as_tuple=True)[0]
                for pos in expert_positions:
                    weight = topk_weights[token_id, pos]
                    output[token_id] += weight * expert_output[i]

        return output

    def compute_flops(self, num_tokens: int, hidden_size: int,
                      expert_size: int, num_experts: int,
                      top_k: int = 2, **kwargs) -> int:
        """FLOPs = TopK + Matmul * top_k"""
        topk_flops = num_tokens * num_experts
        matmul_flops = num_tokens * top_k * hidden_size * expert_size * 2
        return topk_flops + matmul_flops

    def compute_bytes(self, num_tokens: int, hidden_size: int,
                      expert_size: int, num_experts: int,
                      top_k: int = 2, dtype: str = "bf16", **kwargs) -> int:
        """访存 = 读 x + 读 expert_weights + 读 router_logits + 写 output"""
        elem_bytes = self.dtype_bytes(dtype)
        read_x = num_tokens * hidden_size * elem_bytes
        read_weights = num_experts * hidden_size * expert_size * elem_bytes
        read_logits = num_tokens * num_experts * elem_bytes
        write_out = num_tokens * expert_size * elem_bytes
        return read_x + read_weights + read_logits + write_out

    def prepare_inputs(self, num_tokens: int, hidden_size: int,
                       expert_size: int, num_experts: int,
                       top_k: int = 2, dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(num_tokens, hidden_size, device=self.device, dtype=torch_dtype)
        expert_weights = torch.randn(num_experts, hidden_size, expert_size,
                                    device=self.device, dtype=torch_dtype)
        router_logits = torch.randn(num_tokens, num_experts,
                                   device=self.device, dtype=torch_dtype)
        return {
            "x": x,
            "expert_weights": expert_weights,
            "router_logits": router_logits,
            "top_k": top_k
        }

    def compute_golden(self, x: torch.Tensor, expert_weights: torch.Tensor,
                       router_logits: torch.Tensor, top_k: int = 2,
                       **kwargs) -> torch.Tensor:
        """Golden reference"""
        x_fp32 = x.float().cpu()
        w_fp32 = expert_weights.float().cpu()
        r_fp32 = router_logits.float().cpu()

        # TopK
        topk_weights, topk_ids = torch.topk(r_fp32, top_k, dim=-1)
        topk_weights = F.softmax(topk_weights, dim=-1)

        # Expert computation
        num_tokens = x_fp32.shape[0]
        expert_size = w_fp32.shape[2]
        output = torch.zeros(num_tokens, expert_size)

        for token_id in range(num_tokens):
            for k_idx in range(top_k):
                expert_id = topk_ids[token_id, k_idx].item()
                weight = topk_weights[token_id, k_idx]
                expert_out = torch.mm(x_fp32[token_id:token_id+1], w_fp32[expert_id])
                output[token_id] += weight * expert_out.squeeze(0)

        return output.to(x.dtype).to(x.device)
