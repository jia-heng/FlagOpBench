"""TopK 算子族

对应算子列表 #8/#18/#49/#54/#55

可用实现:
1. PyTorch 官方 torch.topk (cuDNN/Thrust 后端)
2. vLLM CUDA kernel (top_k_per_row_decode，专为 decode 场景优化)
3. 手动实现 (无)

性能对比 (待实测后更新):
- torch.topk: 通用 topk，大 N 场景 (vocab_size=128K) 表现稳定
- vLLM top_k_per_row_decode: 专为小 batch decode 优化，小 N 场景可能更优

基线选择策略:
  torch.topk 是当前基线（通用性好，覆盖所有场景）。
  vLLM kernel 仅在 decode (小 batch + 小 N) 场景可能有优势。
  实测后如 vLLM kernel 在 decode 场景提升 > 10%，则为 decode 子类切换实现。
"""

import torch
from baseline.operators.registry import BaseOperator, register_operator

# 尝试导入 vLLM CUDA ops (可选)
try:
    import torch.ops._C as vllm_ops
    HAS_VLLM_TOPK = hasattr(vllm_ops, 'top_k_per_row_decode')
except (ImportError, AttributeError):
    HAS_VLLM_TOPK = False


@register_operator("top_k_per_row_prefill")
class TopKPerRowPrefillOperator(BaseOperator):
    """TopK per row - Prefill 场景 (大 batch)"""

    @property
    def name(self) -> str:
        return "top_k_per_row_prefill"

    def forward(self, x: torch.Tensor, k: int = 8, **kwargs) -> torch.Tensor:
        """TopK per row - PyTorch 官方实现"""
        values, indices = torch.topk(x, k, dim=-1)
        return values

    def compute_flops(self, num_tokens: int, num_experts: int,
                      k: int = 8, **kwargs) -> int:
        return num_tokens * num_experts

    def compute_bytes(self, num_tokens: int, num_experts: int,
                      k: int = 8, dtype: str = "bf16", **kwargs) -> int:
        elem_bytes = self.dtype_bytes(dtype)
        read_input = num_tokens * num_experts * elem_bytes
        write_values = num_tokens * k * elem_bytes
        write_indices = num_tokens * k * 4  # int32 indices
        return read_input + write_values + write_indices

    def prepare_inputs(self, num_tokens: int, num_experts: int,
                       k: int = 8, dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(num_tokens, num_experts, device=self.device, dtype=torch_dtype)
        return {"x": x, "k": k}

    def compute_golden(self, x: torch.Tensor, k: int = 8, **kwargs) -> torch.Tensor:
        x_fp32 = x.float().cpu()
        values, _ = torch.topk(x_fp32, k, dim=-1)
        return values.to(x.dtype).to(x.device)


@register_operator("top_k_per_row_decode")
class TopKPerRowDecodeOperator(BaseOperator):
    """TopK per row - Decode 场景 (小 batch)"""

    @property
    def name(self) -> str:
        return "top_k_per_row_decode"

    def forward(self, x: torch.Tensor, k: int = 8, **kwargs) -> torch.Tensor:
        """TopK per row - Decode 场景

        优先级:
        1. PyTorch 官方 torch.topk (最推荐)
        2. vLLM CUDA kernel (可选优化)
        """
        values, indices = torch.topk(x, k, dim=-1)
        return values

    def compute_flops(self, num_tokens: int, num_experts: int,
                      k: int = 8, **kwargs) -> int:
        return num_tokens * num_experts

    def compute_bytes(self, num_tokens: int, num_experts: int,
                      k: int = 8, dtype: str = "bf16", **kwargs) -> int:
        elem_bytes = self.dtype_bytes(dtype)
        read_input = num_tokens * num_experts * elem_bytes
        write_values = num_tokens * k * elem_bytes
        write_indices = num_tokens * k * 4
        return read_input + write_values + write_indices

    def prepare_inputs(self, num_tokens: int, num_experts: int,
                       k: int = 8, dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(num_tokens, num_experts, device=self.device, dtype=torch_dtype)
        return {"x": x, "k": k}

    def compute_golden(self, x: torch.Tensor, k: int = 8, **kwargs) -> torch.Tensor:
        x_fp32 = x.float().cpu()
        values, _ = torch.topk(x_fp32, k, dim=-1)
        return values.to(x.dtype).to(x.device)


@register_operator("persistent_topk")
class PersistentTopKOperator(BaseOperator):
    """Persistent TopK - 大 K 值场景"""

    @property
    def name(self) -> str:
        return "persistent_topk"

    def forward(self, x: torch.Tensor, k: int = 64, **kwargs) -> torch.Tensor:
        values, indices = torch.topk(x, k, dim=-1)
        return values

    def compute_flops(self, num_tokens: int, N: int,
                      k: int = 64, **kwargs) -> int:
        return num_tokens * N

    def compute_bytes(self, num_tokens: int, N: int,
                      k: int = 64, dtype: str = "bf16", **kwargs) -> int:
        elem_bytes = self.dtype_bytes(dtype)
        read_input = num_tokens * N * elem_bytes
        write_values = num_tokens * k * elem_bytes
        write_indices = num_tokens * k * 4
        return read_input + write_values + write_indices

    def prepare_inputs(self, num_tokens: int, N: int,
                       k: int = 64, dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(num_tokens, N, device=self.device, dtype=torch_dtype)
        return {"x": x, "k": k}

    def compute_golden(self, x: torch.Tensor, k: int = 64, **kwargs) -> torch.Tensor:
        x_fp32 = x.float().cpu()
        values, _ = torch.topk(x_fp32, k, dim=-1)
        return values.to(x.dtype).to(x.device)


@register_operator("topk_selector")
class TopKSelectorOperator(BaseOperator):
    """TopK Selector: topk + gather 选出对应 hidden states"""

    @property
    def name(self) -> str:
        return "topk_selector"

    def forward(self, scores: torch.Tensor, hidden_states: torch.Tensor,
                k: int = 8, **kwargs) -> torch.Tensor:
        """Select top-k experts and gather their hidden states"""
        # scores: (num_tokens, num_experts)
        # hidden_states: (num_tokens, num_experts, hidden_size)
        _, indices = torch.topk(scores, k, dim=-1)  # (num_tokens, k)
        # Expand indices for gather
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, hidden_states.shape[-1])
        selected = torch.gather(hidden_states, 1, indices_expanded)
        return selected

    def compute_flops(self, num_tokens: int, num_experts: int,
                      hidden_size: int, k: int = 8, **kwargs) -> int:
        topk_flops = num_tokens * num_experts
        gather_flops = num_tokens * k * hidden_size  # memory move
        return topk_flops + gather_flops

    def compute_bytes(self, num_tokens: int, num_experts: int,
                      hidden_size: int, k: int = 8,
                      dtype: str = "bf16", **kwargs) -> int:
        elem_bytes = self.dtype_bytes(dtype)
        read_scores = num_tokens * num_experts * elem_bytes
        read_hidden = num_tokens * num_experts * hidden_size * elem_bytes
        write_output = num_tokens * k * hidden_size * elem_bytes
        return read_scores + read_hidden + write_output

    def prepare_inputs(self, num_tokens: int, num_experts: int,
                       hidden_size: int, k: int = 8,
                       dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        scores = torch.randn(num_tokens, num_experts, device=self.device, dtype=torch_dtype)
        hidden_states = torch.randn(
            num_tokens, num_experts, hidden_size,
            device=self.device, dtype=torch_dtype
        )
        return {"scores": scores, "hidden_states": hidden_states, "k": k}

    def compute_golden(self, scores: torch.Tensor, hidden_states: torch.Tensor,
                       k: int = 8, **kwargs) -> torch.Tensor:
        # Keep on GPU to ensure topk tie-breaking matches forward()
        # (CPU/GPU have different stable sort behavior for equal values)
        s_fp32 = scores.float()
        h_fp32 = hidden_states.float()
        _, indices = torch.topk(s_fp32, k, dim=-1)
        indices_expanded = indices.unsqueeze(-1).expand(-1, -1, h_fp32.shape[-1])
        selected = torch.gather(h_fp32, 1, indices_expanded)
        return selected.to(scores.dtype)
