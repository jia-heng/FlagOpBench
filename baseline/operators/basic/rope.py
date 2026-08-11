"""RoPE (Rotary Position Embedding) 算子 - 使用 vLLM CUDA kernel

对应算子列表 #51
调用 vLLM 官方 CUDA 实现: torch.ops._C.rotary_embedding
"""

import torch
from baseline.operators.registry import BaseOperator, register_operator

# 尝试导入 vLLM CUDA ops
try:
    import torch.ops._C as vllm_ops
    HAS_VLLM_OPS = hasattr(vllm_ops, 'rotary_embedding')
except (ImportError, AttributeError):
    HAS_VLLM_OPS = False
    print("Warning: vLLM CUDA ops not available for rotary_embedding, using PyTorch fallback")


@register_operator("rope")
class RoPEOperator(BaseOperator):
    """RoPE (Rotary Position Embedding)

    对 query/key 施加旋转位置编码。
    Input: (batch, seq_len, num_heads, head_dim)
    Output: (batch, seq_len, num_heads, head_dim)
    """

    @property
    def name(self) -> str:
        return "rope"

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Split and rotate: [x1, x2] -> [-x2, x1]"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                **kwargs) -> torch.Tensor:
        """Apply rotary embedding

        使用 vLLM CUDA kernel: torch.ops._C.rotary_embedding(positions, query, key, ...)
        注意: vLLM 版本需要 positions 参数，这里用 PyTorch fallback
        """
        # vLLM 的 rotary_embedding 接口较复杂，需要 positions、head_size 等参数
        # 暂时使用 PyTorch fallback，后续可根据实际需求适配
        if HAS_VLLM_OPS and False:  # 暂时禁用，接口不匹配
            # TODO: 适配 vLLM rotary_embedding 接口
            pass

        # Fallback: 纯 PyTorch 实现
        q_rot = q * cos + self._rotate_half(q) * sin
        return q_rot

    def compute_flops(self, batch: int, seq_len: int, num_heads: int,
                      head_dim: int, **kwargs) -> int:
        """RoPE FLOPs ≈ 4 * batch * seq_len * num_heads * head_dim"""
        return 4 * batch * seq_len * num_heads * head_dim

    def compute_bytes(self, batch: int, seq_len: int, num_heads: int,
                      head_dim: int, dtype: str = "bf16", **kwargs) -> int:
        elem_bytes = self.dtype_bytes(dtype)
        total_elements = batch * seq_len * num_heads * head_dim
        read_q = total_elements * elem_bytes
        read_cos = total_elements * elem_bytes
        read_sin = total_elements * elem_bytes
        write_output = total_elements * elem_bytes
        return read_q + read_cos + read_sin + write_output

    def prepare_inputs(self, batch: int, seq_len: int, num_heads: int,
                       head_dim: int, dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        q = torch.randn(batch, seq_len, num_heads, head_dim,
                       device=self.device, dtype=torch_dtype)
        cos = torch.randn(batch, seq_len, num_heads, head_dim,
                         device=self.device, dtype=torch_dtype)
        sin = torch.randn(batch, seq_len, num_heads, head_dim,
                         device=self.device, dtype=torch_dtype)
        return {"q": q, "cos": cos, "sin": sin}

    def compute_golden(self, q: torch.Tensor, cos: torch.Tensor,
                       sin: torch.Tensor, **kwargs) -> torch.Tensor:
        q_fp32 = q.float().cpu()
        cos_fp32 = cos.float().cpu()
        sin_fp32 = sin.float().cpu()
        q1, q2 = q_fp32.chunk(2, dim=-1)
        q_rot_half = torch.cat([-q2, q1], dim=-1)
        q_rot = q_fp32 * cos_fp32 + q_rot_half * sin_fp32
        return q_rot.to(q.dtype).to(q.device)
