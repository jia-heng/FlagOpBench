"""Fused Inverse RoPE + FP8 Quantization 算子

对应算子列表: fused_inv_rope_fp8_quant

融合操作: Inverse RoPE → FP8 量化
应用场景: KV cache 存储前的预处理（先去除位置编码再量化压缩）

可用实现:
1. PyTorch 分步实现 (inv_rope + fp8_quant 组合)
2. vLLM fused CUDA kernel (未集成)

性能对比 (待实测后更新):
- PyTorch 分步: 两次 kernel launch + 中间 tensor，有 bandwidth 开销
- vLLM fused kernel: 单 kernel 完成，无中间 tensor 分配

基线选择策略:
  当前使用 PyTorch 分步实现作为基线。
  fused kernel 预期减少 ~30% 延迟（省去中间 tensor 读写）。
  集成后需实测对比，如提升 > 10% 则切换。
"""

import torch
import torch.nn.functional as F
from baseline.operators.registry import BaseOperator, register_operator


@register_operator("fused_inv_rope_fp8_quant")
class FusedInvRopeFP8QuantOperator(BaseOperator):
    """Fused Inverse RoPE + FP8 Quantization

    Input:
        q: (num_tokens, num_heads, head_dim) - 带位置编码的 query/key
        cos: (num_tokens, 1, head_dim) - RoPE cos 表
        sin: (num_tokens, 1, head_dim) - RoPE sin 表
    Output:
        q_quant: (num_tokens, num_heads, head_dim) - int8 量化后的结果
        scale: (num_tokens, num_heads) - 量化 scale
    """

    @property
    def name(self) -> str:
        return "fused_inv_rope_fp8_quant"

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """Split and rotate: [x1, x2] -> [-x2, x1]"""
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def forward(self, q: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor,
                **kwargs) -> tuple:
        """Fused Inverse RoPE + FP8 量化

        可用实现:
        1. PyTorch 分步 (inv_rope + quant) ⭐ 当前基线
        2. vLLM fused kernel (未集成)

        基线选择: PyTorch 分步实现
        TODO: 集成 fused kernel 后对比，预期减少 ~30% 延迟
        """
        # Step 1: Inverse RoPE (逆操作: q_orig = q * cos - rotate_half(q) * sin)
        # RoPE forward: q_rot = q * cos + rotate_half(q) * sin
        # RoPE inverse: q_orig = q * cos - rotate_half(q) * sin
        #   (因为 cos^2 + sin^2 = 1，逆操作等价于用 -sin)
        q_inv_rope = q * cos - self._rotate_half(q) * sin

        # Step 2: Per-head FP8 量化
        # 计算每个 head 的 absmax scale
        scale = q_inv_rope.abs().amax(dim=-1) / 127.0  # (num_tokens, num_heads)
        scale = scale.clamp(min=1e-8)

        # 量化
        q_quant = (q_inv_rope / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)

        return q_quant, scale

    def compute_flops(self, num_tokens: int, num_heads: int,
                      head_dim: int, **kwargs) -> int:
        """FLOPs = InvRoPE(4*N) + Quant(3*N)"""
        total_elements = num_tokens * num_heads * head_dim
        inv_rope_flops = 4 * total_elements  # mul, rotate, mul, sub
        quant_flops = 3 * total_elements  # abs, div, round
        return inv_rope_flops + quant_flops

    def compute_bytes(self, num_tokens: int, num_heads: int,
                      head_dim: int, dtype: str = "bf16", **kwargs) -> int:
        """访存 = 读 q + 读 cos/sin + 写 q_quant + 写 scale"""
        elem_bytes = self.dtype_bytes(dtype)
        read_q = num_tokens * num_heads * head_dim * elem_bytes
        read_cos_sin = 2 * num_tokens * 1 * head_dim * elem_bytes
        write_quant = num_tokens * num_heads * head_dim * 1  # int8
        write_scale = num_tokens * num_heads * 4  # float32
        return read_q + read_cos_sin + write_quant + write_scale

    def prepare_inputs(self, num_tokens: int, num_heads: int,
                       head_dim: int, dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        q = torch.randn(num_tokens, num_heads, head_dim,
                       device=self.device, dtype=torch_dtype)
        cos = torch.randn(num_tokens, 1, head_dim,
                         device=self.device, dtype=torch_dtype)
        sin = torch.randn(num_tokens, 1, head_dim,
                         device=self.device, dtype=torch_dtype)
        return {"q": q, "cos": cos, "sin": sin}

    def compute_golden(self, q: torch.Tensor, cos: torch.Tensor,
                       sin: torch.Tensor, **kwargs) -> tuple:
        """Golden reference (CPU FP32)"""
        q_fp32 = q.float().cpu()
        cos_fp32 = cos.float().cpu()
        sin_fp32 = sin.float().cpu()

        # Inverse RoPE
        q1, q2 = q_fp32.chunk(2, dim=-1)
        q_rot_half = torch.cat([-q2, q1], dim=-1)
        q_inv = q_fp32 * cos_fp32 - q_rot_half * sin_fp32

        # Quantize
        scale = q_inv.abs().amax(dim=-1) / 127.0
        scale = scale.clamp(min=1e-8)
        q_quant = (q_inv / scale.unsqueeze(-1)).round().clamp(-127, 127).to(torch.int8)

        return q_quant.to(q.device), scale.to(q.dtype).to(q.device)
