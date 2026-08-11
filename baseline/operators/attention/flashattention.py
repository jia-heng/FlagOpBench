"""Flash Attention 算子

标准的 Flash Attention 实现

可用实现:
1. PyTorch 官方 F.scaled_dot_product_attention (PyTorch 2.0+)
2. flash-attn 库 (flash_attn_func)
3. 手动实现 (fallback)

性能对比 (待实测后更新):
- PyTorch SDPA: 内部自动选择 FlashAttention-2/Memory-Efficient/Math 后端
- flash-attn 库: 直接调用 FlashAttention-2 CUDA kernel
- 手动实现: 纯 PyTorch ops 组合，无 memory-efficient 优化

基线选择策略:
  PyTorch SDPA 和 flash-attn 底层均使用 FlashAttention-2 kernel，
  性能差异 < 5%，优先选择 PyTorch SDPA（兼容性最好，无额外依赖）。
  实测后根据结果调整。
"""

import torch
import torch.nn.functional as F
import math
from baseline.operators.registry import BaseOperator, register_operator

# 尝试导入 flash-attn
try:
    from flash_attn import flash_attn_func
    HAS_FLASH_ATTN = True
except ImportError:
    HAS_FLASH_ATTN = False


@register_operator("flashattention")
class FlashAttentionOperator(BaseOperator):
    """Flash Attention: 高效的 Attention 实现

    Input:
        q, k, v: (batch, seq_len, num_heads, head_dim)
    Output:
        (batch, seq_len, num_heads, head_dim)
    """

    @property
    def name(self) -> str:
        return "flashattention"

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                causal: bool = True, **kwargs) -> torch.Tensor:
        """Flash Attention 前向

        可用实现及性能预期:
        1. PyTorch SDPA - 内部自动选择最优后端 (FlashAttention-2/xformers)
        2. flash-attn 库 - 直接调用 FlashAttention-2 kernel
        3. 手动实现 - 纯 PyTorch matmul+softmax，无 tiling 优化

        基线选择: PyTorch SDPA（性能与 flash-attn 相当，兼容性更好）
        TODO: 实测对比三种实现的性能，如 flash-attn 显著更优则切换
        """
        if hasattr(F, 'scaled_dot_product_attention'):
            # 基线选择: PyTorch 2.0+ SDPA (内部自动分发到最优 kernel)
            q_t = q.transpose(1, 2)
            k_t = k.transpose(1, 2)
            v_t = v.transpose(1, 2)
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=causal
            )
            return out.transpose(1, 2)
        elif HAS_FLASH_ATTN:
            # 备选: flash-attn 库 (底层相同 kernel，API 不同)
            return flash_attn_func(q, k, v, causal=causal)
        else:
            # Fallback: 手动实现 (性能远低于前两者，仅用于兼容)
            return self._manual_attention(q, k, v, causal)

    def _manual_attention(self, q: torch.Tensor, k: torch.Tensor,
                         v: torch.Tensor, causal: bool) -> torch.Tensor:
        """手动实现的 Attention（用于 fallback）"""
        batch, seq_len, num_heads, head_dim = q.shape

        # (B, S, H, D) -> (B, H, S, D)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        # Attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(head_dim)

        # Causal mask
        if causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=q.device, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(mask, float('-inf'))

        # Softmax + weighted sum
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v)

        # (B, H, S, D) -> (B, S, H, D)
        return out.transpose(1, 2)

    def compute_flops(self, batch: int, seq_len: int, num_heads: int,
                      head_dim: int, **kwargs) -> int:
        """Attention FLOPs ≈ 4 * batch * num_heads * seq_len^2 * head_dim"""
        return 4 * batch * num_heads * seq_len * seq_len * head_dim

    def compute_bytes(self, batch: int, seq_len: int, num_heads: int,
                      head_dim: int, dtype: str = "bf16", **kwargs) -> int:
        """访存 = 读 Q/K/V + 写 output"""
        elem_bytes = self.dtype_bytes(dtype)
        total_elements = batch * seq_len * num_heads * head_dim
        read_qkv = 3 * total_elements * elem_bytes
        write_out = total_elements * elem_bytes
        return read_qkv + write_out

    def prepare_inputs(self, batch: int, seq_len: int, num_heads: int,
                       head_dim: int, dtype: str = "bf16",
                       causal: bool = True, **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        q = torch.randn(batch, seq_len, num_heads, head_dim,
                       device=self.device, dtype=torch_dtype)
        k = torch.randn(batch, seq_len, num_heads, head_dim,
                       device=self.device, dtype=torch_dtype)
        v = torch.randn(batch, seq_len, num_heads, head_dim,
                       device=self.device, dtype=torch_dtype)
        return {"q": q, "k": k, "v": v, "causal": causal}

    def compute_golden(self, q: torch.Tensor, k: torch.Tensor,
                       v: torch.Tensor, causal: bool = True,
                       **kwargs) -> torch.Tensor:
        """Golden reference 使用手动实现"""
        q_fp32 = q.float().cpu()
        k_fp32 = k.float().cpu()
        v_fp32 = v.float().cpu()

        batch, seq_len, num_heads, head_dim = q_fp32.shape

        # (B, S, H, D) -> (B, H, S, D)
        q_fp32 = q_fp32.transpose(1, 2)
        k_fp32 = k_fp32.transpose(1, 2)
        v_fp32 = v_fp32.transpose(1, 2)

        scores = torch.matmul(q_fp32, k_fp32.transpose(-2, -1)) / math.sqrt(head_dim)

        if causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, dtype=torch.bool),
                diagonal=1
            )
            scores = scores.masked_fill(mask, float('-inf'))

        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v_fp32)
        out = out.transpose(1, 2)

        return out.to(q.dtype).to(q.device)
