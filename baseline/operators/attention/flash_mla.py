"""Flash MLA (Multi-Head Latent Attention) 算子

对应算子列表: flash_mla
来源: DeepSeek-V2/V3 架构

核心思想: 将 KV 通过低秩压缩 (low-rank) 减少 KV cache 显存占用
KV cache 存储压缩后的 latent vector，推理时动态还原

可用实现:
1. PyTorch 分步实现 (低秩还原 + scaled_dot_product_attention)
2. flash-attn + custom projection (未集成)
3. DeepSeek 官方 CUDA kernel (未开源)

性能对比 (待实测后更新):
- PyTorch 分步: matmul(还原) + SDPA，两次 kernel，但 SDPA 本身高效
- flash-attn custom: 可能单 kernel 完成投影+attention
- 自定义 kernel: 理论最优，但实现复杂

基线选择策略:
  使用 PyTorch 低秩还原 + SDPA 作为基线。
  SDPA 已内置 FlashAttention-2，attention 部分性能有保障。
  主要优化点在 KV 还原投影，如有 fused kernel 可对比。
"""

import torch
import torch.nn.functional as F
import math
from baseline.operators.registry import BaseOperator, register_operator


@register_operator("flash_mla")
class FlashMLAOperator(BaseOperator):
    """Flash MLA: Multi-Head Latent Attention (DeepSeek-V2/V3)

    核心: KV cache 使用低秩 latent vector，推理时通过投影还原
    - 传统 MHA: cache = (K, V) → 显存大
    - MLA: cache = c_kv (latent) → 通过 W_k, W_v 还原 K, V

    Input:
        q: (batch, seq_len, num_heads, head_dim)
        c_kv: (batch, kv_seq_len, kv_lora_rank) - 压缩的 KV latent
        w_k: (kv_lora_rank, num_kv_heads * head_dim) - K 还原投影
        w_v: (kv_lora_rank, num_kv_heads * head_dim) - V 还原投影
    Output:
        (batch, seq_len, num_heads, head_dim)
    """

    @property
    def name(self) -> str:
        return "flash_mla"

    def forward(self, q: torch.Tensor, c_kv: torch.Tensor,
                w_k: torch.Tensor, w_v: torch.Tensor,
                num_kv_heads: int = 8, causal: bool = True,
                **kwargs) -> torch.Tensor:
        """Flash MLA 前向

        可用实现:
        1. PyTorch 低秩还原 + SDPA ⭐ 当前基线
        2. fused projection + attention (未集成)

        基线选择: PyTorch 分步（SDPA 部分已足够高效）
        TODO: 如有 fused kernel 可对比投影+attention 的整体性能
        """
        batch, seq_len, num_heads, head_dim = q.shape
        kv_seq_len = c_kv.shape[1]

        # Step 1: 从 latent 还原 K, V
        # c_kv: (batch, kv_seq_len, kv_lora_rank)
        # w_k: (kv_lora_rank, num_kv_heads * head_dim)
        k = torch.matmul(c_kv, w_k)  # (batch, kv_seq_len, num_kv_heads * head_dim)
        v = torch.matmul(c_kv, w_v)  # (batch, kv_seq_len, num_kv_heads * head_dim)

        # Reshape to multi-head format
        k = k.reshape(batch, kv_seq_len, num_kv_heads, head_dim)
        v = v.reshape(batch, kv_seq_len, num_kv_heads, head_dim)

        # Step 2: GQA - 扩展 KV heads 到 match Q heads (如果需要)
        num_groups = num_heads // num_kv_heads
        if num_groups > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, num_groups, -1)
            k = k.reshape(batch, kv_seq_len, num_heads, head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, num_groups, -1)
            v = v.reshape(batch, kv_seq_len, num_heads, head_dim)

        # Step 3: Attention (使用 PyTorch SDPA)
        # 转换为 (batch, num_heads, seq_len, head_dim)
        q_t = q.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        # Decode 场景 (seq_len < kv_seq_len): 不使用 causal mask
        # 因为当前 token 可以 attend 到所有历史 KV cache
        use_causal = causal and (seq_len == kv_seq_len)

        if hasattr(F, 'scaled_dot_product_attention'):
            out = F.scaled_dot_product_attention(
                q_t, k_t, v_t, is_causal=use_causal
            )
        else:
            # Fallback
            scale = 1.0 / math.sqrt(head_dim)
            scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale
            if causal and seq_len == kv_seq_len:
                mask = torch.triu(
                    torch.ones(seq_len, kv_seq_len, device=q.device, dtype=torch.bool),
                    diagonal=1
                )
                scores = scores.masked_fill(mask, float('-inf'))
            out = torch.matmul(F.softmax(scores, dim=-1), v_t)

        # (batch, num_heads, seq_len, head_dim) -> (batch, seq_len, num_heads, head_dim)
        return out.transpose(1, 2)

    def compute_flops(self, batch: int, seq_len: int, kv_seq_len: int,
                      num_heads: int, num_kv_heads: int,
                      head_dim: int, kv_lora_rank: int,
                      **kwargs) -> int:
        """FLOPs = KV Projection + Attention"""
        # KV projection: 2 * batch * kv_seq_len * kv_lora_rank * (num_kv_heads * head_dim) * 2
        proj_flops = 2 * batch * kv_seq_len * kv_lora_rank * num_kv_heads * head_dim * 2
        # Attention: 4 * batch * num_heads * seq_len * kv_seq_len * head_dim
        attn_flops = 4 * batch * num_heads * seq_len * kv_seq_len * head_dim
        return proj_flops + attn_flops

    def compute_bytes(self, batch: int, seq_len: int, kv_seq_len: int,
                      num_heads: int, num_kv_heads: int,
                      head_dim: int, kv_lora_rank: int,
                      dtype: str = "bf16", **kwargs) -> int:
        """访存 = 读 q + 读 c_kv + 读 w_k/w_v + 写 output"""
        elem_bytes = self.dtype_bytes(dtype)
        read_q = batch * seq_len * num_heads * head_dim * elem_bytes
        read_ckv = batch * kv_seq_len * kv_lora_rank * elem_bytes
        read_proj = 2 * kv_lora_rank * num_kv_heads * head_dim * elem_bytes
        write_out = batch * seq_len * num_heads * head_dim * elem_bytes
        return read_q + read_ckv + read_proj + write_out

    def prepare_inputs(self, batch: int, seq_len: int, kv_seq_len: int,
                       num_heads: int, num_kv_heads: int,
                       head_dim: int, kv_lora_rank: int,
                       causal: bool = True, dtype: str = "bf16",
                       **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        q = torch.randn(batch, seq_len, num_heads, head_dim,
                       device=self.device, dtype=torch_dtype)
        c_kv = torch.randn(batch, kv_seq_len, kv_lora_rank,
                          device=self.device, dtype=torch_dtype)
        w_k = torch.randn(kv_lora_rank, num_kv_heads * head_dim,
                         device=self.device, dtype=torch_dtype) * 0.02
        w_v = torch.randn(kv_lora_rank, num_kv_heads * head_dim,
                         device=self.device, dtype=torch_dtype) * 0.02
        return {
            "q": q, "c_kv": c_kv, "w_k": w_k, "w_v": w_v,
            "num_kv_heads": num_kv_heads, "causal": causal
        }

    def compute_golden(self, q: torch.Tensor, c_kv: torch.Tensor,
                       w_k: torch.Tensor, w_v: torch.Tensor,
                       num_kv_heads: int = 8, causal: bool = True,
                       **kwargs) -> torch.Tensor:
        """Golden reference (CPU FP32)"""
        q_fp32 = q.float().cpu()
        ckv_fp32 = c_kv.float().cpu()
        wk_fp32 = w_k.float().cpu()
        wv_fp32 = w_v.float().cpu()

        batch, seq_len, num_heads, head_dim = q_fp32.shape
        kv_seq_len = ckv_fp32.shape[1]

        # KV projection
        k = torch.matmul(ckv_fp32, wk_fp32).reshape(batch, kv_seq_len, num_kv_heads, head_dim)
        v = torch.matmul(ckv_fp32, wv_fp32).reshape(batch, kv_seq_len, num_kv_heads, head_dim)

        # GQA expand
        num_groups = num_heads // num_kv_heads
        if num_groups > 1:
            k = k.unsqueeze(3).expand(-1, -1, -1, num_groups, -1).reshape(batch, kv_seq_len, num_heads, head_dim)
            v = v.unsqueeze(3).expand(-1, -1, -1, num_groups, -1).reshape(batch, kv_seq_len, num_heads, head_dim)

        # Attention
        q_t = q_fp32.transpose(1, 2)
        k_t = k.transpose(1, 2)
        v_t = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q_t, k_t.transpose(-2, -1)) * scale

        if causal and seq_len == kv_seq_len:
            mask = torch.triu(torch.ones(seq_len, kv_seq_len, dtype=torch.bool), diagonal=1)
            scores = scores.masked_fill(mask, float('-inf'))

        out = torch.matmul(F.softmax(scores, dim=-1), v_t)
        out = out.transpose(1, 2)

        return out.to(q.dtype).to(q.device)
