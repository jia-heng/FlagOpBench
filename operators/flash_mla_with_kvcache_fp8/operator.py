"""Flash MLA with KVCache算子 — 混合精度 (Sparse FP8)

Sparse decode混合精度模式: q(bf16) + k_cache(fp8/uint8), 通过indices稀疏attend。
Q保持BF16高精度，KV Cache用FP8低精度存储，kernel内部做dequant。

签名:
    flash_mla_with_kvcache(
        q, k_cache, block_table=None, cache_seqlens=None, head_dim_v,
        tile_scheduler_metadata,
        softmax_scale=None, causal=False,
        is_fp8_kvcache=True, indices=indices, ...
    ) -> (out, softmax_lse)

输入:
    q: (B, s_q, h_q, head_dim_k) bf16
    k_cache: (num_kv_tokens, 1, 1, cache_token_bytes) uint8
             head_dim_k=512 → 584 bytes/token (MODEL1)
             head_dim_k=576 → 656 bytes/token (V32)
    indices: (B, s_q, topk) int32  — topk必须是64的倍数

输出:
    out: (B, s_q, h_q, head_dim_v) bf16
    softmax_lse: (B, h_q, s_q) fp32
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("flash_mla_with_kvcache_fp8")
class FlashMLAWithKVCacheFP8Operator(BaseOperator):

    @property
    def name(self) -> str:
        return "flash_mla_with_kvcache_fp8"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    @property
    def impl_name(self) -> str:
        """实际调用的函数名与注册名不同"""
        return "flash_mla_with_kvcache"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            b: batch size
            s_q: query序列长度 (decode=1)
            h_q: query head数 (64或128)
            head_dim_k: K head dim (512或576)
            head_dim_v: V head dim (512)
            topk: 每个token attend的KV数 (必须是64的倍数)
            num_kv_tokens: KV token总数
        """
        from flaggems_vllm.ops.flash_mla_with_kvcache import FlashMLASchedMeta

        b = params["b"]
        s_q = params.get("s_q", 1)
        h_q = params.get("h_q", 128)
        head_dim_k = params.get("head_dim_k", 512)
        head_dim_v = params.get("head_dim_v", 512)
        topk = params.get("topk", 64)
        num_kv_tokens = params.get("num_kv_tokens", 8192)

        # MODEL1: head_dim_k=512 → 584 bytes/token
        # V32:    head_dim_k=576 → 656 bytes/token
        if head_dim_k == 576:
            cache_token_bytes = 656
        else:
            cache_token_bytes = 584

        q = torch.randn(b, s_q, h_q, head_dim_k, dtype=torch.bfloat16, device="cuda")

        # k_cache: (num_kv_tokens, 1, 1, cache_token_bytes) uint8
        k_cache = torch.randint(
            0, 255,
            (num_kv_tokens, 1, 1, cache_token_bytes),
            dtype=torch.uint8, device="cuda"
        )

        # indices: (B, s_q, topk) int32
        indices = torch.randint(
            0, num_kv_tokens,
            (b, s_q, topk),
            dtype=torch.int32, device="cuda"
        )

        sched_meta = FlashMLASchedMeta()

        return {
            "q": q,
            "k_cache": k_cache,
            "block_table": None,
            "cache_seqlens": None,
            "head_dim_v": head_dim_v,
            "tile_scheduler_metadata": sched_meta,
            "softmax_scale": head_dim_k ** -0.5,
            "causal": False,
            "is_fp8_kvcache": True,
            "indices": indices,
        }

    def compute_flops(self, **params):
        """理论FLOPs

        Sparse attention只attend topk个KV token:
          Q @ K^T + softmax + attn @ V
        """
        b = params["b"]
        s_q = params.get("s_q", 1)
        h_q = params.get("h_q", 128)
        head_dim_k = params.get("head_dim_k", 512)
        head_dim_v = params.get("head_dim_v", 512)
        topk = params.get("topk", 64)

        qk_flops = 2 * b * s_q * h_q * topk * head_dim_k
        softmax_flops = 5 * b * s_q * h_q * topk
        av_flops = 2 * b * s_q * h_q * topk * head_dim_v

        return int(qk_flops + softmax_flops + av_flops)

    def compute_bytes(self, **params):
        """理论访存量

        读:
          q: B * s_q * h_q * head_dim_k * 2 (bf16)
          k_cache: B * topk * cache_token_bytes (fp8 packed)
          indices: B * s_q * topk * 4 (int32)
        写:
          out: B * s_q * h_q * head_dim_v * 2 (bf16)
        """
        b = params["b"]
        s_q = params.get("s_q", 1)
        h_q = params.get("h_q", 128)
        head_dim_k = params.get("head_dim_k", 512)
        head_dim_v = params.get("head_dim_v", 512)
        topk = params.get("topk", 64)

        if head_dim_k == 576:
            cache_token_bytes = 656
        else:
            cache_token_bytes = 584

        q_bytes = b * s_q * h_q * head_dim_k * 2
        kv_bytes = b * topk * cache_token_bytes
        idx_bytes = b * s_q * topk * 4
        out_bytes = b * s_q * h_q * head_dim_v * 2

        return int(q_bytes + kv_bytes + idx_bytes + out_bytes)
