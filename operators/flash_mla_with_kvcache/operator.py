"""Flash MLA with KVCache算子 (Dense BF16)

Dense decode模式: q(bf16) + k_cache(bf16), 通过block_table + cache_seqlens索引paged KV cache。

签名:
    flash_mla_with_kvcache(
        q, k_cache, block_table, cache_seqlens, head_dim_v,
        tile_scheduler_metadata,
        softmax_scale=None, causal=False,
        is_fp8_kvcache=False,
    ) -> (out, softmax_lse)

输入:
    q: (B, s_q, h_q, d) bf16
    k_cache: (num_blocks, page_block_size, 1, d) bf16
    block_table: (B, max_blocks_per_seq) int32
    cache_seqlens: (B,) int32

输出:
    out: (B, s_q, h_q, head_dim_v) bf16
    softmax_lse: (B, h_q, s_q) fp32
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("flash_mla_with_kvcache")
class FlashMLAWithKVCacheOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "flash_mla_with_kvcache"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            b: batch size
            s_q: query序列长度 (decode=1)
            h_q: query head数
            head_dim_k: K head dim (512或576)
            head_dim_v: V head dim (512)
            block_size: page大小
            max_seq: 最大序列长度
        """
        from flaggems_vllm.ops.flash_mla_with_kvcache import FlashMLASchedMeta

        b = params["b"]
        s_q = params.get("s_q", 1)
        h_q = params.get("h_q", 128)
        head_dim_k = params.get("head_dim_k", 576)
        head_dim_v = params.get("head_dim_v", 512)
        block_size = params.get("block_size", 64)
        max_seq = params.get("max_seq", 4096)

        num_blocks_per_seq = (max_seq + block_size - 1) // block_size
        total_blocks = b * num_blocks_per_seq

        q = torch.randn(b, s_q, h_q, head_dim_k, dtype=torch.bfloat16, device="cuda")

        # k_cache: (total_blocks, block_size, 1, head_dim_k) bf16
        k_cache = torch.randn(
            total_blocks, block_size, 1, head_dim_k,
            dtype=torch.bfloat16, device="cuda"
        )

        # block_table: (B, num_blocks_per_seq) int32
        block_table = torch.arange(
            total_blocks, dtype=torch.int32, device="cuda"
        ).view(b, num_blocks_per_seq)

        # cache_seqlens: (B,) int32
        cache_seqlens = torch.full(
            (b,), max_seq, dtype=torch.int32, device="cuda"
        )

        sched_meta = FlashMLASchedMeta()

        return {
            "q": q,
            "k_cache": k_cache,
            "block_table": block_table,
            "cache_seqlens": cache_seqlens,
            "head_dim_v": head_dim_v,
            "tile_scheduler_metadata": sched_meta,
            "softmax_scale": head_dim_k ** -0.5,
            "causal": False,
            "is_fp8_kvcache": False,
        }

    def compute_flops(self, **params):
        """理论FLOPs

        Attention: Q @ K^T + softmax + attn @ V
        """
        b = params["b"]
        s_q = params.get("s_q", 1)
        h_q = params.get("h_q", 128)
        head_dim_k = params.get("head_dim_k", 576)
        head_dim_v = params.get("head_dim_v", 512)
        max_seq = params.get("max_seq", 4096)

        qk_flops = 2 * b * s_q * h_q * max_seq * head_dim_k
        softmax_flops = 5 * b * s_q * h_q * max_seq
        av_flops = 2 * b * s_q * h_q * max_seq * head_dim_v

        return int(qk_flops + softmax_flops + av_flops)

    def compute_bytes(self, **params):
        """理论访存量"""
        b = params["b"]
        s_q = params.get("s_q", 1)
        h_q = params.get("h_q", 128)
        head_dim_k = params.get("head_dim_k", 576)
        head_dim_v = params.get("head_dim_v", 512)
        max_seq = params.get("max_seq", 4096)
        block_size = params.get("block_size", 64)

        num_blocks_per_seq = (max_seq + block_size - 1) // block_size

        # Q
        q_bytes = b * s_q * h_q * head_dim_k * 2
        # KV cache
        kv_bytes = b * max_seq * head_dim_k * 2
        # block_table + cache_seqlens
        idx_bytes = b * num_blocks_per_seq * 4 + b * 4
        # output
        out_bytes = b * s_q * h_q * head_dim_v * 2

        return int(q_bytes + kv_bytes + idx_bytes + out_bytes)
