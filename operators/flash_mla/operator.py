"""Flash MLA算子

Flash Multi-head Latent Attention，用于DeepSeek-V3的MLA decode阶段。
使用paged KV cache，支持variable-length序列。

签名:
    flash_mla(q, block_table, blocked_k, max_seqlen_pad, block_size,
              b, s_q, cache_seqlens, h_q, h_kv, d, dv, causal)
        -> output (b, s_q, h_q, dv)

输入:
    q: (b, s_q, h_q, d)                    — query (bfloat16)
    block_table: (b, max_num_blocks)       — page table (int32)
    blocked_k: (total_pages, block_size, h_kv, d) — paged KV cache (bfloat16)
    max_seqlen_pad: int                    — 最大序列长度(padding到block_size倍数)
    block_size: int                        — 每个page的token数
    b: int                                 — batch size
    s_q: int                               — query序列长度（decode时为1）
    cache_seqlens: (b,)                    — 每个序列的实际KV长度 (int32)
    h_q: int                               — query head数
    h_kv: int                              — KV head数（MLA为1）
    d: int                                 — head dim含rope
    dv: int                                — value dim（不含rope）
    causal: bool                           — 因果mask

输出:
    out: (b, s_q, h_q, dv) (bfloat16)
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("flash_mla")
class FlashMLAOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "flash_mla"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            b: batch size
            s_q: query序列长度（decode=1）
            h_q: query head数
            h_kv: kv head数（MLA=1）
            d: head dim含rope（如576=512+64）
            dv: value dim（如512）
            block_size: page大小（通常64）
            max_seq: 最大KV序列长度
            dtype: 数据类型
        """
        b = params["b"]
        s_q = params.get("s_q", 1)
        h_q = params.get("h_q", 128)
        h_kv = params.get("h_kv", 1)
        d = params.get("d", 576)
        dv = params.get("dv", 512)
        block_size = params.get("block_size", 64)
        max_seq = params.get("max_seq", 4096)
        dtype = self.get_dtype(params.get("dtype", "bfloat16"))

        # 计算page数
        num_blocks_per_seq = (max_seq + block_size - 1) // block_size
        total_blocks = b * num_blocks_per_seq
        max_seqlen_pad = num_blocks_per_seq * block_size

        # q: (b, s_q, h_q, d)
        q = torch.randn(b, s_q, h_q, d, dtype=dtype, device="cuda")

        # blocked_k: (total_blocks, block_size, h_kv, d)
        blocked_k = torch.randn(
            total_blocks, block_size, h_kv, d, dtype=dtype, device="cuda"
        )

        # block_table: (b, num_blocks_per_seq)
        block_table = torch.arange(
            total_blocks, dtype=torch.int32, device="cuda"
        ).view(b, num_blocks_per_seq)

        # cache_seqlens: 每个序列的实际长度
        cache_seqlens = torch.full((b,), max_seq, dtype=torch.int32, device="cuda")

        return {
            "q": q,
            "block_table": block_table,
            "blocked_k": blocked_k,
            "max_seqlen_pad": max_seqlen_pad,
            "block_size": block_size,
            "b": b,
            "s_q": s_q,
            "cache_seqlens": cache_seqlens,
            "h_q": h_q,
            "h_kv": h_kv,
            "d": d,
            "dv": dv,
            "causal": True,
        }

    def compute_flops(self, **params):
        """理论FLOPs

        Attention: Q @ K^T + softmax + attn @ V
          - Q @ K^T: b * s_q * h_q * max_seq * d * 2
          - softmax: b * s_q * h_q * max_seq * 5
          - attn @ V: b * s_q * h_q * max_seq * dv * 2
        """
        b = params["b"]
        s_q = params.get("s_q", 1)
        h_q = params.get("h_q", 128)
        d = params.get("d", 576)
        dv = params.get("dv", 512)
        max_seq = params.get("max_seq", 4096)

        # Q @ K^T
        qk_flops = 2 * b * s_q * h_q * max_seq * d
        # softmax
        softmax_flops = 5 * b * s_q * h_q * max_seq
        # attn @ V
        av_flops = 2 * b * s_q * h_q * max_seq * dv

        return int(qk_flops + softmax_flops + av_flops)

    def compute_bytes(self, **params):
        """理论访存量

        读:
          q: b * s_q * h_q * d (dtype)
          blocked_k (有效部分): b * max_seq * h_kv * d (dtype)
          block_table: b * num_blocks * 4 (int32)
          cache_seqlens: b * 4 (int32)
        写:
          out: b * s_q * h_q * dv (dtype)
        """
        b = params["b"]
        s_q = params.get("s_q", 1)
        h_q = params.get("h_q", 128)
        h_kv = params.get("h_kv", 1)
        d = params.get("d", 576)
        dv = params.get("dv", 512)
        max_seq = params.get("max_seq", 4096)
        block_size = params.get("block_size", 64)
        elem_bytes = self.dtype_bytes(params.get("dtype", "bfloat16"))

        num_blocks_per_seq = (max_seq + block_size - 1) // block_size

        read_bytes = (
            b * s_q * h_q * d * elem_bytes          # q
            + b * max_seq * h_kv * d * elem_bytes   # KV cache (effective)
            + b * num_blocks_per_seq * 4            # block_table
            + b * 4                                 # cache_seqlens
        )
        write_bytes = b * s_q * h_q * dv * elem_bytes  # output

        return int(read_bytes + write_bytes)
