"""Indexer K Quant and Cache算子

将K tensor量化为FP8并通过slot_mapping写入paged KV cache。

签名:
    indexer_k_quant_and_cache(
        k, kv_cache, slot_mapping, quant_block_size, scale_fmt
    ) -> None (in-place写kv_cache)

输入:
    k: (num_tokens, head_dim) bf16               — K tensor
    kv_cache: (num_blocks, block_size, ...) uint8 — paged KV cache
              Layout: [block_size * head_dim bytes (fp8 data) | scales (fp32)]
    slot_mapping: (num_tokens,) int64            — 每个token对应的slot索引
    quant_block_size: int                        — 量化block大小
    scale_fmt: str                               — scale格式 ("ue8m0" 或 "e4m3")

输出:
    in-place修改kv_cache
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("indexer_k_quant_and_cache")
class IndexerKQuantAndCacheOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "indexer_k_quant_and_cache"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """准备输入

        casegen 路径参数: num_tokens, index_head_dim, index_n_heads, index_topk, dtype
        直接调用参数:     num_tokens, head_dim, block_size, quant_block_size, num_blocks, scale_fmt

        Args:
            num_tokens: token数
            head_dim/index_head_dim: K head维度
            block_size: cache block大小
            quant_block_size: 量化block大小
            num_blocks: cache总block数
            scale_fmt: scale格式
        """
        num_tokens = params["num_tokens"]
        head_dim = params.get("index_head_dim") or params.get("head_dim", 576)
        block_size = params.get("block_size", 16)
        quant_block_size = params.get("quant_block_size", 64)
        num_blocks = params.get("num_blocks", 256)
        scale_fmt = params.get("scale_fmt", "e4m3")

        num_quant_blocks = head_dim // quant_block_size

        # kv_cache: (num_blocks, block_size, ...)
        # flat view: (num_blocks, block_size * head_dim + block_size * num_quant_blocks * 4)
        # 但实际shape需要满足 kv_cache.shape[0]=num_blocks, kv_cache.shape[1]=block_size
        # kv_cache.view(num_blocks, -1) 后 data = [:, :block_size*head_dim], scale = [:, block_size*head_dim:]
        # 每个block: block_size * head_dim (fp8 bytes) + block_size * num_quant_blocks * 4 (scale bytes)
        cols = head_dim + num_quant_blocks * 4
        kv_cache = torch.zeros(
            num_blocks, block_size, cols,
            dtype=torch.uint8, device="cuda"
        )

        # k: (num_tokens, head_dim) bf16
        k = torch.randn(
            num_tokens, head_dim,
            dtype=torch.bfloat16, device="cuda"
        )

        # slot_mapping: (num_tokens,) int64
        # 每个token映射到一个slot: slot_idx = block_idx * block_size + offset
        max_slots = num_blocks * block_size
        slot_mapping = torch.randint(
            0, min(num_tokens * 2, max_slots),
            (num_tokens,),
            dtype=torch.int64, device="cuda"
        )
        # 确保没有重复
        slot_mapping = torch.arange(num_tokens, dtype=torch.int64, device="cuda") % max_slots

        return {
            "k": k,
            "kv_cache": kv_cache,
            "slot_mapping": slot_mapping,
            "quant_block_size": quant_block_size,
            "scale_fmt": scale_fmt,
        }

    def compute_flops(self, **params):
        """量化 + 索引写入

        量化FLOPs: 每个quant block需要找max(abs)和scale -> ~2*quant_block_size per block
        """
        num_tokens = params["num_tokens"]
        head_dim = params.get("index_head_dim") or params.get("head_dim", 576)
        quant_block_size = params.get("quant_block_size", 64)
        num_quant_blocks = head_dim // quant_block_size
        # 每个quant block: reduce找absmax + scale + 量化
        return num_tokens * num_quant_blocks * quant_block_size * 3

    def compute_bytes(self, **params):
        """访存量

        读: k (bf16)
        写: kv_cache data (fp8) + scale (fp32)
        """
        num_tokens = params["num_tokens"]
        head_dim = params.get("index_head_dim") or params.get("head_dim", 576)
        quant_block_size = params.get("quant_block_size", 64)
        num_quant_blocks = head_dim // quant_block_size

        read_bytes = (
            num_tokens * head_dim * 2              # k (bf16)
            + num_tokens * 8                       # slot_mapping (int64)
        )
        write_bytes = (
            num_tokens * head_dim * 1              # fp8 data
            + num_tokens * num_quant_blocks * 4    # scale (fp32)
        )
        return int(read_bytes + write_bytes)
