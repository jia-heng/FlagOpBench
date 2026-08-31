"""CP Gather + Indexer + K Quantize Cache算子

Context-Parallel场景下，将gather到的k_fp8数据写入paged k_cache中。
通过block_table和cu_seqlen定位每个token对应的cache slot。

签名:
    cp_gather_indexer_k_quant_cache(
        k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen
    ) -> None (in-place写k_cache)

输入:
    k_cache: (num_blocks, block_size, ...) uint8  — paged KV cache
             Layout: [block_size * head_dim bytes data | scales (fp32)]
    k_fp8: (num_tokens, head_dim) float8_e4m3fn   — FP8量化后的K
    k_fp8_scale: (num_tokens, num_quant_blocks) fp32 — 每quant block的scale
    block_table: (batch_size, max_blocks_per_seq) int32 — page table
    cu_seqlen: (batch_size + 1,) int32            — cumulative sequence lengths

输出:
    in-place修改k_cache
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("cp_gather_indexer_k_quant_cache")
class CpGatherIndexerKQuantCacheOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "cp_gather_indexer_k_quant_cache"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            num_tokens: token总数
            batch_size: batch大小
            head_dim: K head维度
            block_size: cache block大小
            quant_block_size: 量化block大小
            max_seq_len: 最大序列长度(用于分配block_table)
        """
        num_tokens = params["num_tokens"]
        batch_size = params.get("batch_size", 4)
        head_dim = params.get("head_dim", 576)
        block_size = params.get("block_size", 16)
        quant_block_size = params.get("quant_block_size", 64)
        max_seq_len = params.get("max_seq_len", 2048)

        num_quant_blocks = head_dim // quant_block_size

        # 计算每个block的总字节: data + scale
        # data: block_size * head_dim bytes (fp8)
        # scale: block_size * num_quant_blocks * 4 bytes (fp32)
        bytes_per_block_data = block_size * head_dim
        bytes_per_block_scale = block_size * num_quant_blocks * 4
        total_bytes_per_block = bytes_per_block_data + bytes_per_block_scale

        # 需要的block数 = ceil(num_tokens / block_size)
        max_blocks_per_seq = (max_seq_len + block_size - 1) // block_size
        num_blocks = batch_size * max_blocks_per_seq

        # k_cache: (num_blocks, block_size, ...) 但实际按uint8 flat存储
        # shape需要让view(num_blocks, -1)后切分data和scale
        # k_cache.shape[0] = num_blocks, k_cache.shape[1] = block_size
        # k_cache_flat = k_cache.view(num_blocks, -1)
        # 实际shape: (num_blocks, block_size, head_dim + num_quant_blocks*4)
        # 简化: 用(num_blocks, total_bytes_per_block // block_size) 不对
        # 源码: k_cache.size(1) = block_size => shape (num_blocks, block_size, ...)
        # k_cache.view(num_blocks, -1) => (num_blocks, total_flat)
        # total_flat = block_size * (head_dim + num_quant_blocks * 4)
        # 但data部分是uint8, scale部分需要view为fp32
        # 所以k_cache dtype=uint8, shape=(num_blocks, block_size, head_dim + num_quant_blocks*4)
        k_cache = torch.zeros(
            num_blocks, block_size, head_dim + num_quant_blocks * 4,
            dtype=torch.uint8, device="cuda"
        )

        # k_fp8: (num_tokens, head_dim) fp8
        k_fp8 = torch.randn(
            num_tokens, head_dim,
            dtype=torch.bfloat16, device="cuda"
        ).to(torch.float8_e4m3fn)

        # k_fp8_scale: (num_tokens, num_quant_blocks) fp32
        # 源码推导: quant_block_size = head_dim * 4 // k_fp8_scale.size(1)
        # => k_fp8_scale.size(1) = head_dim * 4 // quant_block_size
        scale_cols = head_dim * 4 // quant_block_size
        k_fp8_scale = torch.ones(
            num_tokens, scale_cols,
            dtype=torch.float32, device="cuda"
        )

        # block_table: (batch_size, max_blocks_per_seq) int32
        block_table = torch.arange(
            num_blocks, dtype=torch.int32, device="cuda"
        ).reshape(batch_size, max_blocks_per_seq)

        # cu_seqlen: (batch_size + 1,) int32
        # 均匀分配tokens到各sequence
        tokens_per_seq = num_tokens // batch_size
        cu_seqlen = torch.zeros(batch_size + 1, dtype=torch.int32, device="cuda")
        for i in range(batch_size):
            cu_seqlen[i + 1] = cu_seqlen[i] + tokens_per_seq
        cu_seqlen[-1] = num_tokens  # 确保总和正确

        return {
            "k_cache": k_cache,
            "k_fp8": k_fp8,
            "k_fp8_scale": k_fp8_scale,
            "block_table": block_table,
            "cu_seqlen": cu_seqlen,
        }

    def compute_flops(self, **params):
        """纯数据搬运+索引，无计算"""
        return 0

    def compute_bytes(self, **params):
        """访存量

        读: k_fp8 + k_fp8_scale + block_table + cu_seqlen
        写: k_cache (data + scale)
        """
        num_tokens = params["num_tokens"]
        batch_size = params.get("batch_size", 4)
        head_dim = params.get("head_dim", 576)
        quant_block_size = params.get("quant_block_size", 64)

        num_quant_blocks = head_dim // quant_block_size
        scale_cols = head_dim * 4 // quant_block_size

        read_bytes = (
            num_tokens * head_dim * 1              # k_fp8 (fp8 = 1 byte)
            + num_tokens * scale_cols * 4          # k_fp8_scale (fp32)
            + batch_size * 128 * 4                 # block_table (估算)
            + (batch_size + 1) * 4                 # cu_seqlen
        )
        write_bytes = (
            num_tokens * head_dim * 1              # cache data
            + num_tokens * num_quant_blocks * 4    # cache scale
        )
        return int(read_bytes + write_bytes)
