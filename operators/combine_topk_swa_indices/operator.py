"""Combine TopK SWA Indices算子

用于DeepSeek V4 sparse attention prefill阶段。
将topk索引与sliding window attention索引合并，生成combined indices。

签名:
    combine_topk_swa_indices(topk_indices, query_start_loc, seq_lens,
                              gather_lens, window_size, compress_ratio,
                              topk, M, N)
        -> (combined, lens)

输入:
    topk_indices: (total_tokens, topk)  — top-k压缩KV索引 (int32)
    query_start_loc: (num_reqs,)        — 每个request在total_tokens中的起始位置 (int32)
    seq_lens: (num_reqs,)               — 每个request的序列长度 (int32)
    gather_lens: (num_reqs,)            — 压缩后的context长度 (int32)
    window_size: int                    — sliding window大小
    compress_ratio: int                 — KV压缩比
    topk: int                           — top-k数
    M: int                              — total_tokens
    N: int                              — 压缩后的序列长度

输出:
    combined: (total_tokens, combined_topk) int32 — 合并后的索引
    lens: (total_tokens,) int32                  — 每个token的有效索引数
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("combine_topk_swa_indices")
class CombineTopkSwaIndicesOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "combine_topk_swa_indices"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            batch_size / num_reqs: request数
            seqlen_q / seq_len: 每个request的序列长度
            index_topk / topk: top-k数
            sliding_window / window_size: sliding window大小
            compress_ratio: KV压缩比
        """
        num_reqs = params.get("batch_size") or params.get("num_reqs") or 1
        seq_len = params.get("seqlen_q") or params.get("seq_len") or 512
        topk = params.get("index_topk") or params.get("topk", 64)
        window_size = params.get("sliding_window") or params.get("window_size", 128)
        compress_ratio = params.get("compress_ratio", 4)

        total_tokens = num_reqs * seq_len
        compressed_len = seq_len // compress_ratio

        # query_start_loc: cumulative start positions
        query_start_loc = torch.arange(
            0, total_tokens, seq_len, dtype=torch.int32, device="cuda"
        )

        seq_lens = torch.full(
            (num_reqs,), seq_len, dtype=torch.int32, device="cuda"
        )
        gather_lens = torch.full(
            (num_reqs,), compressed_len, dtype=torch.int32, device="cuda"
        )

        # topk_indices: random indices into compressed KV
        # Ensure topk_indices can fit: use max(compressed_len, topk)
        max_index = max(compressed_len, topk)
        topk_indices = torch.randint(
            0, max_index, (total_tokens, topk),
            dtype=torch.int32, device="cuda"
        )

        M = total_tokens
        N = max_index

        return {
            "topk_indices": topk_indices,
            "query_start_loc": query_start_loc,
            "seq_lens": seq_lens,
            "gather_lens": gather_lens,
            "window_size": window_size,
            "compress_ratio": compress_ratio,
            "topk": topk,
            "M": M,
            "N": N,
        }

    def compute_flops(self, **params):
        """理论FLOPs

        每个token:
          - 合并topk索引: topk ops（复制+去重）
          - 计算window索引: window_size ops
          - 排序/padding: ~combined_topk * log2(combined_topk)
        近似: total_tokens * (topk + window_size) * 2
        """
        num_reqs = params.get("batch_size") or params.get("num_reqs") or 1
        seq_len = params.get("seqlen_q") or params.get("seq_len") or 512
        topk = params.get("index_topk") or params.get("topk", 64)
        window_size = params.get("sliding_window") or params.get("window_size", 128)

        total_tokens = num_reqs * seq_len
        return total_tokens * (topk + window_size) * 2

    def compute_bytes(self, **params):
        """理论访存量

        读:
          topk_indices: total_tokens * topk * 4 (int32)
          query_start_loc: num_reqs * 4
          seq_lens: num_reqs * 4
          gather_lens: num_reqs * 4
        写:
          combined: total_tokens * combined_topk * 4 (int32)
          lens: total_tokens * 4 (int32)
        """
        num_reqs = params.get("batch_size") or params.get("num_reqs") or 1
        seq_len = params.get("seqlen_q") or params.get("seq_len") or 512
        topk = params.get("index_topk") or params.get("topk", 64)
        window_size = params.get("sliding_window") or params.get("window_size", 128)

        total_tokens = num_reqs * seq_len
        # combined_topk is aligned up
        _ALIGNMENT = 128
        combined_topk = ((topk + window_size + _ALIGNMENT - 1) // _ALIGNMENT) * _ALIGNMENT

        read_bytes = (
            total_tokens * topk * 4     # topk_indices
            + num_reqs * 4              # query_start_loc
            + num_reqs * 4              # seq_lens
            + num_reqs * 4              # gather_lens
        )
        write_bytes = (
            total_tokens * combined_topk * 4    # combined
            + total_tokens * 4                  # lens
        )

        return int(read_bytes + write_bytes)
