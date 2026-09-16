"""Compute Global TopK Indices and Lens算子

用于DeepSeek V4 sparse attention prefill/decode阶段。
将local topk索引通过block_table映射到global物理slot索引，
并计算每个token的有效索引数。

签名:
    compute_global_topk_indices_and_lens(
        topk_indices, token_to_req_indices, block_table,
        block_size, is_valid_token=None
    ) -> (global_indices, lens)

输入:
    topk_indices: (num_tokens, topk)        — 局部top-k KV块索引 (int32)
    token_to_req_indices: (num_tokens,)     — 每个token对应的request索引 (int32)
    block_table: (num_reqs, max_blocks)     — block table映射 (int32)
    block_size: int                         — 每个block的slot数
    is_valid_token: (num_tokens,) optional  — token有效性掩码 (int32), 默认全1

输出:
    global_indices: (num_tokens, topk) int32 — 全局物理slot索引
    lens: (num_tokens,) int32               — 每个token的有效索引数
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("compute_global_topk_indices_and_lens")
class ComputeGlobalTopkIndicesAndLensOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "compute_global_topk_indices_and_lens"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            batch_size / num_reqs: request数
            seqlen_q / seq_len: 每个request的序列长度
            index_topk / topk: top-k数
            block_size: block大小
            max_blocks: 每个request最大block数
            valid_ratio: 有效索引比例 (0~1)
        """
        num_reqs = params.get("batch_size") or params.get("num_reqs") or 1
        seq_len = params.get("seqlen_q") or params.get("seq_len") or 512
        num_tokens = num_reqs * seq_len
        topk = params.get("index_topk") or params.get("topk", 64)
        block_size = params.get("block_size", 16)
        max_blocks = params.get("max_blocks", 128)
        valid_ratio = params.get("valid_ratio", 0.9)

        # token_to_req_indices: 每个token属于哪个request
        tokens_per_req = num_tokens // num_reqs
        token_to_req_indices = torch.arange(
            num_reqs, dtype=torch.int32, device="cuda"
        ).repeat_interleave(tokens_per_req)
        # 处理余数
        if token_to_req_indices.shape[0] < num_tokens:
            pad = torch.full(
                (num_tokens - token_to_req_indices.shape[0],),
                num_reqs - 1, dtype=torch.int32, device="cuda"
            )
            token_to_req_indices = torch.cat([token_to_req_indices, pad])

        # block_table: (num_reqs, max_blocks)  物理block编号
        total_blocks = num_reqs * max_blocks
        block_table = torch.arange(
            total_blocks, dtype=torch.int32, device="cuda"
        ).reshape(num_reqs, max_blocks)

        # topk_indices: 局部索引，范围 [0, max_blocks * block_size)
        max_local_idx = max_blocks * block_size
        # 一部分为有效索引，一部分为-1（无效）
        num_valid = int(topk * valid_ratio)
        valid_part = torch.randint(
            0, max_local_idx, (num_tokens, num_valid),
            dtype=torch.int32, device="cuda"
        )
        invalid_part = torch.full(
            (num_tokens, topk - num_valid), -1,
            dtype=torch.int32, device="cuda"
        )
        topk_indices = torch.cat([valid_part, invalid_part], dim=1)

        # is_valid_token: 全1
        is_valid_token = torch.ones(
            num_tokens, dtype=torch.int32, device="cuda"
        )

        return {
            "topk_indices": topk_indices,
            "token_to_req_indices": token_to_req_indices,
            "block_table": block_table,
            "block_size": block_size,
            "is_valid_token": is_valid_token,
        }

    def compute_flops(self, **params):
        """理论FLOPs

        每个token对topk个索引:
          - 整除/取模计算block_idx和block_off: 2 ops
          - 查block_table: 1 load
          - 计算slot: 1 mul + 1 add
        近似: num_tokens * topk * 5
        """
        num_reqs = params.get("batch_size") or params.get("num_reqs") or 1
        seq_len = params.get("seqlen_q") or params.get("seq_len") or 512
        num_tokens = num_reqs * seq_len
        topk = params.get("index_topk") or params.get("topk", 64)
        return num_tokens * topk * 5

    def compute_bytes(self, **params):
        """理论访存量

        读:
          topk_indices: num_tokens * topk * 4 (int32)
          token_to_req_indices: num_tokens * 4
          block_table: num_reqs * max_blocks * 4
          is_valid_token: num_tokens * 4
        写:
          global_indices: num_tokens * topk * 4 (int32)
          lens: num_tokens * 4
        """
        num_reqs = params.get("batch_size") or params.get("num_reqs") or 1
        seq_len = params.get("seqlen_q") or params.get("seq_len") or 512
        num_tokens = num_reqs * seq_len
        topk = params.get("index_topk") or params.get("topk", 64)
        max_blocks = params.get("max_blocks", 128)

        read_bytes = (
            num_tokens * topk * 4           # topk_indices
            + num_tokens * 4                # token_to_req_indices
            + num_reqs * max_blocks * 4     # block_table
            + num_tokens * 4                # is_valid_token
        )
        write_bytes = (
            num_tokens * topk * 4           # global_indices
            + num_tokens * 4                # lens
        )

        return int(read_bytes + write_bytes)
