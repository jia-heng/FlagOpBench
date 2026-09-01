"""Top-K Per Row Prefill算子

用于DeepSeek V4 sparse attention的prefill阶段，
从每行logits中选出top_k个索引（基于radix selection）。

签名:
    top_k_per_row_prefill(logits, row_starts, row_ends, indices,
                          num_rows, stride0, stride1, top_k)

输入:
    logits: (num_rows, vocab_size)  — 门控logits (float32)
    row_starts: (num_rows,)         — 每行有效范围起始 (int32, inclusive)
    row_ends: (num_rows,)           — 每行有效范围结束 (int32, exclusive)
    indices: (num_rows, top_k)      — 输出buffer (int32, 预分配)
    num_rows: int                   — 行数
    stride0: int                    — logits.stride(0)
    stride1: int                    — logits.stride(1)
    top_k: int                      — 每行选取的top元素数

输出:
    写入indices (in-place)，值为相对row_starts的0-based索引
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("top_k_per_row_prefill")
class TopKPerRowPrefillOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "top_k_per_row_prefill"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """准备输入

        casegen 路径参数: num_tokens, num_experts, topk
        直接调用参数:     num_rows, vocab_size, top_k

        Args:
            num_rows/num_tokens: 行数（tokens数）
            vocab_size/num_experts: 每行的元素数（专家数）
            top_k/topk: 每行选取的top-k数
        """
        num_rows = params.get("num_tokens") or params["num_rows"]
        vocab_size = params.get("num_experts") or params.get("vocab_size", 256)
        top_k = params.get("topk") or params.get("top_k", 8)

        logits = torch.randn(
            num_rows, vocab_size, dtype=torch.float32, device="cuda"
        )
        row_starts = torch.zeros(num_rows, dtype=torch.int32, device="cuda")
        row_ends = torch.full(
            (num_rows,), vocab_size, dtype=torch.int32, device="cuda"
        )
        indices = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")

        return {
            "logits": logits,
            "row_starts": row_starts,
            "row_ends": row_ends,
            "indices": indices,
            "num_rows": num_rows,
            "stride0": logits.stride(0),
            "stride1": logits.stride(1),
            "top_k": top_k,
        }

    def compute_flops(self, **params):
        """理论FLOPs

        每行需要遍历vocab_size个元素做radix selection:
          - 每个元素: ~2 ops (比较 + histogram更新)
          - radix passes: ~log2(vocab_size) / 8 次（每pass 8 bit）
          - final sort: top_k * log2(top_k)
        近似: num_rows * vocab_size * 4
        """
        num_rows = params.get("num_tokens") or params["num_rows"]
        vocab_size = params.get("num_experts") or params.get("vocab_size", 256)
        top_k = params.get("topk") or params.get("top_k", 8)

        return num_rows * vocab_size * 4

    def compute_bytes(self, **params):
        """理论访存量

        读:
          logits: num_rows * vocab_size * 4 (float32)
          row_starts: num_rows * 4 (int32)
          row_ends: num_rows * 4 (int32)
        写:
          indices: num_rows * top_k * 4 (int32)
        """
        num_rows = params.get("num_tokens") or params["num_rows"]
        vocab_size = params.get("num_experts") or params.get("vocab_size", 256)
        top_k = params.get("topk") or params.get("top_k", 8)

        read_bytes = (
            num_rows * vocab_size * 4   # logits
            + num_rows * 4              # row_starts
            + num_rows * 4              # row_ends
        )
        write_bytes = num_rows * top_k * 4  # indices

        return int(read_bytes + write_bytes)
