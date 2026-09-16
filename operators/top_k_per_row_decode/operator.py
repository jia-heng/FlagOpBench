"""Top-K Per Row Decode算子

用于DeepSeek V4的decode阶段，从每行logits中选出top_k个索引。
Decode阶段每次只处理1行(num_rows=1)，使用radix-based selection。

签名:
    top_k_per_row_decode(logits, next_n, seq_lens, indices,
                         num_rows, stride0, stride1, top_k)

输入:
    logits: (num_rows, vocab_size) float32  — 门控logits
    next_n: int                            — next token数(API兼容用，decode阶段通常=1)
    seq_lens: (B,) int32                   — 有效范围 [0, seq_lens[0])
    indices: (num_rows, top_k) int32       — 输出buffer (预分配)
    num_rows: int                          — 行数(decode时=1)
    stride0: int                           — logits.stride(0)
    stride1: int                           — logits.stride(1)
    top_k: int                             — 每行选取的top元素数

输出:
    写入indices (in-place)
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("top_k_per_row_decode")
class TopKPerRowDecodeOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "top_k_per_row_decode"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """准备输入

        casegen 路径参数: num_tokens, num_experts, topk
        直接调用参数:     num_rows, vocab_size, top_k

        Args:
            num_rows/num_tokens: 行数(decode场景通常为batch_size，每行一个token)
            vocab_size/num_experts: 每行的元素数（专家数）
            top_k/topk: 每行选取的top-k数
        """
        num_rows = params.get("num_tokens") or params["num_rows"]
        vocab_size = params.get("num_experts") or params.get("vocab_size", 256)
        top_k = params.get("topk") or params.get("top_k", 8)

        logits = torch.randn(
            num_rows, vocab_size, dtype=torch.float32, device="cuda"
        )

        next_n = 1  # decode阶段

        # seq_lens: (B,) — 有效元素范围
        seq_lens = torch.full(
            (num_rows,), vocab_size, dtype=torch.int32, device="cuda"
        )

        indices = torch.empty(num_rows, top_k, dtype=torch.int32, device="cuda")

        return {
            "logits": logits,
            "next_n": next_n,
            "seq_lens": seq_lens,
            "indices": indices,
            "num_rows": num_rows,
            "stride0": logits.stride(0),
            "stride1": logits.stride(1),
            "top_k": top_k,
        }

    def compute_flops(self, **params):
        """理论FLOPs

        Radix selection: 每行遍历vocab_size个元素
        近似: num_rows * vocab_size * 4
        """
        num_rows = params.get("num_tokens") or params["num_rows"]
        vocab_size = params.get("num_experts") or params.get("vocab_size", 256)
        return num_rows * vocab_size * 4

    def compute_bytes(self, **params):
        """理论访存量

        读:
          logits: num_rows * vocab_size * 4 (float32)
          seq_lens: num_rows * 4 (int32)
        写:
          indices: num_rows * top_k * 4 (int32)
        """
        num_rows = params.get("num_tokens") or params["num_rows"]
        vocab_size = params.get("num_experts") or params.get("vocab_size", 256)
        top_k = params.get("topk") or params.get("top_k", 8)

        read_bytes = (
            num_rows * vocab_size * 4   # logits
            + num_rows * 4              # seq_lens
        )
        write_bytes = num_rows * top_k * 4  # indices

        return int(read_bytes + write_bytes)
