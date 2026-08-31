"""FP8/FP4 Paged MQA Logits算子

计算FP8 query对FP8/FP4 paged KV cache的MQA logits。
用于DeepSeek V4 sparse attention的speculative decoding阶段。

签名:
    fp8_fp4_paged_mqa_logits(q, kv_cache, weights, context_lens,
                              block_tables, schedule_metadata,
                              max_model_len, clean_logits=False)
        -> logits (total_rows, max_model_len)

输入:
    q: tuple(q_values, q_scale)
        q_values: (B, next_n, H, D) float8_e4m3fn
        q_scale: scalar float32
    kv_cache: (num_blocks, block_size, 1, D+4) uint8 — paged KV cache
    weights: (B*next_n, H) float32 — per-head权重
    context_lens: (B,) int32 — 每个序列的context长度
    block_tables: (B, max_blocks) int32 — page table
    schedule_metadata: None（API兼容，Triton kernel不使用）
    max_model_len: int — 最大模型序列长度
    clean_logits: bool — 是否用-inf初始化输出

输出:
    logits: (B*next_n, max_model_len) float32
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("fp8_fp4_paged_mqa_logits")
class FP8FP4PagedMQALogitsOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "fp8_fp4_paged_mqa_logits"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            B: batch size
            next_n: speculative decode步数（通常1）
            H: head数
            D: head dim
            block_size: KV cache page大小
            max_model_len: 最大模型序列长度
            context_len: 实际context长度
            clean_logits: 是否用-inf初始化
        """
        B = params["B"]
        next_n = params.get("next_n", 1)
        H = params.get("H", 128)
        D = params.get("D", 128)
        block_size = params.get("block_size", 16)
        max_model_len = params.get("max_model_len", 4096)
        context_len = params.get("context_len", 2048)
        clean_logits = params.get("clean_logits", False)

        max_blocks_per_seq = (max_model_len + block_size - 1) // block_size
        # 分配足够的物理blocks
        num_phys_blocks = B * max_blocks_per_seq

        # q: fp8
        q_values = torch.randn(
            B, next_n, H, D, dtype=torch.bfloat16, device="cuda"
        ).to(torch.float8_e4m3fn)
        q_scale = torch.tensor(1.0, dtype=torch.float32, device="cuda")

        # kv_cache: [num_phys_blocks, block_size, 1, D+4] uint8
        kv_cache = torch.randint(
            0, 255, (num_phys_blocks, block_size, 1, D + 4),
            dtype=torch.uint8, device="cuda"
        )

        # weights: [B*next_n, H] float32
        weights = torch.ones(B * next_n, H, dtype=torch.float32, device="cuda")

        # context_lens: [B] int32
        context_lens = torch.full(
            (B,), context_len, dtype=torch.int32, device="cuda"
        )

        # block_tables: [B, max_blocks_per_seq] int32
        block_tables = torch.arange(
            num_phys_blocks, dtype=torch.int32, device="cuda"
        ).view(B, max_blocks_per_seq)

        return {
            "q": (q_values, q_scale),
            "kv_cache": kv_cache,
            "weights": weights,
            "context_lens": context_lens,
            "block_tables": block_tables,
            "schedule_metadata": None,
            "max_model_len": max_model_len,
            "clean_logits": clean_logits,
        }

    def compute_flops(self, **params):
        """理论FLOPs

        MQA logits: 对每个token位置做 Q @ K^T（点积）
          - 每个(row, kv_pos): H * D * 2 (mul + add)
          - 加权求和: H (per head weight)
        总: B * next_n * context_len * H * D * 2 + B * next_n * context_len * H
        """
        B = params["B"]
        next_n = params.get("next_n", 1)
        H = params.get("H", 128)
        D = params.get("D", 128)
        context_len = params.get("context_len", 2048)

        total_rows = B * next_n
        # Q @ K^T dot product per position
        dot_flops = total_rows * context_len * H * D * 2
        # weighted sum across heads
        weight_flops = total_rows * context_len * H

        return int(dot_flops + weight_flops)

    def compute_bytes(self, **params):
        """理论访存量

        读:
          q: B * next_n * H * D * 1 (fp8 = 1 byte)
          kv_cache (有效部分): B * context_len * 1 * (D+4) * 1 (uint8)
          weights: B * next_n * H * 4 (float32)
          context_lens: B * 4
          block_tables: B * max_blocks * 4
        写:
          logits: B * next_n * max_model_len * 4 (float32)
        """
        B = params["B"]
        next_n = params.get("next_n", 1)
        H = params.get("H", 128)
        D = params.get("D", 128)
        block_size = params.get("block_size", 16)
        max_model_len = params.get("max_model_len", 4096)
        context_len = params.get("context_len", 2048)

        total_rows = B * next_n
        max_blocks_per_seq = (max_model_len + block_size - 1) // block_size

        read_bytes = (
            total_rows * H * D * 1                  # q (fp8)
            + B * context_len * (D + 4)             # kv_cache effective
            + total_rows * H * 4                    # weights
            + B * 4                                 # context_lens
            + B * max_blocks_per_seq * 4            # block_tables
        )
        write_bytes = total_rows * max_model_len * 4  # logits (float32)

        return int(read_bytes + write_bytes)
