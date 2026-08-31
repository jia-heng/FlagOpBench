"""Fused DeepSeek-V4 QNorm + RoPE + KV RoPE + Quant + Insert算子

水平融合DeepSeekV4-MLA的多个操作:
  - 对Q做per-head RMSNorm (no weight) + GPT-J RoPE
  - 对KV做GPT-J RoPE + UE8M0 FP8量化 + paged cache insert

K Cache block layout (block_size=64 tokens):
  - First 64 * 576 = 36864 bytes: Token data
    - Each token: 448 bytes (fp8 nope) + 128 bytes (bf16 rope, 64 * 2B)
  - Next 64 * 8 = 512 bytes: Scales
    - Each token: 8 bytes (uint8 scales, 7 real + 1 padding)

签名:
    fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(
        q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache,
        eps, cache_block_size
    ) -> None  (in-place修改q和k_cache)

输入:
    q: (num_tokens, num_heads, 512) bf16    — Q tensor (in-place修改)
    kv: (num_tokens, 512) bf16              — KV latent (只读)
    k_cache: (num_blocks, block_bytes) uint8 — paged KV cache (in-place写入)
    slot_mapping: (num_tokens_insert,) int64 — slot映射
    position_ids: (num_tokens,) int64       — 位置ID
    cos_sin_cache: (max_pos, 64) fp32       — RoPE cos/sin缓存
    eps: float                              — RMSNorm epsilon
    cache_block_size: int                   — block大小(tokens per block)
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


# 硬编码常量 (与kernel一致)
HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
QUANT_BLOCK = 64
NUM_QUANT_BLOCKS = NOPE_DIM // QUANT_BLOCK  # 7
SCALE_BYTES_PER_TOKEN = NUM_QUANT_BLOCKS + 1  # 8
TOKEN_DATA_BYTES = NOPE_DIM + 2 * ROPE_DIM   # 576


def compute_block_bytes(cache_block_size):
    """计算每个cache block的字节数"""
    data_bytes = cache_block_size * TOKEN_DATA_BYTES
    scale_bytes = cache_block_size * SCALE_BYTES_PER_TOKEN
    total = data_bytes + scale_bytes
    # 对齐到TOKEN_DATA_BYTES的倍数
    aligned = ((total + TOKEN_DATA_BYTES - 1) // TOKEN_DATA_BYTES) * TOKEN_DATA_BYTES
    return aligned


@register_operator("fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert")
class FusedDeepseekV4QnormRopeKvRopeQuantInsertOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            num_tokens: token数
            num_heads: attention head数
            cache_block_size: 每个cache block的token数
            num_blocks: 总block数
            max_pos: 最大位置数 (cos_sin_cache长度)
            eps: RMSNorm epsilon
        """
        num_tokens = params["num_tokens"]
        num_heads = params.get("num_heads", 128)
        cache_block_size = params.get("cache_block_size", 64)
        num_blocks = params.get("num_blocks", 256)
        max_pos = params.get("max_pos", 8192)
        eps = params.get("eps", 1e-6)

        # q: (num_tokens, num_heads, 512) bf16
        q = torch.randn(
            num_tokens, num_heads, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda"
        )

        # kv: (num_tokens, 512) bf16
        kv = torch.randn(
            num_tokens, HEAD_DIM,
            dtype=torch.bfloat16, device="cuda"
        )

        # k_cache: (num_blocks, block_bytes) uint8
        block_bytes = compute_block_bytes(cache_block_size)
        k_cache = torch.zeros(
            num_blocks, block_bytes,
            dtype=torch.uint8, device="cuda"
        )

        # slot_mapping: (num_tokens,) int64 — 每个token映射到一个slot
        # slots范围: [0, num_blocks * cache_block_size)
        total_slots = num_blocks * cache_block_size
        slot_mapping = torch.randint(
            0, total_slots, (num_tokens,),
            dtype=torch.int64, device="cuda"
        )

        # position_ids: (num_tokens,) int64
        position_ids = torch.randint(
            0, max_pos, (num_tokens,),
            dtype=torch.int64, device="cuda"
        )

        # cos_sin_cache: (max_pos, 64) fp32
        cos_sin_cache = torch.randn(
            max_pos, ROPE_DIM,
            dtype=torch.float32, device="cuda"
        )

        return {
            "q": q,
            "kv": kv,
            "k_cache": k_cache,
            "slot_mapping": slot_mapping,
            "position_ids": position_ids,
            "cos_sin_cache": cos_sin_cache,
            "eps": eps,
            "cache_block_size": cache_block_size,
        }

    def compute_flops(self, **params):
        """理论FLOPs

        对每个token:
          Q path (per head):
            - RMSNorm: 512 (sq) + 1 (mean) + 1 (rsqrt) + 512 (mul) = ~1026
            - RoPE (64 dim): 64*4 = 256 (cos*x - sin*y, sin*x + cos*y)
          KV path:
            - RoPE: 64*4 = 256
            - FP8量化(7 blocks of 64): 7*(64*3) = 1344 (abs+div+clamp)

        Total per token: num_heads * (1026+256) + 256 + 1344
        """
        num_tokens = params["num_tokens"]
        num_heads = params.get("num_heads", 128)

        q_flops_per_head = 1026 + 256  # RMSNorm + RoPE
        kv_flops = 256 + 7 * 64 * 3   # RoPE + quantization

        total_per_token = num_heads * q_flops_per_head + kv_flops
        return num_tokens * total_per_token

    def compute_bytes(self, **params):
        """理论访存量

        读:
          q: num_tokens * num_heads * 512 * 2 (bf16)
          kv: num_tokens * 512 * 2 (bf16)
          cos_sin_cache: num_tokens * 64 * 4 (fp32, per unique position)
          slot_mapping: num_tokens * 8 (int64)
          position_ids: num_tokens * 8 (int64)
        写:
          q (in-place): num_tokens * num_heads * 512 * 2 (bf16)
          k_cache token data: num_tokens * 576 (448 fp8 + 128 bf16)
          k_cache scales: num_tokens * 8 (uint8)
        """
        num_tokens = params["num_tokens"]
        num_heads = params.get("num_heads", 128)

        read_bytes = (
            num_tokens * num_heads * HEAD_DIM * 2   # q
            + num_tokens * HEAD_DIM * 2             # kv
            + num_tokens * ROPE_DIM * 4             # cos_sin_cache
            + num_tokens * 8                        # slot_mapping
            + num_tokens * 8                        # position_ids
        )
        write_bytes = (
            num_tokens * num_heads * HEAD_DIM * 2   # q (in-place)
            + num_tokens * TOKEN_DATA_BYTES         # k_cache data
            + num_tokens * SCALE_BYTES_PER_TOKEN    # k_cache scales
        )

        return int(read_bytes + write_bytes)
