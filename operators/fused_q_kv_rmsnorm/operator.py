"""Fused Q/KV RMSNorm算子

对Q和KV分别做RMSNorm（带weight），在一个kernel中并行处理。
用于DeepSeek-V4 MLA的Q/KV latent归一化。

签名:
    fused_q_kv_rmsnorm(qr, kv, q_weight, kv_weight, eps)
        -> (qr_out, kv_out)

输入:
    qr: (num_tokens, q_size) bf16       — Q latent
    kv: (num_tokens, kv_size) bf16      — KV latent
    q_weight: (q_size,) bf16            — Q RMSNorm权重
    kv_weight: (kv_size,) bf16          — KV RMSNorm权重
    eps: float                          — epsilon

输出:
    qr_out: (num_tokens, q_size) bf16
    kv_out: (num_tokens, kv_size) bf16
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("fused_q_kv_rmsnorm")
class FusedQKvRmsnormOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "fused_q_kv_rmsnorm"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """准备输入

        支持两种参数格式:
          1. 直接: q_size, kv_size
          2. casegen: num_heads, head_dim, q_lora_rank, kv_lora_rank
             -> q_size = q_lora_rank or num_heads * head_dim
             -> kv_size = kv_lora_rank or head_dim (V4-style latent)
        """
        num_tokens = params["num_tokens"]
        eps = params.get("eps", 1e-6)
        dtype_str = params.get("dtype", "bf16")
        dtype = self.get_dtype(dtype_str)

        # Derive q_size / kv_size from casegen params
        if "q_size" in params:
            q_size = params["q_size"]
        elif "q_lora_rank" in params:
            q_size = params["q_lora_rank"]
        else:
            q_size = params.get("num_heads", 128) * params.get("head_dim", 192)

        if "kv_size" in params:
            kv_size = params["kv_size"]
        elif "kv_lora_rank" in params:
            kv_size = params["kv_lora_rank"]
        else:
            # V4-style: no explicit kv_lora_rank, use head_dim as latent dim
            kv_size = params.get("head_dim", 512)

        qr = torch.randn(num_tokens, q_size, dtype=dtype, device="cuda")
        kv = torch.randn(num_tokens, kv_size, dtype=dtype, device="cuda")
        q_weight = torch.ones(q_size, dtype=dtype, device="cuda")
        kv_weight = torch.ones(kv_size, dtype=dtype, device="cuda")

        return {
            "qr": qr,
            "kv": kv,
            "q_weight": q_weight,
            "kv_weight": kv_weight,
            "eps": eps,
        }

    def _resolve_sizes(self, params):
        """Resolve q_size and kv_size from params."""
        if "q_size" in params:
            q_size = params["q_size"]
        elif "q_lora_rank" in params:
            q_size = params["q_lora_rank"]
        else:
            q_size = params.get("num_heads", 128) * params.get("head_dim", 192)

        if "kv_size" in params:
            kv_size = params["kv_size"]
        elif "kv_lora_rank" in params:
            kv_size = params["kv_lora_rank"]
        else:
            kv_size = params.get("head_dim", 512)
        return q_size, kv_size

    def compute_flops(self, **params):
        """理论FLOPs

        RMSNorm per row of size D:
          - x^2: D muls
          - sum: D adds
          - /D + eps + rsqrt: 3 ops
          - x * rrms * w: 2D muls
        ~= 4D per row

        Total: num_tokens * (4*q_size + 4*kv_size)
        """
        num_tokens = params["num_tokens"]
        q_size, kv_size = self._resolve_sizes(params)

        return num_tokens * 4 * (q_size + kv_size)

    def compute_bytes(self, **params):
        """理论访存量

        读:
          qr: num_tokens * q_size * elem_bytes
          kv: num_tokens * kv_size * elem_bytes
          q_weight: q_size * elem_bytes
          kv_weight: kv_size * elem_bytes
        写:
          qr_out: num_tokens * q_size * elem_bytes
          kv_out: num_tokens * kv_size * elem_bytes
        """
        num_tokens = params["num_tokens"]
        q_size, kv_size = self._resolve_sizes(params)
        dtype_str = params.get("dtype", "bf16")
        elem_bytes = self.dtype_bytes(dtype_str)

        read_bytes = (
            num_tokens * q_size * elem_bytes        # qr
            + num_tokens * kv_size * elem_bytes     # kv
            + q_size * elem_bytes                   # q_weight
            + kv_size * elem_bytes                  # kv_weight
        )
        write_bytes = (
            num_tokens * q_size * elem_bytes        # qr_out
            + num_tokens * kv_size * elem_bytes     # kv_out
        )

        return int(read_bytes + write_bytes)
