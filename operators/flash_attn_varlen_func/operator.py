"""Flash Attention Variable Length Function operator."""
import torch
from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("flash_attn_varlen_func")
class FlashAttnVarlenFuncOperator(BaseOperator):
    """Flash Attention with variable length sequences.

    Standard flash attention for prefill/decode with packed sequences.
    """

    name = "flash_attn_varlen_func"
    library = "flaggems_vllm"

    def prepare_inputs(self, **params) -> dict:
        """Prepare inputs for flash_attn_varlen_func.

        Args:
            **params: Configuration parameters:
                - batch_size: number of sequences
                - num_heads: number of attention heads
                - head_dim: dimension per head
                - seqlen_q: query sequence length per batch
                - seqlen_k: key/value sequence length per batch
                - causal: whether to use causal masking
                - dtype: data type (default: bfloat16)
        """
        batch_size = params.get("batch_size", 1)
        num_heads = params.get("num_heads", 32)
        head_dim = params.get("head_dim", 128)
        seqlen_q = params.get("seqlen_q", 1024)
        seqlen_k = params.get("seqlen_k", 1024)
        causal = params.get("causal", True)
        dtype_str = params.get("dtype", "bfloat16")
        # casegen uses short names like "bf16" / "fp16", map to torch dtype names
        _dtype_map = {"bf16": "bfloat16", "fp16": "float16", "fp32": "float32"}
        dtype = getattr(torch, _dtype_map.get(dtype_str, dtype_str))

        # Total tokens
        total_q = batch_size * seqlen_q
        total_k = batch_size * seqlen_k

        # Create packed Q, K, V
        q = torch.randn(total_q, num_heads, head_dim, dtype=dtype, device="cuda")
        k = torch.randn(total_k, num_heads, head_dim, dtype=dtype, device="cuda")
        v = torch.randn(total_k, num_heads, head_dim, dtype=dtype, device="cuda")

        # Create cumulative sequence lengths
        cu_seqlens_q = torch.arange(0, (batch_size + 1) * seqlen_q, seqlen_q,
                                     dtype=torch.int32, device="cuda")
        cu_seqlens_k = torch.arange(0, (batch_size + 1) * seqlen_k, seqlen_k,
                                     dtype=torch.int32, device="cuda")

        return {
            "q": q,
            "k": k,
            "v": v,
            "cu_seqlens_q": cu_seqlens_q,
            "cu_seqlens_k": cu_seqlens_k,
            "max_seqlen_q": seqlen_q,
            "max_seqlen_k": seqlen_k,
            "causal": causal,
            "softmax_scale": 1.0 / (head_dim ** 0.5),
        }

    def compute_flops(self, **params) -> float:
        """Compute FLOPs for flash attention.

        Flash attention: 2 * batch * seqlen_q * seqlen_k * num_heads * head_dim
        (QK^T matmul + softmax + attention * V)
        """
        batch_size = params.get("batch_size", 1)
        num_heads = params.get("num_heads", 32)
        head_dim = params.get("head_dim", 128)
        seqlen_q = params.get("seqlen_q", 1024)
        seqlen_k = params.get("seqlen_k", 1024)

        # QK^T: batch * num_heads * seqlen_q * seqlen_k * head_dim
        # softmax: batch * num_heads * seqlen_q * seqlen_k (negligible)
        # attn @ V: batch * num_heads * seqlen_q * seqlen_k * head_dim
        flops = 2 * batch_size * num_heads * seqlen_q * seqlen_k * head_dim
        return flops

    def compute_bytes(self, **params) -> float:
        """Compute memory bytes accessed.

        Read: Q, K, V
        Write: output
        """
        batch_size = params.get("batch_size", 1)
        num_heads = params.get("num_heads", 32)
        head_dim = params.get("head_dim", 128)
        seqlen_q = params.get("seqlen_q", 1024)
        seqlen_k = params.get("seqlen_k", 1024)
        dtype = params.get("dtype", "bfloat16")

        bytes_per_elem = 2 if dtype in ["float16", "bfloat16", "fp16", "bf16"] else 4

        total_q = batch_size * seqlen_q
        total_k = batch_size * seqlen_k

        # Read: Q, K, V
        read_bytes = (total_q + 2 * total_k) * num_heads * head_dim * bytes_per_elem
        # Write: output (same shape as Q)
        write_bytes = total_q * num_heads * head_dim * bytes_per_elem

        return read_bytes + write_bytes
