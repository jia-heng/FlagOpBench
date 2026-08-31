"""chunk_gated_delta_rule (flag_attn / FlagAttention 实现)

GDN (Gated Delta Network) 线性注意力的 chunk-wise prefill 算子。
FlagAttention 版本带有 TLE 优化 + two_kernel_fused_forward 路径。

签名 (head_first=False):
    chunk_gated_delta_rule(
        q, k, v, beta, g,
        BT=64, initial_state=None, output_final_state=True,
        cu_seqlens=None, head_first=False, scale=None,
        use_qk_l2norm_in_kernel=False,
    ) -> (output, final_state)

输入 layout (head_first=False):
    q: [B, T, H_q, K]  bf16
    k: [B, T, H_q, K]  bf16
    v: [B, T, H_v, V]  bf16
    beta: [B, T, H_v]  bf16
    g: [B, T, H_v]     bf16
"""
import torch
import triton

from framework.base_operator import BaseOperator
from framework.registry import register_operator


def _ensure_triton_allocator():
    """确保 Triton TLE 所需的 allocator 已设置"""
    def alloc_fn(size, align, stream):
        return torch.empty(size, dtype=torch.int8, device="cuda")
    triton.set_allocator(alloc_fn)


@register_operator("chunk_gated_delta_rule_flag_attn")
class ChunkGatedDeltaRuleFlagAttnOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "chunk_gated_delta_rule_flag_attn"

    @property
    def library(self) -> str:
        return "flagattention"

    @property
    def impl_name(self) -> str:
        """实际函数名"""
        return "chunk_gated_delta_rule"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            num_tokens: 序列长度 T
            num_heads: Q/K head 数 (H_q)
            num_value_heads: V head 数 (H_v)，默认与 num_heads 相同
            head_dim_k: Q/K head dim (K)
            head_dim_v: V head dim (V)，默认与 head_dim_k 相同
            batch_size: batch 大小，默认 1
            dtype: 数据类型
        """
        _ensure_triton_allocator()
        T = params.get("num_tokens", 2048)
        H_q = params.get("num_heads", 96)
        H_v = params.get("num_value_heads", H_q)
        K = params.get("head_dim_k", 128)
        V = params.get("head_dim_v", K)
        B = params.get("batch_size", 1)
        dtype = self.get_dtype(params.get("dtype", "bfloat16"))

        # 确保 T 是 BT=64 的倍数
        BT = 64
        T = max(BT, (T // BT) * BT)

        q = torch.randn(B, T, H_q, K, dtype=dtype, device="cuda")
        k = torch.randn(B, T, H_q, K, dtype=dtype, device="cuda")
        v = torch.randn(B, T, H_v, V, dtype=dtype, device="cuda")
        # beta 和 g 的 shape 都是 [B, T, H_v] (per value head)
        beta = torch.randn(B, T, H_v, dtype=dtype, device="cuda").sigmoid()
        g = torch.randn(B, T, H_v, dtype=dtype, device="cuda")

        return {
            "q": q,
            "k": k,
            "v": v,
            "beta": beta,
            "g": g,
            "BT": BT,
            "initial_state": None,
            "output_final_state": True,  # TLE 路径要求 output_final_state=True
            "cu_seqlens": None,
            "head_first": False,
            "scale": K ** -0.5,
            "use_qk_l2norm_in_kernel": False,
        }

    def compute_flops(self, **params):
        """理论 FLOPs"""
        T = params.get("num_tokens", 2048)
        H_q = params.get("num_heads", 96)
        H_v = params.get("num_value_heads", H_q)
        K = params.get("head_dim_k", 128)
        V = params.get("head_dim_v", K)
        B = params.get("batch_size", 1)
        BT = 64

        T = max(BT, (T // BT) * BT)
        num_chunks = T // BT
        H = max(H_q, H_v)

        intra_qk = 2 * B * T * H * K * BT
        intra_av = 2 * B * T * H * V * BT
        inter_qs = 2 * B * num_chunks * H * K * V
        state_update = 2 * B * T * H * K * V

        return int(intra_qk + intra_av + inter_qs + state_update)

    def compute_bytes(self, **params):
        """理论访存量"""
        T = params.get("num_tokens", 2048)
        H_q = params.get("num_heads", 96)
        H_v = params.get("num_value_heads", H_q)
        K = params.get("head_dim_k", 128)
        V = params.get("head_dim_v", K)
        B = params.get("batch_size", 1)
        BT = 64
        T = max(BT, (T // BT) * BT)

        elem_bytes = 2
        qk_bytes = 2 * B * T * H_q * K * elem_bytes
        v_bytes = B * T * H_v * V * elem_bytes
        bg_bytes = (B * T * H_v + B * T * H_q) * 4
        out_bytes = B * T * H_v * V * elem_bytes

        return int(qk_bytes + v_bytes + bg_bytes + out_bytes)
