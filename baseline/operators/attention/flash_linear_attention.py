"""Flash Linear Attention 算子 (Gated Delta Rule)

对应算子: flash_linear_attention

Flash Linear Attention 来自 flash-linear-attention 项目 (Songlin Yang, Yu Zhang)，
实现 Gated Delta Rule 线性注意力机制，用于 GLA, HGRN2, Falcon-H1 等模型。

核心递推公式 (Delta Rule with gating):
    S_t = exp(g_t) * S_{t-1} + beta_t * (v_t - S_{t-1} @ k_t) ⊗ k_t
    o_t = q_t @ S_t

其中:
- S_t: (head_dim_v, head_dim_k) 隐状态矩阵
- g_t: scalar gating (log-space)，控制遗忘
- beta_t: scalar learning rate，控制写入强度
- Delta Rule: v_t - S_{t-1} @ k_t 是预测误差

可用实现:
1. PyTorch token-by-token recurrence (golden reference)
2. vLLM fused_recurrent Triton kernel (decode 最优)
3. vLLM chunk_gated_delta_rule Triton kernel (prefill 最优)

性能对比:
- PyTorch recurrence: O(T*d^2)，逐 token 循环，无并行，仅用于验证
- Triton fused_recurrent: 融合 kernel，减少 HBM 读写，decode 场景 10-100x
- Triton chunk: 分块并行 + state 递推，prefill 场景 5-50x

基线选择策略:
  forward 使用 vLLM Triton kernel（实际推理路径）。
  golden reference 使用 PyTorch recurrence（数值精确）。

参考代码:
  vllm/third_party/flash_linear_attention/ops/fused_recurrent.py
  vllm/third_party/flash_linear_attention/ops/chunk.py
"""

import torch
import torch.nn.functional as F
import math
from baseline.operators.registry import BaseOperator, register_operator

# 尝试导入 vLLM flash_linear_attention Triton kernels
try:
    from vllm.third_party.flash_linear_attention.ops.fused_recurrent import (
        fused_recurrent_gated_delta_rule
    )
    HAS_VLLM_FLA_RECURRENT = True
except ImportError:
    HAS_VLLM_FLA_RECURRENT = False

try:
    from vllm.third_party.flash_linear_attention.ops import chunk_gated_delta_rule
    HAS_VLLM_FLA_CHUNK = True
except ImportError:
    HAS_VLLM_FLA_CHUNK = False


def _gated_delta_rule_recurrence(q, k, v, g, beta, scale, initial_state=None):
    """Token-by-token Gated Delta Rule recurrence (reference).

    Matches the semantics of vLLM's fused_recurrent_gated_delta_rule_fwd_kernel:
        for each token t:
            S = exp(g_t) * S
            delta = v_t - S @ k_t  (prediction error)
            S += beta_t * delta ⊗ k_t  (state update)
            o_t = S @ q_t  (query output)

    Args:
        q: (B, T, H, K) queries
        k: (B, T, H, K) keys (typically L2-normalized)
        v: (B, T, H, V) values
        g: (B, T, H) gating in log-space
        beta: (B, T, H) learning rate (sigmoid-range)
        scale: float, query scaling
        initial_state: (B, H, V, K) or None

    Returns:
        output: (B, T, H, V)
        final_state: (B, H, V, K)
    """
    B, T, H, K = q.shape
    V = v.shape[-1]

    q = q * scale

    # State: (B, H, V, K) - matches vLLM layout
    if initial_state is not None:
        state = initial_state.clone().float()
    else:
        state = torch.zeros(B, H, V, K, device=q.device, dtype=torch.float32)

    output = torch.zeros(B, T, H, V, device=q.device, dtype=q.dtype)

    for t in range(T):
        # Gating: decay state
        gt = g[:, t, :].float()  # (B, H)
        decay = torch.exp(gt).unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
        state = state * decay

        # Delta rule update
        kt = k[:, t, :, :].float()  # (B, H, K)
        vt = v[:, t, :, :].float()  # (B, H, V)
        bt = beta[:, t, :].float()  # (B, H)

        # Prediction error: delta = v_t - S @ k_t
        # state: (B, H, V, K), kt: (B, H, K) -> pred: (B, H, V)
        pred = torch.einsum('bhvk,bhk->bhv', state, kt)
        delta = vt - pred  # (B, H, V)

        # State update: S += beta * delta ⊗ k
        # (B, H, V) x (B, H, K) -> (B, H, V, K)
        state = state + bt.unsqueeze(-1).unsqueeze(-1) * delta.unsqueeze(-1) * kt.unsqueeze(-2)

        # Output: o_t = S @ q_t
        qt = q[:, t, :, :].float()  # (B, H, K)
        out_t = torch.einsum('bhvk,bhk->bhv', state, qt)  # (B, H, V)
        output[:, t] = out_t.to(q.dtype)

    return output, state


@register_operator("flash_linear_attention")
class FlashLinearAttentionOperator(BaseOperator):
    """Flash Linear Attention: Gated Delta Rule

    实现 vLLM 中 flash_linear_attention 的 Gated Delta Rule 递推。
    参考: vllm/third_party/flash_linear_attention/ops/fused_recurrent.py
         vllm/third_party/flash_linear_attention/ops/chunk.py

    Input:
        q: (batch, seq_len, num_heads, head_dim) - queries
        k: (batch, seq_len, num_heads, head_dim) - keys (L2-normalized)
        v: (batch, seq_len, num_heads, head_dim) - values
        g: (batch, seq_len, num_heads) - gating (log-space decay)
        beta: (batch, seq_len, num_heads) - learning rate
    Output:
        (batch, seq_len, num_heads, head_dim)
    """

    @property
    def name(self) -> str:
        return "flash_linear_attention"

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                g: torch.Tensor, beta: torch.Tensor, scale: float = None,
                initial_state: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """Flash Linear Attention 前向 - 使用 vLLM Triton kernel

        实现选择:
        1. vLLM fused_recurrent (decode, 短序列) ⭐ 默认
        2. vLLM chunk_gated_delta_rule (长序列 prefill)
        3. PyTorch recurrence (fallback)

        基线选择: vLLM Triton kernel（实际推理路径）
        """
        return self.forward_vllm(q, k, v, g, beta, scale, initial_state)

    def forward_vllm(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     g: torch.Tensor, beta: torch.Tensor, scale: float = None,
                     initial_state: torch.Tensor = None, **kwargs) -> torch.Tensor:
        """使用 vLLM flash_linear_attention Triton kernel

        - 短序列 / decode: fused_recurrent_gated_delta_rule
        - 长序列 / prefill: chunk_gated_delta_rule

        参考:
          vllm/third_party/flash_linear_attention/ops/fused_recurrent.py
          vllm/third_party/flash_linear_attention/ops/chunk.py
        """
        B, T, H, K = q.shape
        V = v.shape[-1]

        if scale is None:
            scale = K ** -0.5

        # 初始化 state
        if initial_state is None:
            initial_state = torch.zeros(B, H, V, K, device=q.device, dtype=q.dtype)

        # 选择实现路径:
        # - fused_recurrent: 适合 decode/短序列, 但 head_dim=128 + batch>4 可能触发 Triton kernel 限制
        # - chunk: 适合 prefill/长序列, 无 head_dim 限制
        use_chunk = HAS_VLLM_FLA_CHUNK and (T > 64 or (K >= 128 and B > 4))

        if use_chunk:
            # chunk_gated_delta_rule: prefill 优化路径
            # API: q(B,T,H,K), k(B,T,H,K), v(B,T,H,V), g(B,T,H), beta(B,T,H)
            #      initial_state(B,H,V,K)
            o, _ = chunk_gated_delta_rule(
                q=q.contiguous(),
                k=k.contiguous(),
                v=v.contiguous(),
                g=g.contiguous(),
                beta=beta.contiguous(),
                scale=scale,
                initial_state=initial_state.contiguous(),
                output_final_state=False,
            )
            return o
        elif HAS_VLLM_FLA_RECURRENT:
            # fused_recurrent_gated_delta_rule: decode / 短序列优化路径
            # API: q(B,T,H,K), k(B,T,H,K), v(B,T,HV,V), g(B,T,HV), beta(B,T,HV)
            #      initial_state(B,HV,V,K)
            o, _ = fused_recurrent_gated_delta_rule(
                q=q.contiguous(),
                k=k.contiguous(),
                v=v.contiguous(),
                g=g.contiguous(),
                beta=beta.contiguous(),
                scale=scale,
                initial_state=initial_state.contiguous(),
                inplace_final_state=False,
            )
            return o
        else:
            # Fallback: PyTorch recurrence
            output, _ = _gated_delta_rule_recurrence(
                q, k, v, g, beta, scale, initial_state
            )
            return output

    def compute_flops(self, batch: int, seq_len: int, num_heads: int,
                      head_dim: int, **kwargs) -> int:
        """FLOPs for Gated Delta Rule recurrence

        Per token:
        - State decay: H * V * K (element-wise multiply)
        - Prediction (S @ k): H * V * K (matvec)
        - Delta: H * V (subtract)
        - State update (outer product): H * V * K
        - Output (S @ q): H * V * K (matvec)
        Total per token: ~4 * H * V * K
        Total: batch * seq_len * 4 * num_heads * head_dim^2
        """
        return batch * seq_len * 4 * num_heads * head_dim * head_dim

    def compute_bytes(self, batch: int, seq_len: int, num_heads: int,
                      head_dim: int, dtype: str = "bf16", **kwargs) -> int:
        """访存 = 读 Q/K/V/G/Beta + State R/W + 写 output"""
        elem_bytes = self.dtype_bytes(dtype)
        # Read Q, K, V: (B, T, H, D) each
        read_qkv = 3 * batch * seq_len * num_heads * head_dim * elem_bytes
        # Read G, Beta: (B, T, H) each
        read_g_beta = 2 * batch * seq_len * num_heads * elem_bytes
        # Write output: (B, T, H, D)
        write_out = batch * seq_len * num_heads * head_dim * elem_bytes
        # State: (B, H, D, D) - read+write per token (fp32)
        state_rw = 2 * batch * num_heads * head_dim * head_dim * 4 * seq_len
        return read_qkv + read_g_beta + write_out + state_rw

    def prepare_inputs(self, batch: int, seq_len: int, num_heads: int,
                       head_dim: int, dtype: str = "bf16", **kwargs) -> dict:
        torch_dtype = self.get_dtype(dtype)
        q = torch.randn(batch, seq_len, num_heads, head_dim,
                        device=self.device, dtype=torch_dtype)
        k = torch.randn(batch, seq_len, num_heads, head_dim,
                        device=self.device, dtype=torch_dtype)
        # L2-normalize K (standard practice, matches vLLM usage)
        k = F.normalize(k, p=2, dim=-1)
        v = torch.randn(batch, seq_len, num_heads, head_dim,
                        device=self.device, dtype=torch_dtype)
        # Gating in log-space (log-sigmoid ensures g < 0 -> decay < 1)
        g = F.logsigmoid(torch.randn(batch, seq_len, num_heads,
                                     device=self.device, dtype=torch_dtype))
        # Beta: learning rate in (0, 1), via sigmoid
        beta = torch.sigmoid(torch.randn(batch, seq_len, num_heads,
                                         device=self.device, dtype=torch_dtype))
        return {
            "q": q, "k": k, "v": v, "g": g, "beta": beta,
        }

    def compute_golden(self, q: torch.Tensor, k: torch.Tensor,
                       v: torch.Tensor, g: torch.Tensor,
                       beta: torch.Tensor, scale: float = None,
                       initial_state: torch.Tensor = None,
                       **kwargs) -> torch.Tensor:
        """Golden reference: FP32 CPU token-by-token recurrence"""
        if scale is None:
            scale = q.shape[-1] ** -0.5
        output, _ = _gated_delta_rule_recurrence(
            q.float().cpu(), k.float().cpu(), v.float().cpu(),
            g.float().cpu(), beta.float().cpu(), scale,
            initial_state.float().cpu() if initial_state is not None else None,
        )
        return output.to(q.dtype).to(q.device)
