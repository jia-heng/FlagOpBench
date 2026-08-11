"""KV RMSNorm + RoPE + Cache 融合算子

对应算子列表: kv_rms_norm_rope_cache

融合操作: RMSNorm(K,V) → RoPE(K) → Cache Write
应用场景: LLM 推理中 KV 的预处理流水线

可用实现:
1. PyTorch 分步实现 (组合 rms_norm + rope + cat)
2. vLLM fused CUDA kernel (未集成)

性能对比 (待实测后更新):
- PyTorch 分步: 多次 kernel launch，中间 tensor 开销
- vLLM fused kernel: 单 kernel 完成 norm+rope+write，显存高效

基线选择策略:
  当前使用 PyTorch 分步实现作为基线（组合已实现算子）。
  fused 版本减少 2 次中间 tensor 分配 + 2 次 kernel launch。
  集成后需实测对比，预期 decode 场景 (小 seq) 提升 20-40%。
"""

import torch
import torch.nn.functional as F
from baseline.operators.registry import BaseOperator, register_operator


@register_operator("kv_rms_norm_rope_cache")
class KVRMSNormRopeCacheOperator(BaseOperator):
    """KV RMSNorm + RoPE + Cache: 融合 KV 预处理流水线

    Input:
        k: (num_tokens, num_kv_heads, head_dim) - key
        v: (num_tokens, num_kv_heads, head_dim) - value
        norm_weight: (head_dim,) - RMSNorm weight
        cos: (num_tokens, 1, head_dim) - RoPE cos
        sin: (num_tokens, 1, head_dim) - RoPE sin
        k_cache: (cache_len, num_kv_heads, head_dim) - existing key cache
        v_cache: (cache_len, num_kv_heads, head_dim) - existing value cache
    Output:
        k_cached: (cache_len + num_tokens, num_kv_heads, head_dim)
        v_cached: (cache_len + num_tokens, num_kv_heads, head_dim)
    """

    @property
    def name(self) -> str:
        return "kv_rms_norm_rope_cache"

    def _rms_norm(self, x: torch.Tensor, weight: torch.Tensor,
                  eps: float = 1e-6) -> torch.Tensor:
        """RMSNorm: x / sqrt(mean(x^2) + eps) * weight"""
        if hasattr(F, 'rms_norm'):
            return F.rms_norm(x, [x.shape[-1]], weight, eps)
        input_dtype = x.dtype
        x_float = x.float()
        variance = x_float.pow(2).mean(-1, keepdim=True)
        x_normed = x_float * torch.rsqrt(variance + eps)
        return (x_normed * weight.float()).to(input_dtype)

    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        x1, x2 = x.chunk(2, dim=-1)
        return torch.cat([-x2, x1], dim=-1)

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor,
                    sin: torch.Tensor) -> torch.Tensor:
        """Apply RoPE: x * cos + rotate_half(x) * sin"""
        return x * cos + self._rotate_half(x) * sin

    def forward(self, k: torch.Tensor, v: torch.Tensor,
                norm_weight: torch.Tensor, cos: torch.Tensor,
                sin: torch.Tensor, k_cache: torch.Tensor,
                v_cache: torch.Tensor, eps: float = 1e-6,
                **kwargs) -> tuple:
        """KV RMSNorm + RoPE + Cache 前向

        可用实现:
        1. PyTorch 分步 (norm + rope + cat) ⭐ 当前基线
        2. vLLM fused kernel (未集成)

        基线选择: PyTorch 分步（组合已有算子，通用性好）
        TODO: 集成 fused kernel 后对比，decode 场景预期提升 20-40%
        """
        # Step 1: RMSNorm on K and V
        k_normed = self._rms_norm(k, norm_weight, eps)
        v_normed = self._rms_norm(v, norm_weight, eps)

        # Step 2: Apply RoPE to K only (V 不需要位置编码)
        k_roped = self._apply_rope(k_normed, cos, sin)

        # Step 3: Append to cache
        k_cached = torch.cat([k_cache, k_roped], dim=0)
        v_cached = torch.cat([v_cache, v_normed], dim=0)

        return k_cached, v_cached

    def compute_flops(self, num_tokens: int, num_kv_heads: int,
                      head_dim: int, cache_len: int = 0, **kwargs) -> int:
        """FLOPs = RMSNorm(K) + RMSNorm(V) + RoPE(K)"""
        elements = num_tokens * num_kv_heads * head_dim
        rmsnorm_flops = 2 * 4 * elements  # 2x RMSNorm, each ~4 ops/elem
        rope_flops = 4 * elements  # RoPE ~4 ops/elem
        return rmsnorm_flops + rope_flops

    def compute_bytes(self, num_tokens: int, num_kv_heads: int,
                      head_dim: int, cache_len: int = 0,
                      dtype: str = "bf16", **kwargs) -> int:
        """访存 = 读 k,v,weight,cos,sin + 读 cache + 写新 cache"""
        elem_bytes = self.dtype_bytes(dtype)
        kv_elements = num_tokens * num_kv_heads * head_dim
        read_kv = 2 * kv_elements * elem_bytes
        read_weight = head_dim * elem_bytes
        read_cos_sin = 2 * num_tokens * 1 * head_dim * elem_bytes
        read_cache = 2 * cache_len * num_kv_heads * head_dim * elem_bytes
        write_cache = 2 * (cache_len + num_tokens) * num_kv_heads * head_dim * elem_bytes
        return read_kv + read_weight + read_cos_sin + read_cache + write_cache

    def prepare_inputs(self, num_tokens: int, num_kv_heads: int,
                       head_dim: int, cache_len: int = 128,
                       dtype: str = "bf16", eps: float = 1e-6,
                       **kwargs) -> dict:
        # eps 可能从 YAML 传入为字符串，强制转为 float
        eps = float(eps)

        torch_dtype = self.get_dtype(dtype)
        k = torch.randn(num_tokens, num_kv_heads, head_dim,
                       device=self.device, dtype=torch_dtype)
        v = torch.randn(num_tokens, num_kv_heads, head_dim,
                       device=self.device, dtype=torch_dtype)
        norm_weight = torch.ones(head_dim, device=self.device, dtype=torch_dtype)
        cos = torch.randn(num_tokens, 1, head_dim,
                         device=self.device, dtype=torch_dtype)
        sin = torch.randn(num_tokens, 1, head_dim,
                         device=self.device, dtype=torch_dtype)
        k_cache = torch.randn(cache_len, num_kv_heads, head_dim,
                             device=self.device, dtype=torch_dtype)
        v_cache = torch.randn(cache_len, num_kv_heads, head_dim,
                             device=self.device, dtype=torch_dtype)
        return {
            "k": k, "v": v, "norm_weight": norm_weight,
            "cos": cos, "sin": sin,
            "k_cache": k_cache, "v_cache": v_cache,
            "eps": eps
        }

    def compute_golden(self, k: torch.Tensor, v: torch.Tensor,
                       norm_weight: torch.Tensor, cos: torch.Tensor,
                       sin: torch.Tensor, k_cache: torch.Tensor,
                       v_cache: torch.Tensor, eps: float = 1e-6,
                       **kwargs) -> tuple:
        """Golden reference (CPU FP32)"""
        k_fp32 = k.float().cpu()
        v_fp32 = v.float().cpu()
        w_fp32 = norm_weight.float().cpu()
        cos_fp32 = cos.float().cpu()
        sin_fp32 = sin.float().cpu()
        kc_fp32 = k_cache.float().cpu()
        vc_fp32 = v_cache.float().cpu()

        # RMSNorm
        def rms_norm_ref(x, w, eps):
            variance = x.pow(2).mean(-1, keepdim=True)
            return x * torch.rsqrt(variance + eps) * w

        k_normed = rms_norm_ref(k_fp32, w_fp32, eps)
        v_normed = rms_norm_ref(v_fp32, w_fp32, eps)

        # RoPE on K
        k1, k2 = k_normed.chunk(2, dim=-1)
        k_rot = torch.cat([-k2, k1], dim=-1)
        k_roped = k_normed * cos_fp32 + k_rot * sin_fp32

        # Cache
        k_cached = torch.cat([kc_fp32, k_roped], dim=0)
        v_cached = torch.cat([vc_fp32, v_normed], dim=0)

        return k_cached.to(k.dtype).to(k.device), v_cached.to(v.dtype).to(v.device)
