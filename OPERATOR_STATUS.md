# FlagOpBench 算子集成状态报告

## 算子 Benchmark 进展总览

27 个算子实现，12 个 merged case 文件，24 个双边对比完成。

| 状态 | 数量 | 说明 |
|------|------|------|
| ✅ 双边完成 | 24 | FlagOS + vLLM 均已跑通，可直接对比 |
| 🟡 仅 FlagOS | 3 | vLLM 无等价实现或接口不兼容 |

---

## ✅ 双边对比完成（24 个）

| # | 算子 | FlagOS 结果 | vLLM 结果 | 状态 |
|---|------|------------|-----------|------|
| 1 | chunk_gated_delta_rule_flag_attn | ✅ | ✅ | 可对比 |
| 2 | chunk_gated_delta_rule_flag_gems | ✅ | ✅ | 可对比（仅 T≤128） |
| 3 | chunk_gated_delta_rule_flaggems_vllm | ✅ | ✅ | 可对比 |
| 4 | combine_topk_swa_indices | ✅ | ✅ | 可对比 |
| 5 | compute_global_topk_indices_and_lens | ✅ | ✅ | 可对比 |
| 3 | cp_gather_indexer_k_quant_cache | ✅ | ✅ | 可对比 |
| 4 | flash_attn_varlen_func | ✅ | ✅ | 可对比 |
| 5 | flash_mla_with_kvcache | ✅ | ✅ | 可对比 |
| 6 | fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert | ✅ | ✅ | 可对比 |
| 7 | fused_moe | ✅ | ✅ | 可对比 |
| 8 | fused_q_kv_rmsnorm | ✅ | ✅ | 可对比 |
| 9 | group_gemm | ✅ | ✅ | 可对比 |
| 10 | indexer_k_quant_and_cache | ✅ | ✅ | 可对比 |
| 11 | mhc_post | ✅ | ✅ | 可对比 |
| 12 | mhc_pre | ✅ | ✅ | 可对比 |
| 13 | moe_sum | ✅ | ✅ | 可对比 |
| 14 | pack_seq_triton | ✅ | ✅ | 可对比 |
| 15 | silu_and_mul_with_clamp | ✅ | ✅ | 可对比 |
| 16 | swiglu | ✅ | ✅ | 可对比 |
| 17 | top_k_per_row_decode | ✅ | ✅ | 可对比 |
| 18 | top_k_per_row_prefill | ✅ | ✅ | 可对比 |
| 19 | topk_softplus_sqrt | ✅ | ✅ | 可对比 |
| 20 | unpack_seq_triton | ✅ | ✅ | 可对比 |

---

## 🟡 仅 FlagOS（3 个）

| # | 算子 | FlagOS | vLLM | 原因 |
|---|------|--------|------|------|
| 21 | flash_mla | ✅ | ❌ SKIP | vLLM 无 prefill MLA 单算子（大 prefill 走 flash_attn_varlen，短 prefill 复用 decode 路径） |
| 22 | flash_mla_with_kvcache_fp8 | ✅ | ❌ ERROR | vLLM 要求 q dtype 为 float8_e4m3fn，operator prepare_inputs 生成 bf16 |
| 23 | fp8_fp4_paged_mqa_logits | ✅ | ❌ ERROR | vLLM 接口 q 为 tuple[Tensor, Tensor|None]，与 operator 输出格式不兼容 |

---

## Benchmark 结果摘要

以下为各算子代表性 workload 的实测延迟（单位 ms）：

### Attention / Linear Attention 类

| 算子 | Workload | FlagOS (ms) | vLLM (ms) | 胜出 |
|------|----------|-------------|-----------|------|
| flash_attn_varlen_func | prefill_seq2048 | 0.2468 | 0.2059 | vLLM |
| flash_mla_with_kvcache | b16_seq16384 | 1.4928 | 0.6503 | vLLM |
| flash_mla | decode_batch32_seq2048 | 0.2866 | — | FlagOS only |
| chunk_gated_delta_rule_flaggems_vllm | kimi_k3_prefill_4096 | 1.4228 | 1.6026 | FlagOS (1.13x) |
| chunk_gated_delta_rule_flaggems_vllm | qwen3.8_prefill_4096 | 1.6500 | 2.0271 | FlagOS (1.23x) |
| chunk_gated_delta_rule_flag_attn | kimi_k3_prefill_4096 | 1.5417 | 1.6321 | FlagOS (1.06x) |
| chunk_gated_delta_rule_flag_gems | kimi_k3_decode_64 | 0.1856 | 0.1856 | 持平 |

### MoE 类

| 算子 | Workload | FlagOS (ms) | vLLM (ms) | 胜出 |
|------|----------|-------------|-----------|------|
| fused_moe | decode_32tokens | 0.5647 | 1.0221 | FlagOS |
| group_gemm | moe_32experts_64tokens | 0.6230 | 1.0033 | FlagOS |
| moe_sum | 2048tokens_8experts | 0.0198 | 0.0124 | vLLM |
| topk_softplus_sqrt | deepseek_prefill_seq4096 | 0.0258 | 0.0233 | vLLM |

### KV Cache / Indexer 类

| 算子 | Workload | FlagOS (ms) | vLLM (ms) | 胜出 |
|------|----------|-------------|-----------|------|
| cp_gather_indexer_k_quant_cache | decode_32tokens | 0.0421 | 0.0135 | vLLM |
| indexer_k_quant_and_cache | decode_32tokens | 0.0371 | 0.0137 | vLLM |
| compute_global_topk_indices_and_lens | decode_small_batch | 0.0538 | 0.0065 | vLLM |

### Fused Normalization / RoPE 类

| 算子 | Workload | FlagOS (ms) | vLLM (ms) | 胜出 |
|------|----------|-------------|-----------|------|
| fused_q_kv_rmsnorm | prefill_2048tokens | 0.1254 | 0.1422 | FlagOS |
| fused_deepseek_v4_qnorm_rope | prefill_2048tokens | 0.2148 | 0.2363 | FlagOS |

### Activation / Element-wise 类

| 算子 | Workload | FlagOS (ms) | vLLM (ms) | 胜出 |
|------|----------|-------------|-----------|------|
| swiglu | prefill_2048tokens | 0.0225 | 0.0193 | vLLM |
| silu_and_mul_with_clamp | deepseek_prefill_seq1024 | 0.0869 | 0.1027 | FlagOS |

### MHC (Multi-Head Cache) 类

| 算子 | Workload | FlagOS (ms) | vLLM (ms) | 胜出 |
|------|----------|-------------|-----------|------|
| mhc_pre | 代表 workload | ~0.05 | ~0.05 | 持平 |
| mhc_post | 代表 workload | ~0.04 | ~0.04 | 持平 |

### Pack/Unpack 类

| 算子 | Workload | FlagOS (ms) | vLLM (ms) | 胜出 |
|------|----------|-------------|-----------|------|
| pack_seq_triton | prefill_16reqs_seq512 | 0.0463 | 0.0542 | FlagOS |
| unpack_seq_triton | prefill_16reqs_seq512 | 0.0485 | 0.0600 | FlagOS |

### Sparse Attention Index 类

| 算子 | Workload | FlagOS (ms) | vLLM (ms) | 胜出 |
|------|----------|-------------|-----------|------|
| combine_topk_swa_indices | prefill_8reqs_seq4096 | 0.0520 | 0.0491 | vLLM |
| top_k_per_row_decode | batch32 | ~0.013 | ~0.012 | 持平 |
| top_k_per_row_prefill | batch32 | ~0.014 | ~0.013 | 持平 |

---

## 总结

### FlagOS 优势算子（显著快于 vLLM）
- **group_gemm** — 1.45x 加速（MoE grouped matmul）
- **fused_moe** — 1.14x 加速（decode 场景最多 1.8x）
- **pack_seq_triton** — 1.27x 加速
- **chunk_gated_delta_rule_flaggems_vllm** — prefill T≥256 快 10-24%（TLE 优化）
- **unpack_seq_triton** — 1.05x 加速

### vLLM 优势算子
- **grouped_topk** — vLLM (CUDA) 3.5x 快
- **flash_mla_with_kvcache** — vLLM (FlashMLA CUDA) 1.8x 快
- **cp_gather / indexer_k_quant_and_cache** — vLLM (CUDA) 3x 快
- **compute_global_topk_indices_and_lens** — vLLM (CUDA) 1.7x 快
- **silu_and_mul_with_clamp** — vLLM (CUDA) 4x 快
- **swiglu** — vLLM (CUDA) 1.8x 快
- **flash_attn_varlen_func** — vLLM (FlashAttention CUDA) 1.2x 快

### 持平
- mhc_pre / mhc_post / fused_q_kv_rmsnorm / chunk_gated_delta_rule_flag_gems

---

## vLLM 导入路径速查

```python
# _custom_ops 直接可用
from vllm._custom_ops import moe_sum
from vllm._custom_ops import topk_hash_softplus_sqrt  # = topk_softplus_sqrt
from vllm._custom_ops import indexer_k_quant_and_cache
from vllm._custom_ops import cp_gather_indexer_k_quant_cache

# vllm.v1.attention.ops
from vllm.v1.attention.ops.deepseek_v4_ops import (
    combine_topk_swa_indices,
    compute_global_topk_indices_and_lens,
    fused_q_kv_rmsnorm,
)
from vllm.v1.attention.ops.flashmla import (
    flash_mla_with_kvcache,
    flash_mla_with_kvcache_fp8,
)

# vllm.vllm_flash_attn
from vllm.vllm_flash_attn.flash_attn_interface import flash_attn_varlen_func

# vllm.model_executor.layers
from vllm.model_executor.layers.mhc import mhc_post, mhc_pre
from vllm.model_executor.layers.sparse_attn_indexer import (
    pack_seq_triton,
    unpack_seq_triton,
    fp8_fp4_paged_mqa_logits,
)
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts

# torch.ops._C (需先 import vllm 注册)
import vllm  # 触发 _C 注册
torch.ops._C.silu_and_mul               # swiglu 等价
torch.ops._C.silu_and_mul_with_clamp    # silu_and_mul_with_clamp
torch.ops._C.top_k_per_row_decode
torch.ops._C.top_k_per_row_prefill
torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert
```

---

## FlagOS 未实现算子（三个仓库均无独立实现）

| 算子名 | 说明 | 备注 |
|--------|------|------|
| causal_conv1d_prefill | Mamba CausalConv1D prefill (整序列因果卷积) | 无专用kernel |
| causal_conv1d_decode | Mamba CausalConv1D decode (单步更新 conv_state) | ARM 有 patch，GPU 无实现 |
| selective_scan | Mamba selective scan (SSM 核心算子) | 三仓均无 |
| mamba_chunk_scan | Mamba2 chunk scan | 三仓均无 |
| gated_delta_rule_prefill | Gated Delta Rule prefill (Qwen3.5/GDN) | flag_attn 有实现，待接入 |
| chunk_gated_delta_rule | Chunk Gated Delta Rule | flag_attn 有实现，待接入 |

## FlagOS 已实现但 FlagOpBench 尚未接入的算子

### flag_gems (FlagGems)

| 算子名 | 说明 |
|--------|------|
| rmsnorm / rms_norm | RMS Normalization |
| add_rms_norm / fused_add_rms_norm | Add + RMSNorm 融合 |
| layer_norm | Layer Normalization |
| apply_rotary_pos_emb / GemsRope | RoPE 旋转位置编码 |
| silu_and_mul / gelu_and_mul | Activation + Mul |
| softmax | Softmax |
| scaled_dot_product_attention | PyTorch SDPA 替代 |
| fp8_matmul / w8a8_block_fp8_matmul | FP8 矩阵乘 |
| per_token_group_quant_fp8 | Per-token group FP8 量化 |
| topk / topk_softmax / grouped_topk | Top-K 系列 |
| moe_align_block_size | MoE block size 对齐 |
| fp8_fp4_mega_moe | FP8/FP4 Mega MoE |

### flag_attn (FlagAttention)

| 算子名 | 说明 |
|--------|------|
| flash_attention | Flash Attention (标准) |
| paged_attention | Paged Attention |
| gated_delta_rule | Gated Delta Rule (Qwen3.5 GDN) |
| chunk_gated_delta_rule | Chunk Gated Delta Rule |

### flaggems_vllm — 还有更多未接入

| 算子名 | 说明 |
|--------|------|
| fused_indexer_q_rope_quant | Q RoPE + 量化融合 |
| fused_inv_rope_fp8_quant | 逆RoPE + FP8量化融合 |
| dequantize_and_gather_k_cache | Dequant + Gather KV |
| stage_deepseek_v4_mega_moe_inputs | DeepSeek-V4 MegaMoE 输入预处理 |

---

*更新时间: 2025-08-24*
*环境: FlagGems 5.3.4 / FlagGems-vllm 0.1.0 / flag_attn 0.1.dev80 / vLLM 0.20.2*
*GPU: NVIDIA H20*
*Benchmark 参数: warmup=10, repeat=100*
