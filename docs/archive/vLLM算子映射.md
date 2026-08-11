# 55 个关键算子在 vLLM 中的实现映射

## 已通过 PyTorch 实现（23/55）✅

| # | 算子名称 | 实现状态 | 测试场景 |
|---|---------|---------|---------|
| 2 | silu_and_mul_with_clamp | ✅ 已实现 | 5 |
| 4 | moe_sum | ✅ 已实现 | 5 |
| 6 | topk_softplus_sqrt | ✅ 已实现 | 5 |
| 7 | swiglu | ✅ 已实现 | 5 |
| 8 | top_k_per_row_prefill | ✅ 已实现 | 4 |
| 15 | fused_q_kv_rmsnorm | ✅ 已实现 | 5 |
| 18 | top_k_per_row_decode | ✅ 已实现 | 4 |
| 21 | AddRmsNormBias | ✅ 已实现 | 5 |
| 22 | CausalConv1DPrefill | ✅ 已实现 | 4 |
| 23 | CausualConv1DDecode | ✅ 已实现 | 4 |
| 30 | fp8_einsum/w8a8_block_fp8_bmm | ✅ 已实现 | 4 |
| 39 | gemma_rms_norm | ✅ 已实现 | 5 |
| 41 | GroupedMatmul | ✅ 已实现 | 5 |
| 49 | persistent topk | ✅ 已实现 | 4 |
| 50 | rms_norm混合精度 | ✅ 已实现 | 7 |
| 51 | RoPE | ✅ 已实现 | 5 |
| 52 | router_gemm_bf16_fp32 | ✅ 已实现 | 5 |
| 54 | TopK Selector | ✅ 已实现 | 4 |
| 55 | topk 混合精度 | ✅ 已实现 | 4 |
| - | mm, bmm | ✅ 已实现 | 12 |
| - | layernorm | ✅ 已实现 | 6 |
| - | gelu | ✅ 已实现 | 5 |
| - | softmax | ✅ 已实现 | 6 |

## vLLM _custom_ops.py 中已有实现（待接入）

### MoE 系列

| # | 算子名称 | vLLM 实现 | 文件位置 |
|---|---------|-----------|---------|
| 4 | moe_sum | `moe_sum()` | `vllm/_custom_ops.py` |
| 14 | fused_moe | `cutlass_moe_mm()` | `vllm/_custom_ops.py` + `layers/fused_moe/` |
| 32 | fused_marlin_moe fp8 w8a16 | `gptq_marlin_moe_repack()` | `vllm/_custom_ops.py` |
| 33 | fused_marlin_moe int4 w4a16 | `awq_marlin_moe_repack()` | `vllm/_custom_ops.py` |
| 34 | fused_marlin_moe int8 w8a16 | `gptq_marlin_moe_repack()` | `vllm/_custom_ops.py` |
| 35 | fused_marlin_moe mxfp4 w4a16 | `cutlass_mxfp4_moe_mm()` | `vllm/_custom_ops.py` |
| 46 | MegaMoe | `cutlass_moe_mm()` | `layers/fused_moe/` |
| 47 | MoeAlignBlockSize | `moe_align_block_size()` | `vllm/_custom_ops.py` |

### TopK / Routing 系列

| # | 算子名称 | vLLM 实现 | 文件位置 |
|---|---------|-----------|---------|
| 6 | topk_softplus_sqrt | `topk_hash_softplus_sqrt()` | `vllm/_custom_ops.py` |
| 8 | top_k_per_row_prefill | `top_k_per_row_prefill()` | `vllm/_custom_ops.py` |
| 9 | combine_topk_swa_indices | 待查找具体实现 | - |
| 10 | compute_global_topk_indices_and_lens | 待查找具体实现 | - |
| 18 | top_k_per_row_decode | `top_k_per_row_decode()` | `vllm/_custom_ops.py` |
| 54 | TopK Selector | `topk_softmax()` / `grouped_topk()` | `vllm/_custom_ops.py` |

### MHC 系列

| # | 算子名称 | vLLM 实现 | 文件位置 |
|---|---------|-----------|---------|
| 1 | mhc_post | `MHCPostOp` (CustomOp) | `layers/mhc.py` |
| 3 | mhc_pre | `MHCPreOp` (CustomOp) | `layers/mhc.py` |

### MLA (Multi-head Latent Attention) 系列

| # | 算子名称 | vLLM 实现 | 文件位置 |
|---|---------|-----------|---------|
| 5 | flash_mla | `MLAAttention` | `layers/attention/mla_attention.py` |
| 24 | flash_mla_with_kvcache | `concat_and_cache_mla()` | `vllm/_custom_ops.py` |
| 25 | flash_mla_with_kvcache混合精度 | `concat_and_cache_mla_grouped()` | `vllm/_custom_ops.py` |
| 29 | flashmla_sparse混合精度 | `SparseMLA` | `layers/attention/sparse_mla_attention.py` |

### Flash Attention 系列

| # | 算子名称 | vLLM 实现 | 文件位置 |
|---|---------|-----------|---------|
| 26 | flashattention | `flash_attn_interface.py` | `vllm/vllm_flash_attn/` |
| 27 | Flashattention 混合精度 | 同上 | `vllm/vllm_flash_attn/` |
| 28 | FlashKDA | `fused_kda_decode()` | `vllm/_custom_ops.py` |

### KV Cache 系列

| # | 算子名称 | vLLM 实现 | 文件位置 |
|---|---------|-----------|---------|
| 11 | cp_gather_indexer_k_quant_cache | `cp_gather_indexer_k_quant_cache()` | `vllm/_custom_ops.py` |
| 12 | dequantize_and_gather_k_cache | `gather_and_maybe_dequant_cache()` | `vllm/_custom_ops.py` |
| 16 | indexer_k_quant_and_cache | `indexer_k_quant_and_cache()` | `vllm/_custom_ops.py` |
| 20 | fp8_fp4_paged_mqa_logits | `reshape_and_cache()` | `vllm/_custom_ops.py` |
| 42 | kv_rms_norm_rope_cache | 待查找具体实现 | - |

### RoPE / Norm 融合算子

| # | 算子名称 | vLLM 实现 | 文件位置 |
|---|---------|-----------|---------|
| 13 | fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert | `concat_and_cache_mla_rope_fused()` | `vllm/_custom_ops.py` |
| 15 | fused_q_kv_rmsnorm | `fused_qk_norm_rope()` | `vllm/_custom_ops.py` |
| 31 | fused_inv_rope_fp8_quant | 待查找具体实现 | - |
| 39 | gemma_rms_norm | `rms_norm()` | `vllm/_custom_ops.py` |
| 48 | per_token_group_fp8_quant | `rms_norm_dynamic_per_token_quant()` | `vllm/_custom_ops.py` |
| 50 | rms_norm混合精度 | `rms_norm()` | `vllm/_custom_ops.py` |

### Sequence Packing

| # | 算子名称 | vLLM 实现 | 文件位置 |
|---|---------|-----------|---------|
| 17 | pack_seq_triton | 待查找具体实现 | - |
| 19 | unpack_seq_triton | 待查找具体实现 | - |

### Linear Attention (需要 flash-linear-attention 库)

| # | 算子名称 | vLLM 实现 | 文件位置 |
|---|---------|-----------|---------|
| 36 | GDN/chunk_gated_delta_rule_fwd | `chunk_gated_delta_rule_cpu()` | `vllm/_custom_ops.py` |
| 37 | GDN2 | `fused_sigmoid_gating_delta_rule_update_*()` | `vllm/_custom_ops.py` |
| 40 | gla | 待查找具体实现 | - |
| 44 | MegaGDN | `fused_gdn_gating_cpu()` | `vllm/_custom_ops.py` |

### GEMM / 量化

| # | 算子名称 | vLLM 实现 | 文件位置 |
|---|---------|-----------|---------|
| 30 | fp8_einsum/w8a8_block_fp8_bmm | `cutlass_scaled_mm()` | `vllm/_custom_ops.py` |
| 38 | gemm w8a8 混合精度 | `cutlass_scaled_mm()` | `vllm/_custom_ops.py` |

### 其他

| # | 算子名称 | vLLM 实现 | 文件位置 |
|---|---------|-----------|---------|
| 43 | lightning_indexer | `LightningAttention` | `layers/lightning_attn.py` |
| 45 | MegaKernel待定 | - | - |
| 53 | sparse_attention | `SparseMLA` | `layers/attention/sparse_mla_attention.py` |

## 总结

### 实现状态统计

- ✅ **已完成 PyTorch 实现**: 23/55 (42%)
- 🔧 **vLLM 中有现成实现，可直接接入**: 约 25 个
- 🔍 **需要进一步查找或依赖外部库**: 约 7 个

### 下一步接入优先级

#### P0 - 高优先级（核心性能算子）

1. **fused_moe** (#14) - MoE 核心算子
   - 实现：`vllm/_custom_ops.py:cutlass_moe_mm()`
   - 位置：`vllm/model_executor/layers/fused_moe/`

2. **flashattention** (#26) - Attention 核心
   - 实现：`vllm/vllm_flash_attn/flash_attn_interface.py`

3. **flash_mla** (#5) - DeepSeek MLA 核心
   - 实现：`vllm/model_executor/layers/attention/mla_attention.py`

4. **moe_align_block_size** (#47)
   - 实现：`vllm/_custom_ops.py:moe_align_block_size()`

#### P1 - 中优先级（量化 / 特殊场景）

5. **fused_marlin_moe 系列** (#32-35) - 量化 MoE
6. **FlashKDA** (#28) - KV cache decode
7. **mhc_pre / mhc_post** (#1, #3) - MHC 架构

#### P2 - 低优先级（可选优化）

8. **GDN 系列** (#36, #37, #44) - Linear attention
9. **lightning_indexer** (#43) - Lightning attention
10. **sparse_attention** (#53) - 稀疏注意力

### 接入方式

所有 vLLM 算子可通过以下方式接入：

```python
# 在 operators/model/ 下创建新文件
import torch
from vllm._custom_ops import fused_moe, moe_align_block_size
from baseline.operators.registry import BaseOperator, register_operator

@register_operator("fused_moe")
class FusedMoEOperator(BaseOperator):
    def forward(self, ...):
        return fused_moe(...)  # 直接调用 vLLM 实现
```

**预计接入工作量**: 2-3 天可完成 P0 优先级算子（约 7 个）
