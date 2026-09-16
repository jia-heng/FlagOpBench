# FlagOpBench 算子集成状态报告（0901 更新）

## 总览

关键算子列表共 **82** 个，当前已完成 **27** 个（含语义等价），待开发 **55** 个。

| 状态 | 数量 | 说明 |
|------|------|------|
| ✅ 已完成 (精确匹配) | 21 | 算子名与 keyoplist 完全一致 |
| ✅ 已完成 (语义等价) | 6 | 现有算子覆盖了 keyoplist 中对应项 |
| 🔴 待开发 | 55 | 尚未接入 benchmark |

---

## ✅ 已完成算子（27 个）

### 精确匹配（21 个）

| # | 算子 | 算子库 | Baseline | 状态 |
|---|------|--------|----------|------|
| 7 | swiglu | FlagGems | ✅ | 双边对比 |
| 16 | combine_topk_swa_indices | FlagGems-vllm | ✅ | 双边对比 |
| 17 | compute_global_topk_indices_and_lens | FlagGems-vllm | ✅ | 双边对比 |
| 18 | cp_gather_indexer_k_quant_cache | FlagGems-vllm | ✅ | 双边对比 |
| 20 | flash_attn_varlen_func | FlagGems-vllm | ✅ | 双边对比 |
| 21 | flash_mla | FlagGems-vllm | ❌ SKIP | 仅 FlagOS（vLLM 无 prefill MLA 单算子） |
| 24 | fp8_fp4_paged_mqa_logits | FlagGems-vllm | ❌ ERROR | 仅 FlagOS（deep_gemm head_dim=128 限制） |
| 26 | fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert | FlagGems-vllm | ✅ | 双边对比 |
| 28 | fused_q_kv_rmsnorm | FlagGems-vllm | ✅ | 双边对比 |
| 29 | grouped_topk | FlagGems-vllm | ✅ | 双边对比 |
| 30 | indexer_k_quant_and_cache | FlagGems-vllm | ✅ | 双边对比 |
| 31 | mhc_post | FlagGems-vllm | ✅ | 双边对比 |
| 32 | mhc_pre | FlagGems-vllm | ✅ | 双边对比 |
| 33 | moe_sum | FlagGems-vllm | ✅ | 双边对比 |
| 34 | pack_seq_triton | FlagGems-vllm | ✅ | 双边对比 |
| 37 | silu_and_mul_with_clamp | FlagGems-vllm | ✅ | 双边对比 |
| 38 | top_k_per_row_decode | FlagGems-vllm | ✅ | 双边对比 |
| 39 | top_k_per_row_prefill | FlagGems-vllm | ✅ | 双边对比 |
| 40 | topk_softplus_sqrt | FlagGems-vllm | ✅ | 双边对比 |
| 41 | unpack_seq_triton | FlagGems-vllm | ✅ | 双边对比 |
| 55 | flash_mla_with_kvcache | FlagGems-vllm | ✅ | 双边对比 |

### 语义等价（6 个，现有算子名 → keyoplist 对应项）

| 现有算子 | 对应 keyoplist # | 对应算子名 | 说明 |
|----------|-----------------|-----------|------|
| chunk_gated_delta_rule_flaggems_vllm | #15 | chunk_gated_delta_rule_fwd | flaggems_vllm 实现 |
| chunk_gated_delta_rule_flag_attn | #74 | Gated Delta Network (GDN) | flag_attn 实现 |
| chunk_gated_delta_rule_flag_gems | — | (FlagGems 另一实现) | 与 flaggems_vllm 版本并行存在 |
| flash_mla_with_kvcache_fp8 | #56 | flash_mla_with_kvcache_fwd_w8a8_fp8 | FP8 版 flash_mla_with_kvcache |
| fused_moe | #27 | fused_experts_impl | fused_moe 即 fused_experts 上层接口 |
| group_gemm | #45 | group_mm | 同类 group GEMM 算子 |

---

## 🔴 待开发算子（55 个）

### 第一批：基础算子 — FlagGems（12 个）

通用 BLAS / elementwise 算子，通常实现复杂度较低，baseline 用 PyTorch/cuBLAS 即可。

| # | 算子 | 算子库 | 优先级 | 备注 |
|---|------|--------|--------|------|
| 1 | topk | FlagGems | 🔥 高 | 常用 Top-K，复用 PyTorch topk 做 baseline |
| 2 | rms_norm | FlagGems | 🔥 高 | RMSNorm，所有 LLM 都用 |
| 3 | apply_rotary_pos_emb | FlagGems | 🔥 高 | RoPE，所有 LLM 都用 |
| 4 | mul | FlagGems | 中 | elementwise multiply |
| 5 | mv | FlagGems | 中 | matrix-vector multiply |
| 6 | bmm | FlagGems | 中 | batched matmul |
| 8 | baddbmm | FlagGems | 低 | batched add + bmm |
| 9 | mm | FlagGems | 中 | matmul |
| 10 | rms_norm_w8a16_fp8 | FlagGems | 🔥 高 | 量化 RMSNorm |
| 11 | conv2d | FlagGems | 低 | 2D 卷积 |
| 12 | addmm | FlagGems | 中 | add + matmul |
| 13 | glu | FlagGems | 中 | GLU activation |

### 第二批：FlagGems 新增专用算子（8 个）

| # | 算子 | 算子库 | 优先级 | 备注 |
|---|------|--------|--------|------|
| 42 | CausalConv1DPrefill | FlagGems | 🔥 高 | Jamba/Mamba 模型 causal conv |
| 43 | CausualConv1DDecode | FlagGems | 🔥 高 | 同上 decode 路径 |
| 44 | dsv3_router_gemm | FlagGems | 🔥 高 | DeepSeek V3 路由 GEMM |
| 46 | mm_w8a8_fp8 | FlagGems | 🔥 高 | FP8 matmul |
| 47 | router_gemm | FlagGems | 中 | 通用 router GEMM |
| 48 | TopK Selector | FlagGems | 中 | Top-K 选择器 |
| 49 | topk_w8a16_fp8 | FlagGems | 中 | 量化 Top-K |

### 第三批：FlagGems-vllm 推理算子（20 个）

| # | 算子 | 优先级 | 备注 |
|---|------|--------|------|
| 14 | add_rms_norm | 🔥 高 | Add + RMSNorm 融合 |
| 19 | dequantize_and_gather_k_cache | 🔥 高 | Dequant + Gather KV |
| 22 | flash_mla_sparse_fwd | 🔥 高 | Sparse MLA forward |
| 23 | fp8_fp4_mqa_logits | 🔥 高 | FP8/FP4 MQA logits（非 paged） |
| 25 | fused_add_rms_norm | 🔥 高 | Fused Add + RMSNorm |
| 35 | per_token_group_quant_fp8 | 🔥 高 | Per-token FP8 量化 |
| 36 | persistent_topk | 中 | Persistent Top-K |
| 50 | chunk_gdn2 | 🔥 高 | Gated DeltaNet-2 chunk fwd |
| 51 | chunk_gla | 🔥 高 | Gated Linear Attention chunk |
| 52 | chunk_kda | 中 | KDA chunk forward |
| 53 | flash_attn_varlen_func_w8a8_fp8 | 🔥 高 | FP8 版 flash_attn_varlen |
| 54 | flash_mla_sparse_fwd_w8a8_fp8 | 中 | FP8 版 sparse MLA |
| 57 | fp8_einsum | 中 | FP8 einsum |
| 58 | fused_deepseek_v4_qnorm_rope_kv_rope_insert | 中 | 不含 quant 的版本 |
| 59 | fused_inv_rope_fp8_quant | 中 | 逆 RoPE + FP8 量化 |
| 65 | gemma_rms_norm | 低 | Gemma 特有 RMSNorm |
| 66 | lightning_indexer | 中 | Lightning indexer |
| 69 | megamoe | 🔥 高 | MegaMoE 融合 kernel |
| 70 | moe_align_block_size | 中 | MoE block size 对齐 |
| 71 | w8a8_block_fp8_matmul | 🔥 高 | W8A8 block FP8 matmul |

### 第四批：FlagGems-vllm Marlin MoE 系列（5 个）

| # | 算子 | 优先级 | 备注 |
|---|------|--------|------|
| 60 | fused_marlin_moe_w4a16_mxfp4 | 中 | Marlin MoE MXFP4 |
| 61 | fused_marlin_moe_w4a16_int4 | 中 | Marlin MoE INT4 |
| 62 | fused_marlin_moe_w4a16_mxfp4 (v2) | 中 | Marlin MoE MXFP4 v2 |
| 63 | fused_marlin_moe_w8a16_fp8 | 中 | Marlin MoE FP8 |
| 64 | fused_marlin_moe_w8a16_int8 | 中 | Marlin MoE INT8 |

### 第五批：FlagGems-vllm Mega 系列 + 待定（2 个）

| # | 算子 | 优先级 | 备注 |
|---|------|--------|------|
| 67 | MegaGDN | 🔥 高 | Mega Gated Delta Network |
| 68 | MegaKernel (待定) | — | 具体接口待定 |

### 第六批：FlagAttention 注意力算子（10 个）

| # | 算子 | 优先级 | 备注 |
|---|------|--------|------|
| 72 | ACP-enabled Forgetting Attention | 中 | 带遗忘机制的 attention |
| 73 | AttnRes | 中 | Attention with residual |
| 75 | Gated DeltaNet-2 | 🔥 高 | GDN2 的 flag_attn 实现 |
| 76 | Gated Linear Attention | 🔥 高 | GLA 的 flag_attn 实现 |
| 77 | Inkling FA4 Relative Attention | 中 | Inkling 相对位置 attention |
| 78 | log_linear_attn | 中 | Log-linear attention |
| 79 | MiniMax Sparse Attention | 中 | MiniMax 稀疏 attention |
| 80 | moba | 🔥 高 | Mixture of Block Attention |
| 81 | parallax | 中 | Parallax attention |
| 82 | SageAttention | 🔥 高 | Sage attention |

---

## 建议开发顺序

### Phase 1 — 高频基础算子（快速扩充覆盖率）

目标：快速把覆盖率从 27/82 → 42/82

| 优先级 | 算子 | 理由 |
|--------|------|------|
| P0 | rms_norm, apply_rotary_pos_emb, topk | 所有 LLM 推理必用，实现简单 |
| P0 | add_rms_norm, fused_add_rms_norm | 高频融合算子 |
| P0 | per_token_group_quant_fp8 | 量化流程核心 |
| P0 | w8a8_block_fp8_matmul, mm_w8a8_fp8 | FP8 推理核心矩阵乘 |
| P0 | dequantize_and_gather_k_cache | KV cache 量化必需 |
| P0 | flash_mla_sparse_fwd | DeepSeek MLA sparse 路径 |
| P0 | fp8_fp4_mqa_logits | 与已有 paged 版本类似，复用度高 |
| P0 | chunk_gdn2, chunk_gla | 新一代线性注意力 |
| P0 | megamoe, MegaGDN | Mega 系列核心 |

### Phase 2 — 推理专用 + 新模型支持

| 优先级 | 算子 | 理由 |
|--------|------|------|
| P1 | CausalConv1DPrefill/Decode | Jamba/Mamba SSM 模型支持 |
| P1 | dsv3_router_gemm | DeepSeek V3 路由 |
| P1 | flash_attn_varlen_func_w8a8_fp8 | FP8 Flash Attention |
| P1 | SageAttention, moba | 热门 attention 变体 |
| P1 | Gated DeltaNet-2, Gated Linear Attention | FlagAttention 新算子 |
| P1 | fused_inv_rope_fp8_quant | 逆 RoPE 量化融合 |
| P1 | lightning_indexer | 高性能 indexer |

### Phase 3 — Marlin MoE + 其余

| 优先级 | 算子 | 理由 |
|--------|------|------|
| P2 | fused_marlin_moe 系列（5 个） | Marlin 量化 MoE，可批量接入 |
| P2 | mul, mv, bmm, mm, addmm, baddbmm, glu | 通用 BLAS，优先级低但实现简单 |
| P2 | 其余 FlagAttention | attention 变体 |
| P2 | MegaKernel | 接口待定 |

---

*更新时间: 2026-09-01*
*基于 keyoplist0901.md，共 82 个关键算子*
*当前已完成: 27/82 (33%)*
