# 55 个算子 FlashInfer-Trace 真实 Shape 匹配表

**生成日期**: 2026-08-10  
**目的**: 明确哪些算子可以复用 FlashInfer-Trace 中的真实 workload shape

---

## 📊 总体统计

| 维度 | 数量 | 占比 |
|------|------|------|
| **总算子数** | 55 | 100% |
| **已实现** | 33 | 60% |
| **待实现** | 22 | 40% |
| **有 FlashInfer 真实 shape** | 20 | 36% |
| **无 FlashInfer 真实 shape** | 35 | 64% |

---

## 📋 详细映射表

| # | 算子名称 | 状态 | FlashInfer 类别 | Definitions | 可复用 Workload |
|---|---------|------|----------------|-------------|----------------|
| 1 | mm | ✅ | gemm | 8 | ✅ |
| 2 | bmm | ✅ | gemm | 8 | ✅ |
| 3 | grouped_matmul | ✅ | gemm | 8 | ✅ |
| 4 | rms_norm | ✅ | rmsnorm | 9 | ✅ |
| 5 | gemma_rms_norm | ✅ | rmsnorm | 9 | ✅ |
| 6 | layernorm | ✅ | — | — | ❌ |
| 7 | add_rmsnorm_bias | ✅ | rmsnorm | 9 | ✅ |
| 8 | fused_q_kv_rmsnorm | ✅ | rmsnorm | 9 | ✅ |
| 9 | gelu | ✅ | — | — | ❌ |
| 10 | softmax | ✅ | — | — | ❌ |
| 11 | silu_and_mul | ✅ | — | — | ❌ |
| 12 | swiglu | ✅ | — | — | ❌ |
| 13 | silu_and_mul_with_clamp | ✅ | — | — | ❌ |
| 14 | rope | ✅ | — | — | ❌ |
| 15 | flashattention | ✅ | gqa_ragged | 2 | ✅ |
| 16 | sparse_attention | ✅ | gqa_ragged | 2 | ✅ |
| 17 | moe_sum | ✅ | — | — | ❌ |
| 18 | router_gemm | ✅ | — | — | ❌ |
| 19 | topk_softplus_sqrt | ✅ | — | — | ❌ |
| 20 | fused_moe | ✅ | moe | 1 | ✅ |
| 21 | moe_align_block_size | ✅ | — | — | ❌ |
| 22 | top_k_per_row_prefill | ✅ | sampling | 9 | ✅ |
| 23 | top_k_per_row_decode | ✅ | sampling | 9 | ✅ |
| 24 | persistent_topk | ✅ | sampling | 9 | ✅ |
| 25 | topk_selector | ✅ | sampling | 9 | ✅ |
| 26 | gemm_w8a8 | ✅ | — | — | ❌ |
| 27 | per_token_group_fp8_quant | ✅ | — | — | ❌ |
| 28 | fused_marlin_moe | ✅ | moe | 1 | ✅ |
| 29 | causal_conv1d | ✅ | — | — | ❌ |
| 30 | fp8_einsum | ✅ | — | — | ❌ |
| 31 | flash_mla | ✅ | mla_paged | 2 | ✅ |
| 32 | fused_inv_rope_fp8_quant | ✅ | — | — | ❌ |
| 33 | kv_rms_norm_rope_cache | ✅ | — | — | ❌ |
| 34 | paged_attention | ❌ | gqa_paged | 4 | ✅ |
| 35 | paged_attention_v1 | ❌ | gqa_paged | 4 | ✅ |
| 36 | paged_attention_v2 | ❌ | gqa_paged | 4 | ✅ |
| 37 | flash_mla_with_kvcache | ❌ | mla_paged | 2 | ✅ |
| 38 | flash_linear_attention | ❌ | — | — | ❌ |
| 39 | gla_attention | ❌ | — | — | ❌ |
| 40 | GDN | ❌ | — | — | ❌ |
| 41 | append_kv_cache | ❌ | — | — | ❌ |
| 42 | reshape_and_cache | ❌ | — | — | ❌ |
| 43 | deepseek_v3_moe_kernel | ❌ | — | — | ❌ |
| 44 | MegaMoe | ❌ | moe | 1 | ✅ |
| 45 | allreduce | ❌ | — | — | ❌ |
| 46 | permute_tokens | ❌ | — | — | ❌ |
| 47 | pack_sequences | ❌ | — | — | ❌ |
| 48 | unpack_sequences | ❌ | — | — | ❌ |
| 49 | fused_gelu_mul | ❌ | — | — | ❌ |
| 50 | fused_add_rms_norm | ❌ | rmsnorm | 9 | ✅ |
| 51 | quantize_per_token | ❌ | — | — | ❌ |
| 52 | rotary_embedding_neox | ❌ | — | — | ❌ |
| 53 | apply_rotary_pos_emb | ❌ | — | — | ❌ |
| 54 | FlashKDA | ❌ | — | — | ❌ |
| 55 | lightning_indexer | ❌ | — | — | ❌ |

---

## 🎯 FlashInfer-Trace 可复用类别

### 1. GEMM (8 definitions)
**算子**: mm, bmm, grouped_matmul  
**状态**: ✅ 全部已实现  
**可复用**: 8 个真实 shape (不同的 N, K 组合)

### 2. RMSNorm (9 definitions)
**算子**: rms_norm, gemma_rms_norm, add_rmsnorm_bias, fused_q_kv_rmsnorm, fused_add_rms_norm  
**状态**: ✅ 4/5 已实现，1 个待实现  
**可复用**: 9 个真实 shape (hidden_size: 128-7168)

### 3. GQA Ragged (2 definitions)
**算子**: flashattention, sparse_attention  
**状态**: ✅ 全部已实现  
**可复用**: 2 个真实 shape (h32_kv4_d128, h32_kv8_d128)

### 4. GQA Paged (4 definitions)
**算子**: paged_attention, paged_attention_v1, paged_attention_v2  
**状态**: ❌ 全部待实现 (P1 优先级)  
**可复用**: 4 个真实 shape (不同 page_size 和 block_size)

### 5. MLA Paged (2 definitions)
**算子**: flash_mla, flash_mla_with_kvcache  
**状态**: ✅ 1/2 已实现，1 个待实现  
**可复用**: 2 个真实 shape (kv_lora_rank=512)

### 6. MoE (1 definition)
**算子**: fused_moe, fused_marlin_moe, MegaMoe  
**状态**: ✅ 2/3 已实现，1 个待实现  
**可复用**: 1 个真实 shape (topk=8, experts=32)

### 7. Sampling (9 definitions)
**算子**: top_k_per_row_prefill, top_k_per_row_decode, persistent_topk, topk_selector  
**状态**: ✅ 全部已实现  
**可复用**: 9 个真实 shape (vocab_size: 128256-151936)

---

## 🚀 复用优先级建议

### Phase 1: 更新已实现算子的测试用例 (13 个算子)

| 算子类别 | 算子列表 | FlashInfer 类别 | 行动 |
|---------|---------|----------------|------|
| GEMM | mm, bmm, grouped_matmul | gemm (8) | 用真实 shape 替换测试用例 |
| RMSNorm | rms_norm, gemma_rms_norm, add_rmsnorm_bias, fused_q_kv_rmsnorm | rmsnorm (9) | 用真实 shape 替换测试用例 |
| GQA | flashattention, sparse_attention | gqa_ragged (2) | 用真实 shape 替换测试用例 |
| MLA | flash_mla | mla_paged (2) | 用真实 shape 替换测试用例 |
| MoE | fused_moe, fused_marlin_moe | moe (1) | 用真实 shape 替换测试用例 |
| Sampling | top_k_per_row_*, persistent_topk, topk_selector | sampling (9) | 用真实 shape 替换测试用例 |

**预期收益**: 测试用例从"临时占位"升级为"真实 workload"，性能数据更有参考价值

### Phase 2: 优先实现有 FlashInfer shape 的待实现算子 (7 个)

| 优先级 | 算子 | FlashInfer 类别 | Definitions | 理由 |
|--------|-----|----------------|-------------|------|
| P1 | paged_attention | gqa_paged | 4 | 重要且有真实 shape |
| P1 | paged_attention_v1 | gqa_paged | 4 | 重要且有真实 shape |
| P1 | paged_attention_v2 | gqa_paged | 4 | 重要且有真实 shape |
| P1 | flash_mla_with_kvcache | mla_paged | 2 | DeepSeek-V3 核心算子 |
| P2 | MegaMoe | moe | 1 | 可复用 fused_moe 实现 |
| P2 | fused_add_rms_norm | rmsnorm | 9 | 简单融合算子 |

**预期收益**: 实现后立即有真实测试用例，不需要人工设计 shape

---

## 📝 下一步行动

### 立即行动 (本周)
1. **更新已实现算子的 YAML 测试用例**
   - 目标: 13 个已实现且有 FlashInfer shape 的算子
   - 方法: 从 `flashinfer-trace/definitions/` 复制真实 workload
   - 优先级: GEMM > RMSNorm > Attention > MoE > Sampling

2. **验证复用效果**
   - 运行更新后的测试用例
   - 确认 shape 正确性
   - 对比性能数据

### 短期计划 (2 周)
3. **实现 Phase 2 的 7 个待实现算子**
   - paged_attention 系列 (3 个)
   - flash_mla_with_kvcache (1 个)
   - MegaMoe, fused_add_rms_norm (2 个)

4. **完成 P0 剩余算子**
   - 无 FlashInfer shape 的算子仍需实现
   - 使用合理的手工设计 shape

---

## ✅ 成功标准

### Phase 1 (更新测试用例)
- [x] 识别 13 个可复用 FlashInfer shape 的已实现算子
- [ ] 更新 13 个算子的 YAML 文件
- [ ] 测试通过率 100%
- [ ] 性能数据对比分析

### Phase 2 (实现新算子)
- [ ] 7 个有 FlashInfer shape 的待实现算子完成
- [ ] 测试覆盖率 100%
- [ ] 文档完整

### 整体目标
- [ ] 55/55 算子全覆盖
- [ ] 20/55 算子使用真实 FlashInfer workload (36%)
- [ ] 35/55 算子使用合理手工设计 workload (64%)
