## 关键算子列表0901

| #    | 算子名称                                          | 算子库        |
| ---- | ------------------------------------------------- | ------------- |
| 1    | topk                                              | FlagGems      |
| 2    | rms_norm                                          | FlagGems      |
| 3    | apply_rotary_pos_emb                              | FlagGems      |
| 4    | mul                                               | FlagGems      |
| 5    | mv                                                | FlagGems      |
| 6    | bmm                                               | FlagGems      |
| 7    | swiglu                                            | FlagGems      |
| 8    | baddbmm                                           | FlagGems      |
| 9    | mm                                                | FlagGems      |
| 10   | rms_norm_w8a16_fp8                                | FlagGems      |
| 11   | conv2d                                            | FlagGems      |
| 12   | addmm                                             | FlagGems      |
| 13   | glu                                               | FlagGems      |
| 14   | add_rms_norm                                      | FlagGems-vllm |
| 15   | chunk_gated_delta_rule_fwd                        | FlagGems-vllm |
| 16   | combine_topk_swa_indices                          | FlagGems-vllm |
| 17   | compute_global_topk_indices_and_lens              | FlagGems-vllm |
| 18   | cp_gather_indexer_k_quant_cache                   | FlagGems-vllm |
| 19   | dequantize_and_gather_k_cache                     | FlagGems-vllm |
| 20   | flash_attn_varlen_func                            | FlagGems-vllm |
| 21   | flash_mla                                         | FlagGems-vllm |
| 22   | flash_mla_sparse_fwd                              | FlagGems-vllm |
| 23   | fp8_fp4_mqa_logits                                | FlagGems-vllm |
| 24   | fp8_fp4_paged_mqa_logits                          | FlagGems-vllm |
| 25   | fused_add_rms_norm                                | FlagGems-vllm |
| 26   | fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert | FlagGems-vllm |
| 27   | fused_experts_impl                                | FlagGems-vllm |
| 28   | fused_q_kv_rmsnorm                                | FlagGems-vllm |
| 29   | grouped_topk                                      | FlagGems-vllm |
| 30   | indexer_k_quant_and_cache                         | FlagGems-vllm |
| 31   | mhc_post                                          | FlagGems-vllm |
| 32   | mhc_pre                                           | FlagGems-vllm |
| 33   | moe_sum                                           | FlagGems-vllm |
| 34   | pack_seq_triton                                   | FlagGems-vllm |
| 35   | per_token_group_quant_fp8                         | FlagGems-vllm |
| 36   | persistent_topk                                   | FlagGems-vllm |
| 37   | silu_and_mul_with_clamp                           | FlagGems-vllm |
| 38   | top_k_per_row_decode                              | FlagGems-vllm |
| 39   | top_k_per_row_prefill                             | FlagGems-vllm |
| 40   | topk_softplus_sqrt                                | FlagGems-vllm |
| 41   | unpack_seq_triton                                 | FlagGems-vllm |
| 42   | CausalConv1DPrefill                               | FlagGems      |
| 43   | CausualConv1DDecode                               | FlagGems      |
| 44   | dsv3_router_gemm                                  | FlagGems      |
| 45   | group_mm                                          | FlagGems      |
| 46   | mm_w8a8_fp8                                       | FlagGems      |
| 47   | router_gemm                                       | FlagGems      |
| 48   | TopK Selector                                     | FlagGems      |
| 49   | topk_w8a16_fp8                                    | FlagGems      |
| 50   | chunk_gdn2                                        | FlagGems-vllm |
| 51   | chunk_gla                                         | FlagGems-vllm |
| 52   | chunk_kda                                         | FlagGems-vllm |
| 53   | flash_attn_varlen_func_w8a8_fp8                   | FlagGems-vllm |
| 54   | flash_mla_sparse_fwd_w8a8_fp8                     | FlagGems-vllm |
| 55   | flash_mla_with_kvcache                            | FlagGems-vllm |
| 56   | flash_mla_with_kvcache_fwd_w8a8_fp8               | FlagGems-vllm |
| 57   | fp8_einsum                                        | FlagGems-vllm |
| 58   | fused_deepseek_v4_qnorm_rope_kv_rope_insert       | FlagGems-vllm |
| 59   | fused_inv_rope_fp8_quant                          | FlagGems-vllm |
| 60   | fused_marlin_moe mxfp4 w4a16                      | FlagGems-vllm |
| 61   | fused_marlin_moe_w4a16_int4                       | FlagGems-vllm |
| 62   | fused_marlin_moe_w4a16_mxfp4                      | FlagGems-vllm |
| 63   | fused_marlin_moe_w8a16_fp8                        | FlagGems-vllm |
| 64   | fused_marlin_moe_w8a16_int8                       | FlagGems-vllm |
| 65   | gemma_rms_norm                                    | FlagGems-vllm |
| 66   | lightning_indexer                                 | FlagGems-vllm |
| 67   | MegaGDN                                           | FlagGems-vllm |
| 68   | MegaKernel待定                                    | FlagGems-vllm |
| 69   | megamoe                                           | FlagGems-vllm |
| 70   | moe_align_block_size                              | FlagGems-vllm |
| 71   | w8a8_block_fp8_matmul                             | FlagGems-vllm |
| 72   | ACP-enabled Forgetting Attention                  | FlagAttention |
| 73   | AttnRes                                           | FlagAttention |
| 74   | Gated Delta Network （GDN）                       | FlagAttention |
| 75   | Gated DeltaNet-2                                  | FlagAttention |
| 76   | Gated Linear Attention                            | FlagAttention |
| 77   | Inkling FA4 Relative Attention                    | FlagAttention |
| 78   | log_Iinear__attn                                  | FlagAttention |
| 79   | MiniMax Sparse Attention                          | FlagAttention |
| 80   | moba                                              | FlagAttention |
| 81   | parallax                                          | FlagAttention |
| 82   | SageAttention                                     | FlagAttention |