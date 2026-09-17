"""MetaX Provider

沐曦平台性能基线 Provider。

MACA 栈通过 torch.cuda 暴露设备；容器内常无 torch_metax 包。
vllm-metax 是上游 vLLM 的 fork，模块路径与 NV 一致，复用 NvidiaProvider
的 impl_map。NvidiaProvider.setup() 中各子模块独立 try/except，
缺模块保持为 None，对应算子自动 [SKIP]。

注意：NvidiaProvider._load_swiglu 写死 torch.ops._C.silu_and_mul；
沐曦常见情况是符号在 vllm._custom_ops。这里覆盖加载逻辑（对齐 WORKFLOW_VENDOR 2.3）。
"""
import torch
import torch.nn.functional as F

from .nvidia_provider import NvidiaProvider
from .registry import register_provider


@register_provider("metax", platform="metax", is_default=True)
class MetaxProvider(NvidiaProvider):
    """沐曦平台算子实现加载器（vllm-metax 优先，缺模块自动 SKIP）"""

    @property
    def name(self) -> str:
        return "metax"

    @property
    def platform(self) -> str:
        return "metax"

    def get_device(self) -> torch.device:
        # MACA 复用 torch.cuda，设备选择由 MACA_VISIBLE_DEVICES 控制
        return torch.device("cuda:0")

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def is_available(self) -> bool:
        # 以 torch.cuda 为准；torch_metax 并非所有镜像都有
        return torch.cuda.is_available()

    def setup(self):
        try:
            import torch_metax  # noqa: F401
            print("  Loaded torch_metax")
        except ImportError:
            print("  [INFO] torch_metax not present, using torch.cuda interface")

        if not torch.cuda.is_available():
            print(
                "  [WARN] torch.cuda.is_available()=False; "
                "unset CUDA_VISIBLE_DEVICES and set MACA_VISIBLE_DEVICES=6,7"
            )

        super().setup()
        self._ensure_vllm_custom_ops()
        # WORKFLOW_VENDOR 2.3：不要仅因 import vllm 成功就假定 torch.ops._C 可用
        self._torch_ops_registered = hasattr(torch.ops, "_C") and hasattr(
            torch.ops._C, "silu_and_mul"
        )

    def _ensure_vllm_custom_ops(self) -> None:
        """父类 setup 里 from vllm import _custom_ops 可能失败；探测脚本用 vllm._custom_ops 仍可用。"""
        if self._vllm_ops is not None:
            return
        try:
            from vllm import _custom_ops

            self._vllm_ops = _custom_ops
            print("  [INFO] Loaded vllm._custom_ops (metax)")
        except ImportError as e:
            print(f"  [WARN] vllm._custom_ops not available: {e}")

    def get_impl(self, op_name, operator):
        # swiglu 必须走本类覆盖，避免父类 map 绑到 NvidiaProvider._load_swiglu
        if op_name == "swiglu":
            try:
                impl_fn, impl_info = self._load_swiglu()
                if impl_fn is None:
                    return None, {"error": "Failed to load swiglu on metax"}
                return impl_fn, {**impl_info, "platform": "metax"}
            except Exception as e:
                print(f"  [WARN] Exception loading swiglu: {e}")
                return None, {"error": f"Failed to load swiglu: {e}"}

        # topk_hash_softplus_sqrt 常是薄封装，实际落到 torch.ops._moe_C；
        # 沐曦缺 mcoplib 时 hasattr 仍为 True，调用才炸 —— 需本类覆盖。
        if op_name == "topk_softplus_sqrt":
            try:
                impl_fn, impl_info = self._load_topk_softplus_sqrt()
                if impl_fn is None:
                    return None, {"error": "Failed to load topk_softplus_sqrt on metax"}
                return impl_fn, {**impl_info, "platform": "metax"}
            except Exception as e:
                print(f"  [WARN] Exception loading topk_softplus_sqrt: {e}")
                return None, {"error": f"Failed to load topk_softplus_sqrt: {e}"}

        # indexer 同理：Python 符号可能在，实际落到 torch.ops._C / _C_cache_ops
        if op_name == "indexer_k_quant_and_cache":
            try:
                impl_fn, impl_info = self._load_indexer_k_quant_and_cache()
                if impl_fn is None:
                    return None, {
                        "error": "Failed to load indexer_k_quant_and_cache on metax"
                    }
                return impl_fn, {**impl_info, "platform": "metax"}
            except Exception as e:
                print(f"  [WARN] Exception loading indexer_k_quant_and_cache: {e}")
                return None, {
                    "error": f"Failed to load indexer_k_quant_and_cache: {e}"
                }

        # deepseek_v4_ops 在 vllm-metax 常缺 → torch 语义兜底，保证双边出表
        if op_name == "compute_global_topk_indices_and_lens":
            try:
                impl_fn, impl_info = self._load_compute_global_topk()
                if impl_fn is None:
                    return None, {
                        "error": "Failed to load compute_global_topk on metax"
                    }
                return impl_fn, {**impl_info, "platform": "metax"}
            except Exception as e:
                print(f"  [WARN] Exception loading compute_global_topk: {e}")
                return None, {"error": f"Failed to load compute_global_topk: {e}"}

        if op_name == "fused_q_kv_rmsnorm":
            try:
                impl_fn, impl_info = self._load_fused_q_kv_rmsnorm()
                if impl_fn is None:
                    return None, {"error": "Failed to load fused_q_kv_rmsnorm on metax"}
                return impl_fn, {**impl_info, "platform": "metax"}
            except Exception as e:
                print(f"  [WARN] Exception loading fused_q_kv_rmsnorm: {e}")
                return None, {"error": f"Failed to load fused_q_kv_rmsnorm: {e}"}

        # vllm.model_executor.layers.mhc 在 vllm-metax 常缺 → flag_gems ref 兜底
        if op_name in ("mhc_post", "mhc_pre"):
            try:
                load = (
                    self._load_mhc_post if op_name == "mhc_post" else self._load_mhc_pre
                )
                impl_fn, impl_info = load()
                if impl_fn is None:
                    return None, {"error": f"Failed to load {op_name} on metax"}
                return impl_fn, {**impl_info, "platform": "metax"}
            except Exception as e:
                print(f"  [WARN] Exception loading {op_name}: {e}")
                return None, {"error": f"Failed to load {op_name}: {e}"}

        # sparse_attn / deep_gemm 在 vllm-metax 常缺；用例 d=576 也超 deep_gemm 128 限制
        if op_name == "fp8_fp4_paged_mqa_logits":
            try:
                impl_fn, impl_info = self._load_fp8_fp4_paged_mqa_logits()
                if impl_fn is None:
                    return None, {
                        "error": "Failed to load fp8_fp4_paged_mqa_logits on metax"
                    }
                return impl_fn, {**impl_info, "platform": "metax"}
            except Exception as e:
                print(f"  [WARN] Exception loading fp8_fp4_paged_mqa_logits: {e}")
                return None, {
                    "error": f"Failed to load fp8_fp4_paged_mqa_logits: {e}"
                }

        # grouped_topk: vLLM CUDA-only；缺则 torch 语义兜底
        if op_name == "grouped_topk":
            try:
                impl_fn, impl_info = self._load_grouped_topk()
                if impl_fn is None:
                    return None, {"error": "Failed to load grouped_topk on metax"}
                return impl_fn, {**impl_info, "platform": "metax"}
            except Exception as e:
                print(f"  [WARN] Exception loading grouped_topk: {e}")
                return None, {"error": f"Failed to load grouped_topk: {e}"}

        # group_gemm: torch._grouped_mm 在 MACA 常不可用 → mm loop
        if op_name == "group_gemm":
            try:
                impl_fn, impl_info = self._load_group_gemm()
                if impl_fn is None:
                    return None, {"error": "Failed to load group_gemm on metax"}
                return impl_fn, {**impl_info, "platform": "metax"}
            except Exception as e:
                print(f"  [WARN] Exception loading group_gemm: {e}")
                return None, {"error": f"Failed to load group_gemm: {e}"}

        # combine_topk_swa_indices: deepseek_v4_ops 常缺 → torch
        if op_name == "combine_topk_swa_indices":
            try:
                impl_fn, impl_info = self._load_combine_topk_swa_indices()
                if impl_fn is None:
                    return None, {
                        "error": "Failed to load combine_topk_swa_indices on metax"
                    }
                return impl_fn, {**impl_info, "platform": "metax"}
            except Exception as e:
                print(f"  [WARN] Exception loading combine_topk_swa_indices: {e}")
                return None, {
                    "error": f"Failed to load combine_topk_swa_indices: {e}"
                }

        # fused_deepseek_v4...: torch.ops._C 在 MACA 常缺 → torch 参考
        if op_name == "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert":
            try:
                impl_fn, impl_info = (
                    self._load_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert()
                )
                if impl_fn is None:
                    return None, {
                        "error": "Failed to load fused_deepseek_v4 on metax"
                    }
                return impl_fn, {**impl_info, "platform": "metax"}
            except Exception as e:
                print(f"  [WARN] Exception loading fused_deepseek_v4: {e}")
                return None, {"error": f"Failed to load fused_deepseek_v4: {e}"}

        # flash_attn_varlen: vllm_flash_attn / FlagGems 在 MACA 常不可用 → SDPA
        if op_name == "flash_attn_varlen_func":
            try:
                impl_fn, impl_info = self._load_flash_attn_varlen_func()
                if impl_fn is None:
                    return None, {
                        "error": "Failed to load flash_attn_varlen_func on metax"
                    }
                return impl_fn, {**impl_info, "platform": "metax"}
            except Exception as e:
                print(f"  [WARN] Exception loading flash_attn_varlen_func: {e}")
                return None, {
                    "error": f"Failed to load flash_attn_varlen_func: {e}"
                }

        # flash_mla 三件套：vLLM 常无/缺 flashmla → torch 兜底
        if op_name in (
            "flash_mla",
            "flash_mla_with_kvcache",
            "flash_mla_with_kvcache_fp8",
        ):
            try:
                load = {
                    "flash_mla": self._load_flash_mla,
                    "flash_mla_with_kvcache": self._load_flash_mla_with_kvcache,
                    "flash_mla_with_kvcache_fp8": self._load_flash_mla_with_kvcache_fp8,
                }[op_name]
                impl_fn, impl_info = load()
                if impl_fn is None:
                    return None, {"error": f"Failed to load {op_name} on metax"}
                return impl_fn, {**impl_info, "platform": "metax"}
            except Exception as e:
                print(f"  [WARN] Exception loading {op_name}: {e}")
                return None, {"error": f"Failed to load {op_name}: {e}"}

        impl_fn, impl_info = super().get_impl(op_name, operator)
        if impl_fn is not None:
            impl_info = {**impl_info, "platform": "metax"}
        return impl_fn, impl_info

    def _load_swiglu(self):
        """优先 vllm._custom_ops；否则 F.silu*mul fallback（对齐 ascend 策略）。"""
        if self._vllm_ops is not None and hasattr(self._vllm_ops, "silu_and_mul"):
            vllm_fn = self._vllm_ops.silu_and_mul

            def wrapper(input_tensor, **kwargs):
                d = input_tensor.shape[-1] // 2
                output_shape = input_tensor.shape[:-1] + (d,)
                out = torch.empty(
                    output_shape, dtype=input_tensor.dtype, device=input_tensor.device
                )
                vllm_fn(out, input_tensor)
                return out

            return wrapper, {
                "source": "vllm._custom_ops.silu_and_mul (metax)",
                "type": "cuda",
                "platform": "metax",
            }

        def fallback(input_tensor, **kwargs):
            d = input_tensor.shape[-1] // 2
            x = input_tensor[..., :d]
            y = input_tensor[..., d:]
            return F.silu(x) * y

        return fallback, {
            "source": "torch.nn.functional.silu * mul (metax fallback)",
            "type": "torch",
            "platform": "metax",
        }

    def _load_topk_softplus_sqrt(self):
        """优先真可用的 vLLM CUDA；否则 PyTorch 参考（对齐 FlagGems 测试 reference）。"""
        moe_ok = hasattr(torch.ops, "_moe_C") and hasattr(
            torch.ops._moe_C, "topk_softplus_sqrt"
        )
        if (
            moe_ok
            and self._vllm_ops is not None
            and hasattr(self._vllm_ops, "topk_hash_softplus_sqrt")
        ):
            return self._vllm_ops.topk_hash_softplus_sqrt, {
                "source": "vllm._custom_ops.topk_hash_softplus_sqrt",
                "type": "cuda",
                "platform": "metax",
            }

        def fallback(
            topk_weights,
            topk_indices,
            token_expert_indices,
            gating_output,
            renormalize,
            routed_scaling_factor,
            correction_bias=None,
            input_ids=None,
            tid2eid=None,
            **kwargs,
        ):
            scores = F.softplus(gating_output.float()).sqrt()
            original_scores = scores
            if correction_bias is not None:
                scores_for_choice = scores + correction_bias.unsqueeze(0)
            else:
                scores_for_choice = scores

            topk = topk_weights.shape[1]
            if tid2eid is not None:
                topk_ids = tid2eid[input_ids.long()]
            else:
                topk_ids = torch.topk(
                    scores_for_choice, k=topk, dim=-1, sorted=True
                )[1]

            weights = original_scores.gather(1, topk_ids.long())
            if renormalize:
                weights = weights / weights.sum(dim=-1, keepdim=True)
            if routed_scaling_factor != 1.0:
                weights = weights * routed_scaling_factor

            topk_weights.copy_(weights.to(torch.float32))
            topk_indices.copy_(topk_ids.to(torch.int32))
            num_tokens = topk_weights.shape[0]
            tei = (
                torch.arange(num_tokens, device=topk_weights.device).unsqueeze(1) * topk
                + torch.arange(topk, device=topk_weights.device)
            ).to(torch.int32)
            token_expert_indices.copy_(tei)
            return topk_weights, topk_indices, token_expert_indices

        print(
            "  [INFO] topk_softplus_sqrt: _moe_C missing → torch softplus+topk fallback"
        )
        return fallback, {
            "source": "torch.softplus+topk (metax fallback; _moe_C/mcoplib missing)",
            "type": "torch",
            "platform": "metax",
        }

    def _load_indexer_k_quant_and_cache(self):
        """仅当 vllm 符号真实可调时加载；缺 _C 则 SKIP（无合适的 torch 基线）。"""
        torch_ok = False
        for ns_name in ("_C", "_C_cache_ops"):
            ns = getattr(torch.ops, ns_name, None)
            if ns is not None and hasattr(ns, "indexer_k_quant_and_cache"):
                torch_ok = True
                break

        if (
            torch_ok
            and self._vllm_ops is not None
            and hasattr(self._vllm_ops, "indexer_k_quant_and_cache")
        ):
            vllm_fn = self._vllm_ops.indexer_k_quant_and_cache

            def wrapper(k, kv_cache, slot_mapping, quant_block_size, scale_fmt):
                return vllm_fn(
                    k,
                    kv_cache,
                    slot_mapping,
                    quant_block_size,
                    kv_cache_dtype=scale_fmt,
                )

            return wrapper, {
                "source": "vllm._custom_ops.indexer_k_quant_and_cache (adapted)",
                "type": "cuda",
                "platform": "metax",
            }

        print(
            "  [INFO] indexer_k_quant_and_cache: torch.ops/_C missing → baseline SKIP"
        )
        return None, {}

    def _load_compute_global_topk(self):
        """优先 vLLM deepseek_v4_ops；缺则 torch 语义兜底（与 FlagGems kernel 同口径）。"""
        try:
            impl_fn, impl_info = super()._load_compute_global_topk()
            if impl_fn is not None:
                return impl_fn, impl_info
        except Exception as e:
            print(f"  [WARN] vLLM compute_global_topk unavailable: {e}")

        def fallback(
            topk_indices,
            token_to_req_indices,
            block_table,
            block_size,
            is_valid_token=None,
        ):
            num_tokens, topk = topk_indices.shape
            if is_valid_token is None:
                is_valid_token = torch.ones(
                    (num_tokens,), device=topk_indices.device, dtype=torch.int32
                )
            global_indices = torch.full_like(topk_indices, -1)
            valid = topk_indices >= 0
            block_idx = torch.where(valid, topk_indices // block_size, 0)
            block_off = torch.where(valid, topk_indices - block_idx * block_size, 0)
            req = token_to_req_indices.unsqueeze(1).expand(-1, topk)
            # gather block_table[req, block_idx]
            block_no = block_table[
                req.clamp(min=0, max=block_table.shape[0] - 1),
                block_idx.clamp(min=0, max=block_table.shape[1] - 1),
            ]
            slot = block_no * block_size + block_off
            global_indices = torch.where(valid, slot, global_indices)
            counts = valid.to(torch.int32).sum(dim=1)
            lens = torch.where(is_valid_token != 0, counts, torch.zeros_like(counts))
            return global_indices.to(torch.int32), lens.to(torch.int32)

        print(
            "  [INFO] compute_global_topk: deepseek_v4_ops missing → torch fallback"
        )
        return fallback, {
            "source": "torch.block_table_gather (metax fallback)",
            "type": "torch",
            "platform": "metax",
        }

    def _load_fused_q_kv_rmsnorm(self):
        """优先 vLLM deepseek_v4_ops；缺则 torch RMSNorm 兜底。"""
        try:
            impl_fn, impl_info = super()._load_fused_q_kv_rmsnorm()
            if impl_fn is not None:
                return impl_fn, impl_info
        except Exception as e:
            print(f"  [WARN] vLLM fused_q_kv_rmsnorm unavailable: {e}")

        def _rms_norm(x, weight, eps):
            var = x.float().pow(2).mean(dim=-1, keepdim=True)
            y = x.float() * torch.rsqrt(var + eps)
            return (y * weight.float()).to(x.dtype)

        def fallback(qr, kv, q_weight, kv_weight, eps):
            return _rms_norm(qr, q_weight, eps), _rms_norm(kv, kv_weight, eps)

        print("  [INFO] fused_q_kv_rmsnorm: deepseek_v4_ops missing → torch fallback")
        return fallback, {
            "source": "torch.rms_norm (metax fallback)",
            "type": "torch",
            "platform": "metax",
        }

    def _load_mhc_post(self):
        """优先 vLLM mhc；缺则 flag_gems.mhc_post_ref。"""
        try:
            impl_fn, impl_info = super()._load_mhc_post()
            if impl_fn is not None:
                return impl_fn, impl_info
        except Exception as e:
            print(f"  [WARN] vLLM mhc_post unavailable: {e}")

        try:
            from flag_gems.fused.mhc.mhc_post import mhc_post_ref

            def wrap_ref(x, residual, post_layer_mix, comb_res_mix):
                plm = post_layer_mix
                if plm.ndim == 2:
                    plm = plm.unsqueeze(-1)
                return mhc_post_ref(x, residual, plm, comb_res_mix)

            print("  [INFO] mhc_post: vllm.mhc missing → flag_gems.mhc_post_ref")
            return wrap_ref, {
                "source": "flag_gems.mhc_post_ref (metax fallback)",
                "type": "torch",
                "platform": "metax",
            }
        except ImportError as e:
            print(f"  [WARN] mhc_post_ref import failed: {e}")

        def fallback(x, residual, post_layer_mix, comb_res_mix):
            plm = post_layer_mix
            if plm.ndim == 2:
                plm = plm.unsqueeze(-1)
            y = x.unsqueeze(-2) * plm + torch.bmm(comb_res_mix.mT, residual.float())
            return y.type_as(x)

        print("  [INFO] mhc_post: inline torch fallback")
        return fallback, {
            "source": "torch.mhc_post (metax fallback)",
            "type": "torch",
            "platform": "metax",
        }

    def _load_mhc_pre(self):
        """优先 vLLM mhc；缺则 flag_gems.mhc_pre_ref。"""
        try:
            impl_fn, impl_info = super()._load_mhc_pre()
            if impl_fn is not None:
                return impl_fn, impl_info
        except Exception as e:
            print(f"  [WARN] vLLM mhc_pre unavailable: {e}")

        try:
            from flag_gems.fused.mhc.mhc_pre import mhc_pre_ref

            print("  [INFO] mhc_pre: vllm.mhc missing → flag_gems.mhc_pre_ref")
            return mhc_pre_ref, {
                "source": "flag_gems.mhc_pre_ref (metax fallback)",
                "type": "torch",
                "platform": "metax",
            }
        except ImportError as e:
            print(f"  [WARN] mhc_pre_ref import failed: {e}")

        return None, {"error": "mhc_pre unavailable (no vllm.mhc / mhc_pre_ref)"}

    def _load_fp8_fp4_paged_mqa_logits(self):
        """沐曦：即使 import 到 sparse_attn，调用仍依赖 deep_gemm（缺 libcudart）→ 直接 torch。"""
        from providers.flagos_provider import torch_fp8_fp4_paged_mqa_logits

        print(
            "  [INFO] fp8_fp4_paged_mqa_logits: skip vLLM/deep_gemm "
            "→ torch fallback (MetaX)"
        )
        return torch_fp8_fp4_paged_mqa_logits, {
            "source": "torch.fp8_fp4_paged_mqa_logits (metax fallback)",
            "type": "pytorch",
            "platform": "metax",
        }

    def _load_grouped_topk(self):
        """优先 vLLM；沐曦常报 only CUDA → torch 语义兜底。"""
        from providers.flagos_provider import torch_grouped_topk

        if self._vllm_ops is not None and hasattr(self._vllm_ops, "grouped_topk"):
            vllm_fn = self._vllm_ops.grouped_topk
            try:
                s = torch.randn(1, 8, device="cuda", dtype=torch.float32)
                b = torch.zeros(8, device="cuda", dtype=torch.float32)
                vllm_fn(
                    s,
                    num_expert_group=2,
                    topk_group=1,
                    topk=2,
                    renormalize=True,
                    routed_scaling_factor=1.0,
                    bias=b,
                    scoring_func=1,
                )
                torch.cuda.synchronize()

                def wrapper(
                    scores,
                    n_group,
                    topk_group,
                    topk,
                    renormalize,
                    routed_scaling_factor,
                    bias,
                    scoring_func=0,
                    **kwargs,
                ):
                    return vllm_fn(
                        scores,
                        num_expert_group=n_group,
                        topk_group=topk_group,
                        topk=topk,
                        renormalize=renormalize,
                        routed_scaling_factor=routed_scaling_factor,
                        bias=bias,
                        scoring_func=scoring_func,
                    )

                return wrapper, {
                    "source": "vllm._custom_ops.grouped_topk (metax)",
                    "type": "cuda",
                    "platform": "metax",
                }
            except Exception as e:
                print(f"  [INFO] grouped_topk: vLLM probe failed → torch: {e}")

        print("  [INFO] grouped_topk: torch fallback (MetaX)")
        return torch_grouped_topk, {
            "source": "torch.grouped_topk (metax fallback)",
            "type": "torch",
            "platform": "metax",
        }

    def _load_group_gemm(self):
        """优先 torch._grouped_mm；MACA 不可用时 mm loop（对齐 mthreads）。"""
        from providers.flagos_provider import torch_group_mm

        if hasattr(torch, "_grouped_mm"):
            try:
                A = torch.randn(8, 16, device="cuda", dtype=torch.bfloat16)
                B = torch.randn(2, 16, 8, device="cuda", dtype=torch.bfloat16)
                offs = torch.tensor([4, 8], device="cuda", dtype=torch.int32)
                torch._grouped_mm(A, B, offs)
                torch.cuda.synchronize()

                def wrapper(A, B, offs):
                    return torch._grouped_mm(A, B, offs)

                return wrapper, {
                    "source": "torch._grouped_mm (metax)",
                    "type": "cutlass",
                    "platform": "metax",
                }
            except Exception as e:
                print(f"  [INFO] group_gemm: _grouped_mm probe failed → mm loop: {e}")

        print("  [INFO] group_gemm: torch.mm loop (MetaX)")
        return torch_group_mm, {
            "source": "torch.mm loop over groups (metax fallback)",
            "type": "torch",
            "platform": "metax",
        }

    def _load_combine_topk_swa_indices(self):
        """优先 vLLM deepseek_v4_ops；缺则 torch（对齐 FlagGems 单测参考）。"""
        from providers.flagos_provider import torch_combine_topk_swa_indices

        try:
            impl_fn, impl_info = super()._load_combine_topk_swa_indices()
            if impl_fn is not None:
                return impl_fn, impl_info
        except Exception as e:
            print(f"  [WARN] vLLM combine_topk_swa_indices unavailable: {e}")

        print(
            "  [INFO] combine_topk_swa_indices: deepseek_v4_ops missing "
            "→ torch fallback (MetaX)"
        )
        return torch_combine_topk_swa_indices, {
            "source": "torch.combine_topk_swa_indices (metax fallback)",
            "type": "torch",
            "platform": "metax",
        }

    def _load_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert(self):
        """优先 torch.ops._C；缺则 torch 参考（对齐 FlagGems 单测 ref_impl）。"""
        from providers.flagos_provider import (
            torch_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
        )

        op = "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert"
        if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, op):
            try:
                fn = getattr(torch.ops._C, op)
                q = torch.randn(1, 2, 512, device="cuda", dtype=torch.bfloat16)
                kv = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
                block_bytes = ((64 * 584 + 575) // 576) * 576
                k_cache = torch.zeros(2, block_bytes, device="cuda", dtype=torch.uint8)
                slot_mapping = torch.tensor([0], device="cuda", dtype=torch.int64)
                position_ids = torch.tensor([0], device="cuda", dtype=torch.int64)
                cos_sin_cache = torch.randn(16, 64, device="cuda", dtype=torch.float32)
                fn(q, kv, k_cache, slot_mapping, position_ids, cos_sin_cache, 1e-6, 64)
                torch.cuda.synchronize()
                return fn, {
                    "source": f"torch.ops._C.{op}",
                    "type": "cuda",
                    "platform": "metax",
                }
            except Exception as e:
                print(f"  [INFO] {op}: torch.ops._C probe failed → torch: {e}")

        print(f"  [INFO] {op}: torch fallback (MetaX)")
        return torch_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert, {
            "source": "torch.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert (metax fallback)",
            "type": "torch",
            "platform": "metax",
        }

    def _load_flash_attn_varlen_func(self):
        """优先 vllm_flash_attn；缺/挂则 torch SDPA（MetaX blacklist FlagGems 常挂）。"""
        from providers.flagos_provider import torch_flash_attn_varlen_func

        if self._vllm_flash_attn is not None and hasattr(
            self._vllm_flash_attn, "flash_attn_varlen_func"
        ):
            fn = self._vllm_flash_attn.flash_attn_varlen_func
            try:
                q = torch.randn(4, 2, 64, device="cuda", dtype=torch.bfloat16)
                k = torch.randn(4, 2, 64, device="cuda", dtype=torch.bfloat16)
                v = torch.randn(4, 2, 64, device="cuda", dtype=torch.bfloat16)
                cu = torch.tensor([0, 4], device="cuda", dtype=torch.int32)
                fn(
                    q=q,
                    k=k,
                    v=v,
                    cu_seqlens_q=cu,
                    cu_seqlens_k=cu,
                    max_seqlen_q=4,
                    max_seqlen_k=4,
                    causal=True,
                    softmax_scale=1.0 / (64**0.5),
                )
                torch.cuda.synchronize()
                return fn, {
                    "source": "vllm.vllm_flash_attn.flash_attn_varlen_func",
                    "type": "cuda",
                    "platform": "metax",
                }
            except Exception as e:
                print(f"  [INFO] flash_attn_varlen_func: vLLM probe failed → SDPA: {e}")

        print("  [INFO] flash_attn_varlen_func: torch SDPA fallback (MetaX)")
        return torch_flash_attn_varlen_func, {
            "source": "torch.sdpa_varlen (metax fallback)",
            "type": "torch",
            "platform": "metax",
        }

    def _load_flash_mla(self):
        """NV 无 vLLM 单算子；MetaX 用 torch SDPA 作基线。"""
        from providers.flagos_provider import torch_flash_mla

        print("  [INFO] flash_mla: no vLLM equiv → torch SDPA (MetaX)")
        return torch_flash_mla, {
            "source": "torch.flash_mla_sdpa (metax fallback)",
            "type": "torch",
            "platform": "metax",
        }

    def _load_flash_mla_with_kvcache(self):
        """优先 vLLM flashmla；缺则 torch。"""
        from providers.flagos_provider import torch_flash_mla_with_kvcache

        try:
            impl_fn, impl_info = super()._load_flash_mla_with_kvcache()
            if impl_fn is not None:
                return impl_fn, impl_info
        except Exception as e:
            print(f"  [INFO] flash_mla_with_kvcache: vLLM probe failed → torch: {e}")

        print("  [INFO] flash_mla_with_kvcache: torch SDPA (MetaX)")
        return torch_flash_mla_with_kvcache, {
            "source": "torch.flash_mla_with_kvcache_sdpa (metax fallback)",
            "type": "torch",
            "platform": "metax",
        }

    def _load_flash_mla_with_kvcache_fp8(self):
        """NV map 为 None；优先试 vLLM fp8 接口，否则 torch。"""
        from providers.flagos_provider import torch_flash_mla_with_kvcache

        try:
            impl_fn, impl_info = super()._load_flash_mla_with_kvcache_fp8()
            if impl_fn is not None:
                return impl_fn, impl_info
        except Exception as e:
            print(
                f"  [INFO] flash_mla_with_kvcache_fp8: vLLM probe failed → torch: {e}"
            )

        print("  [INFO] flash_mla_with_kvcache_fp8: torch SDPA (MetaX)")
        return torch_flash_mla_with_kvcache, {
            "source": "torch.flash_mla_with_kvcache_fp8_sdpa (metax fallback)",
            "type": "torch",
            "platform": "metax",
        }
