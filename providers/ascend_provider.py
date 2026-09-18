"""Ascend Provider

昇腾平台性能基线 Provider。

vllm-ascend 模块路径与 NV 大体一致，复用 NvidiaProvider 的 impl_map。
算子 case 里 tensor 默认 device=cuda，容器内实际为 npu，setup 时
将 torch 工厂函数的 cuda 设备重定向到 npu:0。

swiglu 优先 torch.ops._C.silu_and_mul；无则 F.silu * mul 兜底。
"""
import torch
import torch.nn.functional as F

from .nvidia_provider import NvidiaProvider
from .registry import register_provider


def _patch_tensor_factory_for_npu() -> None:
    """将 operators 里 device=cuda 的创建重定向到 npu:0。"""
    npu_dev = torch.device("npu:0")

    def _fix_device(kwargs: dict) -> dict:
        dev = kwargs.get("device")
        if dev is None:
            return kwargs
        dev_str = str(dev)
        if dev == "cuda" or dev_str == "cuda" or dev_str.startswith("cuda:"):
            return {**kwargs, "device": npu_dev}
        return kwargs

    # randint 等也要覆盖（combine_topk 等 case 会用）；可重复调用补缺
    names = (
        "randn", "rand", "randint", "empty", "zeros", "ones", "full", "arange", "tensor",
    )
    patched = getattr(torch, "_flagopbench_npu_patched_names", set())
    for name in names:
        if name in patched or not hasattr(torch, name):
            continue
        orig = getattr(torch, name)

        def make_wrapper(fn):
            def wrapper(*args, **kwargs):
                return fn(*args, **_fix_device(kwargs))
            wrapper.__name__ = getattr(fn, "__name__", "wrapped")
            return wrapper

        setattr(torch, name, make_wrapper(orig))
        patched.add(name)

    torch._flagopbench_npu_patched_names = patched
    torch._flagopbench_npu_patch = True


@register_provider("ascend", platform="ascend", is_default=True)
class AscendProvider(NvidiaProvider):
    """昇腾平台算子实现加载器（vllm-ascend 优先，torch_npu / torch 兜底）"""

    @property
    def name(self) -> str:
        return "ascend"

    @property
    def platform(self) -> str:
        return "ascend"

    def get_device(self) -> torch.device:
        return torch.device("npu:0")

    def synchronize(self) -> None:
        torch.npu.synchronize()

    def is_available(self) -> bool:
        try:
            import torch_npu  # noqa: F401
            return torch.npu.is_available()
        except ImportError:
            return False

    def setup(self):
        try:
            import torch_npu
            print(f"  Loaded torch_npu: {torch_npu.__version__}")
        except ImportError as e:
            print(f"  [WARN] Failed to import torch_npu: {e}")
            return

        if not torch.npu.is_available():
            print("  [WARN] torch.npu.is_available()=False; "
                  "check ASCEND_RT_VISIBLE_DEVICES / driver mount")
            return

        _patch_tensor_factory_for_npu()

        try:
            import vllm
            self._vllm = vllm
            print(f"  Loaded vllm-ascend: {vllm.__version__ if hasattr(vllm, '__version__') else 'unknown'}")
        except Exception as e:
            print(f"  [WARN] Failed to import vllm: {type(e).__name__}: {e}")
            self._vllm = None
            self._vllm_ops = None
            self._vllm_v1_ops = None
            self._vllm_mhc = None
            self._vllm_sparse_attn = None
            self._vllm_fused_moe = None
            self._vllm_flash_attn = None
            self._torch_ops_registered = self._has_silu_and_mul()
            print(f"  Loaded vllm modules: _custom_ops=False, v1_ops=False, "
                  f"mhc=False, flash_attn=False, "
                  f"torch_ops._C.silu_and_mul={self._torch_ops_registered}")
            return

        try:
            from vllm import _custom_ops
            self._vllm_ops = _custom_ops
        except Exception as e:
            print(f"  [WARN] Failed to import vllm._custom_ops: {type(e).__name__}: {e}")
            self._vllm_ops = None

        try:
            from vllm.v1.attention.ops import deepseek_v4_ops, flashmla
            self._vllm_v1_ops = {
                "deepseek_v4": deepseek_v4_ops,
                "flashmla": flashmla,
            }
        except Exception as e:
            print(f"  [WARN] Failed to import v1 ops: {type(e).__name__}: {e}")
            self._vllm_v1_ops = None

        try:
            from vllm.model_executor.layers import mhc, sparse_attn_indexer
            from vllm.model_executor.layers.fused_moe import fused_moe as fused_moe_module
            self._vllm_mhc = mhc
            self._vllm_sparse_attn = sparse_attn_indexer
            self._vllm_fused_moe = fused_moe_module
        except Exception as e:
            print(f"  [WARN] Failed to import model_executor layers: {type(e).__name__}: {e}")
            self._vllm_mhc = None
            self._vllm_sparse_attn = None
            self._vllm_fused_moe = None

        try:
            from vllm.vllm_flash_attn import flash_attn_interface
            self._vllm_flash_attn = flash_attn_interface
        except Exception as e:
            print(f"  [WARN] Failed to import flash_attn: {type(e).__name__}: {e}")
            self._vllm_flash_attn = None

        self._torch_ops_registered = self._has_silu_and_mul()
        if not self._torch_ops_registered:
            print("  [WARN] torch.ops._C.silu_and_mul unavailable; "
                  "swiglu will use torch fallback")

        print(f"  Loaded vllm modules: _custom_ops={self._vllm_ops is not None}, "
              f"v1_ops={self._vllm_v1_ops is not None}, "
              f"mhc={self._vllm_mhc is not None}, "
              f"flash_attn={self._vllm_flash_attn is not None}, "
              f"torch_ops._C.silu_and_mul={self._torch_ops_registered}")

    @staticmethod
    def _has_silu_and_mul() -> bool:
        return hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "silu_and_mul")

    def _load_swiglu(self):
        if self._has_silu_and_mul():
            impl_fn, info = super()._load_swiglu()
            if impl_fn is not None:
                return impl_fn, info

        def wrapper(input_tensor, **kwargs):
            d = input_tensor.shape[-1] // 2
            x = input_tensor[..., :d]
            y = input_tensor[..., d:]
            return F.silu(x) * y

        return wrapper, {
            "source": "torch.nn.functional.silu * mul (fallback)",
            "type": "torch",
            "platform": "ascend",
        }

    def _load_moe_sum(self):
        """昇腾无 torch.ops._C.moe_sum；用 torch.sum 语义兜底。

        签名: input (T,K,H) -> output (T,H) 原地写入。
        """
        def wrapper(input, output):
            torch.sum(input, dim=1, out=output)
            return output

        return wrapper, {
            "source": "torch.sum(dim=1, out=...) (ascend fallback)",
            "type": "torch",
            "platform": "ascend",
        }

    def _load_grouped_topk(self):
        """昇腾 vLLM fused grouped_topk 仅 CUDA；直接 torch 语义兜底。"""
        from providers.flagos_provider import torch_grouped_topk

        print("  [INFO] grouped_topk: torch fallback (Ascend; vLLM CUDA-only)")
        return torch_grouped_topk, {
            "source": "torch.grouped_topk (ascend fallback)",
            "type": "torch",
            "platform": "ascend",
        }

    def _load_topk_softplus_sqrt(self):
        """昇腾常无 _moe_C；用 softplus+topk 语义兜底（对齐沐曦）。"""
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
                "type": "npu",
                "platform": "ascend",
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
            "  [INFO] topk_softplus_sqrt: _moe_C missing → torch softplus+topk (Ascend)"
        )
        return fallback, {
            "source": "torch.softplus+topk (ascend fallback; _moe_C missing)",
            "type": "torch",
            "platform": "ascend",
        }

    def _load_combine_topk_swa_indices(self):
        """昇腾 vLLM deepseek_v4_ops 走 CUDA；直接 torch 语义兜底。"""
        from providers.flagos_provider import torch_combine_topk_swa_indices

        print(
            "  [INFO] combine_topk_swa_indices: skip CUDA deepseek_v4_ops "
            "→ torch fallback (Ascend)"
        )
        return torch_combine_topk_swa_indices, {
            "source": "torch.combine_topk_swa_indices (ascend fallback)",
            "type": "torch",
            "platform": "ascend",
        }

    def _load_compute_global_topk(self):
        """昇腾 deepseek_v4_ops 走 CUDA；直接 torch block_table gather 兜底。"""

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
            "  [INFO] compute_global_topk: skip CUDA deepseek_v4_ops "
            "→ torch fallback (Ascend)"
        )
        return fallback, {
            "source": "torch.block_table_gather (ascend fallback)",
            "type": "torch",
            "platform": "ascend",
        }

    def _load_fused_q_kv_rmsnorm(self):
        """昇腾 deepseek_v4_ops 走 CUDA；直接 torch RMSNorm 兜底。"""

        def _rms_norm(x, weight, eps):
            var = x.float().pow(2).mean(dim=-1, keepdim=True)
            y = x.float() * torch.rsqrt(var + eps)
            return (y * weight.float()).to(x.dtype)

        def fallback(qr, kv, q_weight, kv_weight, eps):
            return _rms_norm(qr, q_weight, eps), _rms_norm(kv, kv_weight, eps)

        print(
            "  [INFO] fused_q_kv_rmsnorm: skip CUDA deepseek_v4_ops "
            "→ torch fallback (Ascend)"
        )
        return fallback, {
            "source": "torch.rms_norm (ascend fallback)",
            "type": "torch",
            "platform": "ascend",
        }

    def _load_fused_deepseek_v4(self):
        """昇腾无可用 CUDA fused；直接 torch 参考兜底。"""
        from providers.flagos_provider import (
            torch_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert,
        )

        print(
            "  [INFO] fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert: "
            "skip CUDA → torch fallback (Ascend)"
        )
        return torch_fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert, {
            "source": "torch.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert (ascend fallback)",
            "type": "torch",
            "platform": "ascend",
        }

    def _load_mhc_post(self):
        """昇腾 vLLM mhc 常缺/CUDA；直接 torch 语义兜底。"""

        def fallback(x, residual, post_layer_mix, comb_res_mix):
            plm = post_layer_mix
            if plm.ndim == 2:
                plm = plm.unsqueeze(-1)
            y = x.unsqueeze(-2) * plm + torch.bmm(comb_res_mix.mT, residual.float())
            return y.type_as(x)

        print("  [INFO] mhc_post: skip vLLM mhc → torch fallback (Ascend)")
        return fallback, {
            "source": "torch.mhc_post (ascend fallback)",
            "type": "torch",
            "platform": "ascend",
        }

    def _load_mhc_pre(self):
        """昇腾 vLLM mhc 常缺；优先 flag_gems.mhc_pre_ref。"""
        try:
            from flag_gems.fused.mhc.mhc_pre import mhc_pre_ref

            print("  [INFO] mhc_pre: skip vLLM mhc → flag_gems.mhc_pre_ref (Ascend)")
            return mhc_pre_ref, {
                "source": "flag_gems.mhc_pre_ref (ascend fallback)",
                "type": "torch",
                "platform": "ascend",
            }
        except ImportError as e:
            print(f"  [WARN] mhc_pre_ref import failed: {e}")
            return None, {"error": f"mhc_pre unavailable: {e}"}

    def get_impl(self, op_name, operator):
        impl_fn, impl_info = super().get_impl(op_name, operator)
        if impl_fn is not None:
            impl_info = {**impl_info, "platform": "ascend"}
        return impl_fn, impl_info
