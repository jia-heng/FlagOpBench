"""Nvidia Provider

NV平台最优性能基线Provider。
优先使用vLLM kernels，vLLM未覆盖的算子fallback到torch。
"""
from typing import Tuple, Callable, Dict, Optional
import torch
import torch.nn.functional as F

from framework.base_operator import BaseOperator
from .base_provider import BaseProvider
from .registry import register_provider


@register_provider("nvidia", platform="nvidia", is_default=True)
class NvidiaProvider(BaseProvider):
    """NV平台算子实现加载器（vLLM优先，torch fallback）"""

    def __init__(self):
        self._vllm = None
        self._vllm_ops = None
        self._vllm_v1_ops = None
        self._vllm_mhc = None
        self._vllm_sparse_attn = None
        self._vllm_fused_moe = None
        self._vllm_flash_attn = None
        self._torch_ops_registered = False

    @property
    def name(self) -> str:
        return "nvidia"

    @property
    def platform(self) -> str:
        return "nvidia"

    def get_device(self) -> torch.device:
        return torch.device("cuda:0")

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def setup(self):
        """延迟import vLLM相关库"""
        try:
            import vllm
            self._vllm = vllm
            print(f"  Loaded vllm: {vllm.__version__ if hasattr(vllm, '__version__') else 'unknown'}")
            self._torch_ops_registered = True
        except ImportError as e:
            print(f"  [WARN] Failed to import vllm: {e}")
            return

        # _custom_ops
        try:
            from vllm import _custom_ops
            self._vllm_ops = _custom_ops
        except ImportError:
            pass

        # v1.attention.ops
        try:
            from vllm.v1.attention.ops import deepseek_v4_ops, flashmla
            self._vllm_v1_ops = {
                'deepseek_v4': deepseek_v4_ops,
                'flashmla': flashmla,
            }
        except ImportError:
            pass

        # model_executor.layers
        try:
            from vllm.model_executor.layers import mhc, sparse_attn_indexer
            from vllm.model_executor.layers.fused_moe import fused_moe as fused_moe_module
            self._vllm_mhc = mhc
            self._vllm_sparse_attn = sparse_attn_indexer
            self._vllm_fused_moe = fused_moe_module
        except ImportError:
            pass

        # vllm_flash_attn
        try:
            from vllm.vllm_flash_attn import flash_attn_interface
            self._vllm_flash_attn = flash_attn_interface
        except ImportError:
            pass

        print(f"  Loaded vllm modules: _custom_ops={self._vllm_ops is not None}, "
              f"v1_ops={self._vllm_v1_ops is not None}, "
              f"mhc={self._vllm_mhc is not None}, "
              f"flash_attn={self._vllm_flash_attn is not None}")

    def get_impl(
        self,
        op_name: str,
        operator: BaseOperator
    ) -> Tuple[Optional[Callable], Dict[str, str]]:
        """获取vLLM算子实现"""

        # 映射表：算子名 -> (加载函数, 是否需要wrapper)
        # 共 23 个算子: 18 直接对应 + 3 需适配器 + 1 torch原生 + 1 无对应
        impl_map = {
            # === 签名兼容，直接加载（18个）===
            "moe_sum": (self._load_moe_sum, False),
            "combine_topk_swa_indices": (self._load_combine_topk_swa_indices, False),
            "compute_global_topk_indices_and_lens": (self._load_compute_global_topk, False),
            "flash_attn_varlen_func": (self._load_flash_attn_varlen_func, False),
            "fused_q_kv_rmsnorm": (self._load_fused_q_kv_rmsnorm, False),
            "mhc_post": (self._load_mhc_post, False),
            "mhc_pre": (self._load_mhc_pre, False),
            "pack_seq_triton": (self._load_pack_seq_triton, False),
            "unpack_seq_triton": (self._load_unpack_seq_triton, False),
            "top_k_per_row_decode": (self._load_top_k_per_row_decode, False),
            "top_k_per_row_prefill": (self._load_top_k_per_row_prefill, False),
            "fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert": (self._load_fused_deepseek_v4, False),
            "flash_mla_with_kvcache": (self._load_flash_mla_with_kvcache, False),
            "topk_softplus_sqrt": (self._load_topk_softplus_sqrt, False),
            "fused_moe": (self._load_fused_moe, False),
            "indexer_k_quant_and_cache": (self._load_indexer_k_quant_and_cache, False),
            "cp_gather_indexer_k_quant_cache": (self._load_cp_gather_indexer, False),
            "fp8_fp4_paged_mqa_logits": (self._load_fp8_fp4_paged_mqa_logits, False),

            # === 需要适配器（4个）===
            "swiglu": (self._load_swiglu, True),
            "silu_and_mul_with_clamp": (self._load_silu_and_mul_with_clamp, True),
            "grouped_topk": (self._load_grouped_topk, True),

            # === torch 原生对应（1个）===
            "group_gemm": (self._load_group_gemm, False),

            # === GDN 三方对比 (共用同一个 vLLM baseline) ===
            "chunk_gated_delta_rule_flaggems_vllm": (self._load_chunk_gated_delta_rule, True),
            "chunk_gated_delta_rule_flag_attn": (self._load_chunk_gated_delta_rule, True),
            "chunk_gated_delta_rule_flag_gems": (self._load_chunk_gated_delta_rule, True),

            # === vLLM 无对应（2个）===
            "flash_mla": (None, False),  # Prefill MLA，vLLM无单算子等价
            "flash_mla_with_kvcache_fp8": (None, False),  # Sparse FP8 MLA，vLLM的fp8变体不支持sparse+bf16_q
        }

        if op_name not in impl_map:
            return None, {"error": f"Unknown operator: {op_name}"}

        load_fn, needs_wrapper = impl_map[op_name]
        if load_fn is None:
            return None, {"error": f"vLLM has no equivalent for {op_name}"}

        try:
            impl_fn, impl_info = load_fn()
            if impl_fn is None:
                return None, {"error": f"Failed to load {op_name} from vLLM"}
            return impl_fn, impl_info
        except Exception as e:
            print(f"  [WARN] Exception loading {op_name}: {e}")
            return None, {"error": f"Failed to load {op_name}: {e}"}

    # ============================================================
    # 签名完全一致（直接加载）
    # ============================================================

    def _load_moe_sum(self):
        if self._vllm_ops is None:
            return None, {}
        return self._vllm_ops.moe_sum, {
            "source": "vllm._custom_ops.moe_sum",
            "type": "cuda"
        }

    def _load_combine_topk_swa_indices(self):
        if self._vllm_v1_ops is None:
            return None, {}
        fn = self._vllm_v1_ops['deepseek_v4'].combine_topk_swa_indices
        return fn, {
            "source": "vllm.v1.attention.ops.deepseek_v4_ops.combine_topk_swa_indices",
            "type": "triton"
        }

    def _load_compute_global_topk(self):
        if self._vllm_v1_ops is None:
            return None, {}
        fn = self._vllm_v1_ops['deepseek_v4'].compute_global_topk_indices_and_lens
        return fn, {
            "source": "vllm.v1.attention.ops.deepseek_v4_ops.compute_global_topk_indices_and_lens",
            "type": "triton"
        }

    def _load_fused_q_kv_rmsnorm(self):
        if self._vllm_v1_ops is None:
            return None, {}
        fn = self._vllm_v1_ops['deepseek_v4'].fused_q_kv_rmsnorm
        return fn, {
            "source": "vllm.v1.attention.ops.deepseek_v4_ops.fused_q_kv_rmsnorm",
            "type": "triton"
        }

    def _load_mhc_post(self):
        if self._vllm_mhc is None:
            return None, {}
        return self._vllm_mhc.mhc_post, {
            "source": "vllm.model_executor.layers.mhc.mhc_post",
            "type": "tilelang"
        }

    def _load_mhc_pre(self):
        if self._vllm_mhc is None:
            return None, {}
        return self._vllm_mhc.mhc_pre, {
            "source": "vllm.model_executor.layers.mhc.mhc_pre",
            "type": "tilelang"
        }

    def _load_pack_seq_triton(self):
        if self._vllm_sparse_attn is None:
            return None, {}
        return self._vllm_sparse_attn.pack_seq_triton, {
            "source": "vllm.model_executor.layers.sparse_attn_indexer.pack_seq_triton",
            "type": "triton"
        }

    def _load_unpack_seq_triton(self):
        if self._vllm_sparse_attn is None:
            return None, {}
        return self._vllm_sparse_attn.unpack_seq_triton, {
            "source": "vllm.model_executor.layers.sparse_attn_indexer.unpack_seq_triton",
            "type": "triton"
        }

    def _load_top_k_per_row_decode(self):
        """参数名转换: num_rows->numRows, top_k->topK"""
        if not self._torch_ops_registered:
            return None, {}

        vllm_fn = torch.ops._C.top_k_per_row_decode

        def wrapper(logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k):
            # vLLM 期望驼峰命名: numRows, topK
            return vllm_fn(logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k)

        return wrapper, {
            "source": "torch.ops._C.top_k_per_row_decode (adapted)",
            "type": "cuda"
        }

    def _load_top_k_per_row_prefill(self):
        """参数名转换: row_starts->rowStarts, row_ends->rowEnds, num_rows->numRows, top_k->topK"""
        if not self._torch_ops_registered:
            return None, {}

        vllm_fn = torch.ops._C.top_k_per_row_prefill

        def wrapper(logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k):
            # vLLM 期望驼峰命名: rowStarts, rowEnds, numRows, topK
            return vllm_fn(logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k)

        return wrapper, {
            "source": "torch.ops._C.top_k_per_row_prefill (adapted)",
            "type": "cuda"
        }

    def _load_fused_deepseek_v4(self):
        if not self._torch_ops_registered:
            return None, {}
        return torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert, {
            "source": "torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert",
            "type": "cuda"
        }

    def _load_flash_mla_with_kvcache(self):
        if self._vllm_v1_ops is None:
            return None, {}

        from vllm.third_party.flashmla.flash_mla_interface import FlashMLASchedMeta as VL_Meta
        vllm_fn = self._vllm_v1_ops['flashmla'].flash_mla_with_kvcache

        def wrapper(**kwargs):
            # 检查 tile_scheduler_metadata 类型，如果是 flaggems 的，转换为 vLLM 的
            meta = kwargs.get('tile_scheduler_metadata')
            if meta is not None and type(meta).__name__ == 'FlashMLASchedMeta' and 'flaggems' in type(meta).__module__:
                # 提取 flaggems meta 的属性，构造 vLLM meta
                vl_meta = VL_Meta()
                vl_meta.tile_scheduler_metadata = meta.tile_scheduler_metadata
                vl_meta.num_splits = meta.num_splits
                kwargs['tile_scheduler_metadata'] = vl_meta

            return vllm_fn(**kwargs)

        return wrapper, {
            "source": "vllm.v1.attention.ops.flashmla.flash_mla_with_kvcache (adapted)",
            "type": "cuda"
        }

    def _load_topk_softplus_sqrt(self):
        """topk_softplus_sqrt -> topk_hash_softplus_sqrt"""
        if self._vllm_ops is None:
            return None, {}
        # 函数名不同，可选参数名不同，但位置一致
        return self._vllm_ops.topk_hash_softplus_sqrt, {
            "source": "vllm._custom_ops.topk_hash_softplus_sqrt",
            "type": "cuda",
            "note": "vLLM function name: topk_hash_softplus_sqrt"
        }

    def _load_flash_attn_varlen_func(self):
        """标准 Flash Attention varlen 版本"""
        if self._vllm_flash_attn is None:
            return None, {}
        return self._vllm_flash_attn.flash_attn_varlen_func, {
            "source": "vllm.vllm_flash_attn.flash_attn_interface.flash_attn_varlen_func",
            "type": "cuda"
        }

    # ============================================================
    # 签名兼容（位置参数一致，仅参数名差异）
    # ============================================================
    # ============================================================

    def _load_indexer_k_quant_and_cache(self):
        """参数名转换: scale_fmt -> kv_cache_dtype"""
        if self._vllm_ops is None:
            return None, {}

        vllm_fn = self._vllm_ops.indexer_k_quant_and_cache

        def wrapper(k, kv_cache, slot_mapping, quant_block_size, scale_fmt):
            return vllm_fn(k, kv_cache, slot_mapping, quant_block_size, kv_cache_dtype=scale_fmt)

        return wrapper, {
            "source": "vllm._custom_ops.indexer_k_quant_and_cache (adapted)",
            "type": "cuda"
        }

    def _load_cp_gather_indexer(self):
        """位置参数一致 (仅参数名差异: k_cache->kv_cache, k_fp8->dst_k 等)"""
        if self._vllm_ops is None:
            return None, {}
        return self._vllm_ops.cp_gather_indexer_k_quant_cache, {
            "source": "vllm._custom_ops.cp_gather_indexer_k_quant_cache",
            "type": "cuda"
        }

    # ============================================================
    # 需要适配器（3个）— 签名或调用约定不同
    # ============================================================

    def _load_swiglu(self):
        """flaggems: swiglu(input_tensor) -> output; vllm: silu_and_mul(result, input) 需预分配"""
        if not self._torch_ops_registered:
            return None, {}

        vllm_fn = torch.ops._C.silu_and_mul

        def wrapper(input_tensor, **kwargs):
            # input_tensor: (M, N*2) 拼接的 [gate;up]
            # vllm 需要预分配 output
            d = input_tensor.shape[-1] // 2
            output_shape = input_tensor.shape[:-1] + (d,)
            out = torch.empty(output_shape, dtype=input_tensor.dtype, device=input_tensor.device)
            vllm_fn(out, input_tensor)
            return out

        return wrapper, {
            "source": "torch.ops._C.silu_and_mul (adapted)",
            "type": "cuda"
        }

    def _load_silu_and_mul_with_clamp(self):
        """flaggems: (x, y, limit) 分开; vllm: (result, input, limit) 拼接+预分配"""
        if not self._torch_ops_registered:
            return None, {}

        vllm_fn = torch.ops._C.silu_and_mul_with_clamp

        def wrapper(x, y, limit):
            # flaggems: x, y 分开
            # vllm: 需要拼接成 [x;y]，预分配 output
            input_tensor = torch.cat([x, y], dim=-1)
            out = torch.empty_like(x)
            vllm_fn(out, input_tensor, limit)
            return out

        return wrapper, {
            "source": "torch.ops._C.silu_and_mul_with_clamp (adapted)",
            "type": "cuda"
        }

    def _load_fused_moe(self):
        """vllm.outplace_fused_experts 签名与 flaggems_vllm.outplace_fused_experts 兼容"""
        if self._vllm_fused_moe is None:
            return None, {}

        return self._vllm_fused_moe.outplace_fused_experts, {
            "source": "vllm.model_executor.layers.fused_moe.fused_moe.outplace_fused_experts",
            "type": "triton"
        }

    def _load_fp8_fp4_paged_mqa_logits(self):
        """签名一致: q 已经是 tuple[Tensor, Tensor|None]"""
        if self._vllm_sparse_attn is None:
            return None, {}
        return self._vllm_sparse_attn.fp8_fp4_paged_mqa_logits, {
            "source": "vllm.model_executor.layers.sparse_attn_indexer.fp8_fp4_paged_mqa_logits",
            "type": "cuda"
        }

    def _load_flash_mla_with_kvcache_fp8(self):
        """flaggems 通过 flash_mla_with_kvcache(..., is_fp8_kvcache=True); vllm 是独立函数"""
        if self._vllm_v1_ops is None:
            return None, {}

        vllm_fn = self._vllm_v1_ops['flashmla'].flash_mla_with_kvcache_fp8

        def wrapper(q, k_cache, block_table, cache_seqlens, head_dim_v, tile_scheduler_metadata,
                   num_splits=None, softmax_scale=None, causal=False, **kwargs):
            # flaggems: tile_scheduler_metadata 是 FlashMLASchedMeta 对象
            # vllm: 需要提取其中的 tensor
            if hasattr(tile_scheduler_metadata, 'tile_scheduler_metadata'):
                meta_tensor = tile_scheduler_metadata.tile_scheduler_metadata
                splits_tensor = tile_scheduler_metadata.num_splits
            else:
                # 如果已经是 tensor，直接使用
                meta_tensor = tile_scheduler_metadata
                splits_tensor = num_splits

            return vllm_fn(q, k_cache, block_table, cache_seqlens, head_dim_v,
                          meta_tensor, splits_tensor,
                          softmax_scale=softmax_scale, causal=causal,
                          descale_q=None, descale_k=None)

        return wrapper, {
            "source": "vllm.v1.attention.ops.flashmla.flash_mla_with_kvcache_fp8 (adapted)",
            "type": "cuda"
        }

    def _load_grouped_topk(self):
        """flaggems: n_group; vllm: num_expert_group — 参数名不同"""
        if self._vllm_ops is None:
            return None, {}

        vllm_fn = self._vllm_ops.grouped_topk

        def wrapper(scores, n_group, topk_group, topk, renormalize,
                    routed_scaling_factor, bias, scoring_func=0, **kwargs):
            return vllm_fn(scores, num_expert_group=n_group,
                          topk_group=topk_group, topk=topk,
                          renormalize=renormalize,
                          routed_scaling_factor=routed_scaling_factor,
                          bias=bias, scoring_func=scoring_func)

        return wrapper, {
            "source": "vllm._custom_ops.grouped_topk (adapted)",
            "type": "cuda"
        }

    def _load_group_gemm(self):
        """torch._grouped_mm — 签名与 flag_gems.group_mm(A, B, offs) 一致"""
        def wrapper(A, B, offs):
            return torch._grouped_mm(A, B, offs)

        return wrapper, {
            "source": "torch._grouped_mm (CUTLASS)",
            "type": "cutlass"
        }

    def _load_chunk_gated_delta_rule(self):
        """Baseline: flaggems_vllm non-TLE fallback path

        通过 output_final_state=False 强制走 flaggems_vllm 的 non-TLE 路径
        (即普通 chunk_gated_delta_rule_fwd)，作为 TLE 优化前的 baseline。
        """
        try:
            from flaggems_vllm.ops.chunk_gated_delta_rule import (
                chunk_gated_delta_rule as fgv_fn,
            )
        except ImportError:
            return None, {"error": "Cannot import flaggems_vllm.ops.chunk_gated_delta_rule"}

        def wrapper(q, k, v, beta, g, BT=64, initial_state=None,
                    output_final_state=True, cu_seqlens=None,
                    head_first=False, scale=None,
                    use_qk_l2norm_in_kernel=False, **kwargs):
            # 强制 output_final_state=False → 绕过 TLE 路径，走 non-TLE fallback
            return fgv_fn(
                q=q, k=k, v=v, beta=beta, g=g,
                BT=BT,
                initial_state=initial_state,
                output_final_state=False,
                cu_seqlens=cu_seqlens,
                head_first=head_first,
                scale=scale,
                use_qk_l2norm_in_kernel=use_qk_l2norm_in_kernel,
            )

        return wrapper, {
            "source": "flaggems_vllm non-TLE fallback (output_final_state=False)",
            "type": "triton"
        }
