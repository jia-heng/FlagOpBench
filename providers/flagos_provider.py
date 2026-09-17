"""FlagOS Provider

FlagOS跨平台算子实现加载器。
加载FlagGems/FlagGems-vllm/FlagAttention，支持所有平台。
"""
from typing import Tuple, Callable, Dict, Optional

import torch

from framework.base_operator import BaseOperator
from .base_provider import BaseProvider
from .registry import register_provider


def _detect_accelerator() -> str:
    """按当前环境探测加速器后端: musa / npu / gcu / cuda / cpu"""
    if hasattr(torch, "musa"):
        try:
            if torch.musa.is_available():
                return "musa"
        except Exception:
            pass
    if hasattr(torch, "npu"):
        try:
            if torch.npu.is_available():
                return "npu"
        except Exception:
            pass
    if hasattr(torch, "gcu"):
        try:
            if torch.gcu.is_available():
                return "gcu"
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@register_provider("flagos", platform="all")
class FlagOSProvider(BaseProvider):
    """FlagOS算子实现加载器（跨平台，被测对象）"""

    def __init__(self):
        self._flaggems = None
        self._flaggems_vllm = None
        self._flagattention = None
        self._accel = _detect_accelerator()

    @property
    def name(self) -> str:
        return "flagos"

    @property
    def platform(self) -> str:
        return "all"

    def get_device(self) -> torch.device:
        """FlagOS支持多平台，根据当前环境返回设备"""
        accel = self._accel or _detect_accelerator()
        if accel == "musa":
            return torch.device("musa:0")
        if accel == "npu":
            return torch.device("npu:0")
        if accel == "gcu":
            return torch.device("gcu:0")
        if accel == "cuda":
            return torch.device("cuda:0")
        return torch.device("cpu")

    def synchronize(self) -> None:
        """同步当前设备"""
        accel = self._accel or _detect_accelerator()
        if accel == "musa":
            torch.musa.synchronize()
        elif accel == "npu":
            torch.npu.synchronize()
        elif accel == "gcu":
            torch.gcu.synchronize()
        elif accel == "cuda":
            torch.cuda.synchronize()

    def is_available(self) -> bool:
        """检查FlagOS相关库是否可用"""
        try:
            import flag_gems  # noqa: F401
            return True
        except ImportError:
            pass
        try:
            import flaggems_vllm  # noqa: F401
            return True
        except ImportError:
            pass
        try:
            import flag_attn  # noqa: F401
            return True
        except ImportError:
            pass
        return False

    def setup(self):
        """延迟import FlagOS相关库，并按加速器重定向 device=cuda 的工厂函数"""
        self._accel = _detect_accelerator()
        self._apply_device_redirect_patches()
        print(f"  FlagOS accelerator: {self._accel}, device={self.get_device()}")

        try:
            import flag_gems
            self._flaggems = flag_gems
            print(f"  Loaded flag_gems: {flag_gems.__version__ if hasattr(flag_gems, '__version__') else 'unknown'}")
        except ImportError as e:
            print(f"  [WARN] Failed to import flag_gems: {e}")

        try:
            import flaggems_vllm
            self._flaggems_vllm = flaggems_vllm
            print(f"  Loaded flaggems_vllm")
        except ImportError as e:
            print(f"  [WARN] Failed to import flaggems_vllm: {e}")

        try:
            import flag_attn as flagattention
            self._flagattention = flagattention
            print(f"  Loaded flagattention (flag_attn): {flagattention.__version__ if hasattr(flagattention, '__version__') else 'unknown'}")
        except ImportError as e:
            print(f"  [WARN] Failed to import flagattention: {e}")

    def _apply_device_redirect_patches(self) -> None:
        """算子 prepare_inputs 常写 device=cuda，国产后端需重定向。"""
        if self._accel == "musa":
            from providers.mthreads_provider import _patch_tensor_factory_for_musa
            _patch_tensor_factory_for_musa()
        elif self._accel == "npu":
            from providers.ascend_provider import _patch_tensor_factory_for_npu
            _patch_tensor_factory_for_npu()
        elif self._accel == "gcu":
            from providers.enflame_provider import _patch_tensor_factory_for_gcu
            _patch_tensor_factory_for_gcu()

    def get_impl(
        self,
        op_name: str,
        operator: BaseOperator
    ) -> Tuple[Optional[Callable], Dict[str, str]]:
        """根据算子library属性加载对应实现"""
        lib = operator.library
        # 支持算子自定义impl函数名（与注册名不同时使用）
        fn_name = getattr(operator, "impl_name", op_name)

        # Special handling for operators with parameter name differences
        if op_name == "cp_gather_indexer_k_quant_cache":
            return self._load_cp_gather_indexer_k_quant_cache()

        if op_name == "topk_softplus_sqrt":
            return self._load_topk_softplus_sqrt()

        if op_name == "indexer_k_quant_and_cache":
            return self._load_indexer_k_quant_and_cache()

        if op_name == "top_k_per_row_prefill":
            return self._load_top_k_per_row_prefill()

        if op_name == "top_k_per_row_decode":
            return self._load_top_k_per_row_decode()

        if lib in ("flaggems", "flag_gems") and self._flaggems is not None:
            if hasattr(self._flaggems, fn_name):
                fn = getattr(self._flaggems, fn_name)
                return fn, {"source": f"flag_gems.{fn_name}", "type": "triton"}

        elif lib == "flaggems_vllm" and self._flaggems_vllm is not None:
            if hasattr(self._flaggems_vllm, fn_name):
                fn = getattr(self._flaggems_vllm, fn_name)
                return fn, {"source": f"flaggems_vllm.{fn_name}", "type": "triton"}

        elif lib == "flagattention" and self._flagattention is not None:
            if hasattr(self._flagattention, fn_name):
                fn = getattr(self._flagattention, fn_name)
                return fn, {"source": f"flagattention.{fn_name}", "type": "triton"}

        # 没有找到实现
        return None, {"error": f"No impl for {fn_name} in {lib}"}

    def _load_cp_gather_indexer_k_quant_cache(self):
        """Wrapper for cp_gather_indexer_k_quant_cache - maps vLLM parameter names to FlagOS names

        vLLM signature: (kv_cache, dst_k, dst_scale, block_table, cu_seq_lens)
        FlagOS signature: (k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen)
        """
        if self._flaggems_vllm is None or not hasattr(self._flaggems_vllm, "cp_gather_indexer_k_quant_cache"):
            return None, {"error": "cp_gather_indexer_k_quant_cache not found in flaggems_vllm"}

        flagos_fn = self._flaggems_vllm.cp_gather_indexer_k_quant_cache

        def wrapper(kv_cache, dst_k, dst_scale, block_table, cu_seq_lens, **kwargs):
            # Map parameter names from vLLM to FlagOS
            return flagos_fn(
                k_cache=kv_cache,
                k_fp8=dst_k,
                k_fp8_scale=dst_scale,
                block_table=block_table,
                cu_seqlen=cu_seq_lens
            )

        return wrapper, {"source": "flaggems_vllm.cp_gather_indexer_k_quant_cache (wrapped)", "type": "triton"}

    def _load_topk_softplus_sqrt(self):
        """flaggems_vllm 常绑定 torch.ops._moe_C；沐曦镜像可能只有 vllm topk_hash_softplus_sqrt。"""
        moe_ok = hasattr(torch.ops, "_moe_C") and hasattr(
            torch.ops._moe_C, "topk_softplus_sqrt"
        )
        if not moe_ok and self._flaggems is not None and hasattr(
            self._flaggems, "topk_softplus_sqrt"
        ):
            fn = getattr(self._flaggems, "topk_softplus_sqrt")
            return fn, {
                "source": "flag_gems.topk_softplus_sqrt (fallback: _moe_C missing)",
                "type": "triton",
            }

        if self._flaggems_vllm is not None and hasattr(
            self._flaggems_vllm, "topk_softplus_sqrt"
        ):
            fn = getattr(self._flaggems_vllm, "topk_softplus_sqrt")
            return fn, {
                "source": "flaggems_vllm.topk_softplus_sqrt",
                "type": "triton",
            }

        if self._flaggems is not None and hasattr(self._flaggems, "topk_softplus_sqrt"):
            fn = getattr(self._flaggems, "topk_softplus_sqrt")
            return fn, {"source": "flag_gems.topk_softplus_sqrt", "type": "triton"}

        return None, {"error": "topk_softplus_sqrt not available in flaggems_vllm/flag_gems"}

    def _torch_op_present(self, *candidates) -> bool:
        """candidates: (namespace, op_name) pairs on torch.ops."""
        for ns_name, op_name in candidates:
            ns = getattr(torch.ops, ns_name, None)
            if ns is not None and hasattr(ns, op_name):
                return True
        return False

    def _load_indexer_k_quant_and_cache(self):
        """flaggems_vllm 常绑 torch.ops；沐曦缺 _C/_C_cache_ops 时改走 flag_gems Triton。"""
        torch_ok = self._torch_op_present(
            ("_C", "indexer_k_quant_and_cache"),
            ("_C_cache_ops", "indexer_k_quant_and_cache"),
        )
        if not torch_ok and self._flaggems is not None and hasattr(
            self._flaggems, "indexer_k_quant_and_cache"
        ):
            fn = getattr(self._flaggems, "indexer_k_quant_and_cache")
            return fn, {
                "source": "flag_gems.indexer_k_quant_and_cache (fallback: torch.ops missing)",
                "type": "triton",
            }

        if self._flaggems_vllm is not None and hasattr(
            self._flaggems_vllm, "indexer_k_quant_and_cache"
        ):
            fn = getattr(self._flaggems_vllm, "indexer_k_quant_and_cache")
            return fn, {
                "source": "flaggems_vllm.indexer_k_quant_and_cache",
                "type": "triton",
            }

        if self._flaggems is not None and hasattr(
            self._flaggems, "indexer_k_quant_and_cache"
        ):
            fn = getattr(self._flaggems, "indexer_k_quant_and_cache")
            return fn, {
                "source": "flag_gems.indexer_k_quant_and_cache",
                "type": "triton",
            }

        return None, {
            "error": "indexer_k_quant_and_cache not available in flaggems_vllm/flag_gems"
        }

    def _load_top_k_per_row_prefill(self):
        """沐曦：_metax histogram，但强制 BLOCK/CAND≤512（原 BLOCK_N/CAND_PAD=2048 超限）。"""
        import importlib

        try:
            metax_mod = importlib.import_module(
                "flag_gems.runtime.backend._metax.fused.top_k_per_row_prefill"
            )

            def wrapper(
                logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
            ):
                # 复刻 _metax.top_k_per_row_prefill，但把 arange 维度压到 ≤512
                M = num_rows
                N = logits.shape[1]
                if not logits.is_contiguous():
                    logits = logits.contiguous()
                if not row_starts.is_contiguous():
                    row_starts = row_starts.contiguous()
                if not row_ends.is_contiguous():
                    row_ends = row_ends.contiguous()

                BUCKET_BITS = 12
                BUCKETS = 1 << BUCKET_BITS
                SHIFT = 32 - BUCKET_BITS
                # C550 线程上限 512；sort 的 tl.arange(0, CAND_PAD) 必须 ≤512
                CAND_PAD = metax_mod._next_pow2(max(2 * top_k, 512))
                if CAND_PAD > 512:
                    CAND_PAD = 512
                BLOCK_N = 256
                n_blocks = (N + BLOCK_N - 1) // BLOCK_N
                dev = logits.device
                hist = torch.zeros((M, BUCKETS), dtype=torch.int32, device=dev)
                thr = torch.empty((M,), dtype=torch.int32, device=dev)
                cand = torch.zeros((M, CAND_PAD), dtype=torch.int64, device=dev)
                cand_ctr = torch.zeros((M,), dtype=torch.int32, device=dev)

                metax_mod._hist_kernel[(M, n_blocks)](
                    logits,
                    row_starts,
                    row_ends,
                    hist,
                    N,
                    logits.stride(0),
                    logits.stride(1),
                    hist.stride(0),
                    BLOCK=BLOCK_N,
                    SHIFT=SHIFT,
                    num_warps=4,
                    num_stages=2,
                )
                metax_mod._thr_kernel[(M,)](
                    hist,
                    thr,
                    top_k,
                    hist.stride(0),
                    BUCKETS=BUCKETS,
                    num_warps=4,
                )
                metax_mod._compact_kernel[(M, n_blocks)](
                    logits,
                    row_starts,
                    row_ends,
                    thr,
                    cand,
                    cand_ctr,
                    N,
                    logits.stride(0),
                    logits.stride(1),
                    cand.stride(0),
                    BLOCK=BLOCK_N,
                    SHIFT=SHIFT,
                    num_warps=4,
                    num_stages=2,
                )
                metax_mod._sort_cand_kernel[(M,)](
                    cand,
                    indices,
                    top_k,
                    cand.stride(0),
                    indices.stride(0),
                    CAND_PAD=CAND_PAD,
                    num_warps=4,
                    num_stages=2,
                )

            print("  [FlagOS] top_k_per_row_prefill source=metax_hist block=256 cand<=512")
            return wrapper, {
                "source": "flag_gems._metax.top_k_per_row_prefill (block=256,cand<=512)",
                "type": "triton",
            }
        except Exception as e:
            print(f"  [WARN] metax top_k_per_row_prefill wrap failed: {e}")

        # 禁止回退到通用 TLE/1024 路径；用 torch.topk 保双边能出表
        def torch_fallback(
            logits, row_starts, row_ends, indices, num_rows, stride0, stride1, top_k
        ):
            for i in range(num_rows):
                s = int(row_starts[i].item())
                e = int(row_ends[i].item())
                row = logits[i, s:e]
                k = min(int(top_k), row.numel())
                _, idx = torch.topk(row, k)
                indices[i, :k] = idx.to(indices.dtype)
                if k < top_k:
                    indices[i, k:] = 0

        print("  [FlagOS] top_k_per_row_prefill source=torch.topk fallback")
        return torch_fallback, {
            "source": "torch.topk (metax FlagOS fallback)",
            "type": "pytorch",
        }

    def _load_top_k_per_row_decode(self):
        """沐曦：non-TLE 启动；BLOCK 必须使 BLOCK*VEC≤512（256*4=1024 会炸）。"""
        import importlib.util
        from pathlib import Path

        try:
            import flag_gems

            src = (
                Path(flag_gems.__file__).resolve().parent
                / "fused"
                / "top_k_per_row_decode.py"
            )
            spec = importlib.util.spec_from_file_location(
                "_flaggems_top_k_per_row_decode_mod", src
            )
            decode_mod = importlib.util.module_from_spec(spec)
            assert spec.loader is not None
            spec.loader.exec_module(decode_mod)

            def wrapper(
                logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
            ):
                # 直接复刻 non-TLE 分支，不碰可能被污染的 HAS_TLE
                vocab_size = logits.shape[1]
                device = logits.device
                s_histogram_ptr = torch.empty(
                    (num_rows, decode_mod.NUM_BINS), device=device, dtype=torch.int32
                )
                s_final_logits_ptr = torch.empty(
                    (num_rows, decode_mod.NUM_FILNAL_ITEMS),
                    device=device,
                    dtype=torch.float32,
                )
                s_final_cnt_ptr = torch.empty(
                    (num_rows,), device=device, dtype=torch.int32
                )
                s_threshold_bin_idx_ptr = torch.empty(
                    (num_rows,), device=device, dtype=torch.int32
                )
                s_final_bin_size_ptr = torch.empty(
                    (num_rows,), device=device, dtype=torch.int32
                )
                s_found_topk_values_ptr = torch.empty(
                    (num_rows,), device=device, dtype=torch.int32
                )
                # VEC=4 时 [BLOCK,VEC] 决定线程数：256*4=1024>512；128*4=512
                decode_mod.non_tle_top_k_per_row_decode[(num_rows,)](
                    logits,
                    indices,
                    seq_lens,
                    next_n,
                    stride0,
                    stride1,
                    vocab_size,
                    s_histogram_ptr,
                    s_final_logits_ptr,
                    s_final_cnt_ptr,
                    s_threshold_bin_idx_ptr,
                    s_final_bin_size_ptr,
                    s_found_topk_values_ptr,
                    TOPK=top_k,
                    BLOCK_SIZE=128,
                    num_warps=4,
                )

            print("  [FlagOS] top_k_per_row_decode source=non_tle block=128")
            return wrapper, {
                "source": "flag_gems.top_k_per_row_decode (non_tle launch, block=128)",
                "type": "triton",
            }
        except Exception as e:
            print(f"  [WARN] top_k_per_row_decode non-TLE wrap failed: {e}")

        # 禁止回退到 HAS_TLE / MERGE=1024；用 torch.topk 保双边能出表
        def torch_fallback(
            logits, next_n, seq_lens, indices, num_rows, stride0, stride1, top_k
        ):
            for i in range(num_rows):
                batch_id = i // next_n
                batch_offset = i % next_n
                row_len = int(seq_lens[batch_id].item()) - next_n + batch_offset + 1
                row = logits[i, : max(row_len, 0)]
                k = min(int(top_k), row.numel()) if row.numel() > 0 else 0
                if k > 0:
                    _, idx = torch.topk(row, k)
                    indices[i, :k] = idx.to(indices.dtype)
                if k < top_k:
                    indices[i, k:] = 0

        print("  [FlagOS] top_k_per_row_decode source=torch.topk fallback")
        return torch_fallback, {
            "source": "torch.topk (metax FlagOS fallback)",
            "type": "pytorch",
        }
