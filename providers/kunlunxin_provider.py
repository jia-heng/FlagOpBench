"""Kunlunxin Provider

昆仑芯 XPU / P800 平台性能基线 Provider。

xvllm 容器内设备常通过 torch.cuda 暴露（torch_xmlir），
并非 torch.xpu。vllm 子模块导入可能循环导入 / AssertionError，
需宽捕获。Event.elapsed_time 在本栈上常恒为 0，计时用 KunlunxinTimer。
"""
import torch
import torch.nn.functional as F

from .nvidia_provider import NvidiaProvider
from .registry import register_provider


@register_provider("kunlunxin", platform="kunlunxin", is_default=True)
class KunlunxinProvider(NvidiaProvider):
    """昆仑芯平台算子实现加载器（xvllm 优先，torch 兜底）"""

    @property
    def name(self) -> str:
        return "kunlunxin"

    @property
    def platform(self) -> str:
        return "kunlunxin"

    def get_device(self) -> torch.device:
        return torch.device("cuda:0")

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def setup(self):
        if not torch.cuda.is_available():
            print("  [WARN] torch.cuda.is_available()=False; "
                  "check XPU_VISIBLE_DEVICES / xvllm image")
            return

        print(f"  Loaded torch.cuda: available=True, "
              f"count={torch.cuda.device_count()}, "
              f"name0={torch.cuda.get_device_name(0)}")

        try:
            import vllm
            self._vllm = vllm
            print(f"  Loaded vllm: {vllm.__version__ if hasattr(vllm, '__version__') else 'unknown'}")
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
            "platform": "kunlunxin",
        }

    def get_impl(self, op_name, operator):
        impl_fn, impl_info = super().get_impl(op_name, operator)
        if impl_fn is not None:
            impl_info = {**impl_info, "platform": "kunlunxin"}
        return impl_fn, impl_info
