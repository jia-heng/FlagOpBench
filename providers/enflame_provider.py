"""Enflame Provider

燧原 GCU 平台性能基线 Provider。

设备走 torch.gcu；对位栈为 vllm_gcu。
算子 case 里 tensor 默认 device=cuda，setup 时将
torch 工厂函数的 cuda 设备重定向到 gcu:0。

swiglu 优先 torch.ops._C.silu_and_mul；无则 F.silu * mul 兜底。
"""
import torch
import torch.nn.functional as F

from .nvidia_provider import NvidiaProvider
from .registry import register_provider


def _patch_tensor_factory_for_gcu() -> None:
    """将 operators 里 device=cuda 的创建重定向到 gcu:0。"""
    if getattr(torch, "_flagopbench_gcu_patch", False):
        return

    gcu_dev = torch.device("gcu:0")

    def _fix_device(kwargs: dict) -> dict:
        dev = kwargs.get("device")
        if dev is None:
            return kwargs
        dev_str = str(dev)
        if dev == "cuda" or dev_str == "cuda" or dev_str.startswith("cuda:"):
            return {**kwargs, "device": gcu_dev}
        return kwargs

    for name in ("randn", "empty", "zeros", "ones", "full", "arange", "tensor"):
        if not hasattr(torch, name):
            continue
        orig = getattr(torch, name)

        def make_wrapper(fn):
            def wrapper(*args, **kwargs):
                return fn(*args, **_fix_device(kwargs))
            wrapper.__name__ = getattr(fn, "__name__", name)
            return wrapper

        setattr(torch, name, make_wrapper(orig))

    torch._flagopbench_gcu_patch = True


@register_provider("enflame", platform="enflame", is_default=True)
class EnflameProvider(NvidiaProvider):
    """燧原 GCU 算子实现加载器（vllm_gcu 优先，torch 兜底）"""

    @property
    def name(self) -> str:
        return "enflame"

    @property
    def platform(self) -> str:
        return "enflame"

    def get_device(self) -> torch.device:
        return torch.device("gcu:0")

    def synchronize(self) -> None:
        torch.gcu.synchronize()

    def is_available(self) -> bool:
        return hasattr(torch, "gcu") and torch.gcu.is_available()

    def setup(self):
        if not self.is_available():
            print("  [WARN] torch.gcu.is_available()=False; "
                  "check ENFLAME_VISIBLE_DEVICES / gcu driver / vllm_gcu image")
            return

        _patch_tensor_factory_for_gcu()
        print(f"  Loaded torch.gcu: available=True, "
              f"count={torch.gcu.device_count()}, "
              f"name0={torch.gcu.get_device_name(0)}")

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
            "platform": "enflame",
        }

    def get_impl(self, op_name, operator):
        impl_fn, impl_info = super().get_impl(op_name, operator)
        if impl_fn is not None:
            impl_info = {**impl_info, "platform": "enflame"}
        return impl_fn, impl_info
