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
    if getattr(torch, "_flagopbench_npu_patch", False):
        return

    npu_dev = torch.device("npu:0")

    def _fix_device(kwargs: dict) -> dict:
        dev = kwargs.get("device")
        if dev is None:
            return kwargs
        dev_str = str(dev)
        if dev == "cuda" or dev_str == "cuda" or dev_str.startswith("cuda:"):
            return {**kwargs, "device": npu_dev}
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

    def get_impl(self, op_name, operator):
        impl_fn, impl_info = super().get_impl(op_name, operator)
        if impl_fn is not None:
            impl_info = {**impl_info, "platform": "ascend"}
        return impl_fn, impl_info
