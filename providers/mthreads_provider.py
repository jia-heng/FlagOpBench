"""Moore Threads Provider

摩尔线程平台性能基线 Provider。

设备走 torch.musa；vllm-musa 可与 vllm 并存。
算子 case 里 tensor 默认 device=cuda，setup 时将
torch 工厂函数的 cuda 设备重定向到 musa:0。

swiglu 优先 torch.ops._C.silu_and_mul；无则 F.silu * mul 兜底。
"""
import torch
import torch.nn.functional as F

from .nvidia_provider import NvidiaProvider
from .registry import register_provider


def _patch_tensor_factory_for_musa() -> None:
    """将 operators 里 device=cuda 的创建重定向到 musa:0。"""
    if getattr(torch, "_flagopbench_musa_patch", False):
        return

    musa_dev = torch.device("musa:0")

    def _fix_device(kwargs: dict) -> dict:
        dev = kwargs.get("device")
        if dev is None:
            return kwargs
        dev_str = str(dev)
        if dev == "cuda" or dev_str == "cuda" or dev_str.startswith("cuda:"):
            return {**kwargs, "device": musa_dev}
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

    torch._flagopbench_musa_patch = True


@register_provider("mthreads", platform="mthreads", is_default=True)
class MthreadsProvider(NvidiaProvider):
    """摩尔线程平台算子实现加载器（vllm-musa 优先，torch 兜底）"""

    @property
    def name(self) -> str:
        return "mthreads"

    @property
    def platform(self) -> str:
        return "mthreads"

    def get_device(self) -> torch.device:
        return torch.device("musa:0")

    def synchronize(self) -> None:
        torch.musa.synchronize()

    def is_available(self) -> bool:
        return hasattr(torch, "musa") and torch.musa.is_available()

    def setup(self):
        if not self.is_available():
            print("  [WARN] torch.musa.is_available()=False; "
                  "check MUSA_VISIBLE_DEVICES / driver / vllm-musa image")
            return

        _patch_tensor_factory_for_musa()
        print(f"  Loaded torch.musa: available=True, "
              f"count={torch.musa.device_count()}, "
              f"name0={torch.musa.get_device_name(0)}")

        try:
            import vllm_musa  # noqa: F401
            print(f"  Loaded vllm_musa: {getattr(vllm_musa, '__file__', 'ok')}")
        except Exception as e:
            print(f"  [WARN] Failed to import vllm_musa: {type(e).__name__}: {e}")

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

        # 摩尔旧记录：vllm_musa 侧可能有 _musa_custom_ops
        if self._vllm_ops is None:
            try:
                import vllm_musa
                musa_ops = getattr(vllm_musa, "_musa_custom_ops", None)
                if musa_ops is not None:
                    self._vllm_ops = musa_ops
                    print("  Loaded vllm_musa._musa_custom_ops as custom_ops")
            except Exception as e:
                print(f"  [WARN] Failed to import _musa_custom_ops: {type(e).__name__}: {e}")

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
            "platform": "mthreads",
        }

    def _load_silu_and_mul_with_clamp(self):
        if hasattr(torch.ops, "_C") and hasattr(torch.ops._C, "silu_and_mul_with_clamp"):
            impl_fn, info = super()._load_silu_and_mul_with_clamp()
            if impl_fn is not None:
                return impl_fn, info

        def wrapper(x, y, limit):
            # 与 operators/silu_and_mul_with_clamp 文档一致
            return torch.clamp(F.silu(x) * y, -limit, limit)

        return wrapper, {
            "source": "torch.clamp(F.silu(x)*y, ±limit) (fallback)",
            "type": "torch",
            "platform": "mthreads",
        }

    def _load_moe_sum(self):
        """摩尔上 _custom_ops.moe_sum 常指向空壳 _moe_C，调用才报错。

        签名与 operator 一致: input (T,K,H) -> output (T,H) 原地写入。
        """
        def wrapper(input, output):
            torch.sum(input, dim=1, out=output)
            return output

        return wrapper, {
            "source": "torch.sum(dim=1, out=...) (fallback)",
            "type": "torch",
            "platform": "mthreads",
        }

    def _load_group_gemm(self):
        """摩尔 torch 无可用 _grouped_mm（hasattr 偶发不可信），固定用循环 mm 兜底。

        C[start:end] = A[start:end] @ B[i]，offs 为累积行偏移。
        """
        def wrapper(A, B, offs):
            num_groups = int(offs.shape[0])
            N = B.shape[-1]
            C = torch.empty(A.shape[0], N, dtype=A.dtype, device=A.device)
            start = 0
            for i in range(num_groups):
                end = int(offs[i].item())
                C[start:end] = torch.mm(A[start:end], B[i])
                start = end
            return C

        return wrapper, {
            "source": "torch.mm loop over groups (fallback)",
            "type": "torch",
            "platform": "mthreads",
        }

    def get_impl(self, op_name, operator):
        impl_fn, impl_info = super().get_impl(op_name, operator)
        if impl_fn is not None:
            impl_info = {**impl_info, "platform": "mthreads"}
        return impl_fn, impl_info
