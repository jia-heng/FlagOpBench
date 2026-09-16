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
        # WORKFLOW_VENDOR 2.3：不要仅因 import vllm 成功就假定 torch.ops._C 可用
        self._torch_ops_registered = hasattr(torch.ops, "_C") and hasattr(
            torch.ops._C, "silu_and_mul"
        )

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
