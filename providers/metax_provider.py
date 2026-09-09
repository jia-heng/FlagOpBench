"""MetaX Provider

沐曦平台性能基线 Provider。

MACA 栈通过 torch.cuda 暴露设备；容器内常无 torch_metax 包。
vllm-metax 是上游 vLLM 的 fork，模块路径与 NV 一致，复用 NvidiaProvider
的 impl_map。NvidiaProvider.setup() 中各子模块独立 try/except，
缺模块保持为 None，对应算子自动 [SKIP]。
"""
import torch

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
            print("  [WARN] torch.cuda.is_available()=False; "
                  "unset CUDA_VISIBLE_DEVICES and set MACA_VISIBLE_DEVICES=6,7")

        super().setup()

    def get_impl(self, op_name, operator):
        impl_fn, impl_info = super().get_impl(op_name, operator)
        if impl_fn is not None:
            impl_info = {**impl_info, "platform": "metax"}
        return impl_fn, impl_info
