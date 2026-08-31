"""MetaX Provider

沐曦平台性能基线Provider。

vllm-metax 是上游 vLLM 的 fork，模块路径与 NV 一致（vllm._custom_ops、
vllm.model_executor.layers.* 等），因此直接复用 NvidiaProvider 的 impl_map
与全部 _load_xxx 加载方法。NvidiaProvider.setup() 中每个子模块都是独立
try/except，vllm-metax 未包含的模块会保持为 None，对应算子在 Runner 侧
自动 [SKIP]，从而暴露该平台的算子覆盖缺口。
"""
import torch

from .nvidia_provider import NvidiaProvider
from .registry import register_provider


@register_provider("metax", platform="metax", is_default=True)
class MetaxProvider(NvidiaProvider):
    """沐曦平台算子实现加载器（vllm-metax 优先，torch fallback）"""

    @property
    def name(self) -> str:
        return "metax"

    @property
    def platform(self) -> str:
        return "metax"

    def get_device(self) -> torch.device:
        # MACA 复用 torch.cuda 接口，设备选择由 MACA_VISIBLE_DEVICES 控制
        return torch.device("cuda:0")

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def is_available(self) -> bool:
        # 沐曦 MACA 栈通过 torch.cuda 暴露设备，torch_metax 并非所有镜像都提供，
        # 因此以 torch.cuda 是否可用为准。
        return torch.cuda.is_available()

    def setup(self):
        try:
            import torch_metax  # noqa: F401
            print("  Loaded torch_metax")
        except ImportError:
            print("  [INFO] torch_metax not present, using torch.cuda interface")

        super().setup()

    def get_impl(self, op_name, operator):
        impl_fn, impl_info = super().get_impl(op_name, operator)
        # source 沿用 NV 的模块路径命名，这里标注实际来源平台，避免报告中混淆
        if impl_fn is not None:
            impl_info = {**impl_info, "platform": "metax"}
        return impl_fn, impl_info
