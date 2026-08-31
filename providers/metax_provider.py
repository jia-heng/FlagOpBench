"""MetaX Provider (预留)

沐曦平台最优性能基线Provider。
优先使用vllm-metax kernels，fallback到torch (torch_metax注册的aten ops)。

后续在沐曦平台环境下继续开发。
"""
from typing import Tuple, Callable, Dict, Optional

import torch

from framework.base_operator import BaseOperator
from .base_provider import BaseProvider
from .registry import register_provider


@register_provider("metax", platform="metax", is_default=True)
class MetaxProvider(BaseProvider):
    """沐曦平台算子实现加载器（vllm-metax优先，torch_metax fallback）

    预留接口，后续在沐曦平台开发。
    """

    def __init__(self):
        self._vllm = None

    @property
    def name(self) -> str:
        return "metax"

    @property
    def platform(self) -> str:
        return "metax"

    def get_device(self) -> torch.device:
        return torch.device("cuda:0")  # metax 通常兼容 cuda 接口

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def is_available(self) -> bool:
        try:
            import torch_metax
            return True
        except ImportError:
            return False

    def setup(self):
        try:
            import torch_metax
            print(f"  Loaded torch_metax")
        except ImportError as e:
            print(f"  [WARN] Failed to import torch_metax: {e}")

        try:
            import vllm
            self._vllm = vllm
            print(f"  Loaded vllm-metax: {vllm.__version__}")
        except ImportError as e:
            print(f"  [WARN] Failed to import vllm (metax): {e}")

    def get_impl(
        self,
        op_name: str,
        operator: BaseOperator
    ) -> Tuple[Optional[Callable], Dict[str, str]]:
        """预留: 后续在沐曦平台实现"""
        return None, {
            "error": f"MetaxProvider: {op_name} not implemented yet."
        }

    def teardown(self):
        pass
