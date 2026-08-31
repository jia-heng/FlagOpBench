"""Moore Threads Provider (预留)

摩尔线程平台最优性能基线Provider。
优先使用vllm-mthreads kernels，fallback到torch。

后续在摩尔线程平台环境下继续开发。
"""
from typing import Tuple, Callable, Dict, Optional

import torch

from framework.base_operator import BaseOperator
from .base_provider import BaseProvider
from .registry import register_provider


@register_provider("mthreads", platform="mthreads", is_default=True)
class MthreadsProvider(BaseProvider):
    """摩尔线程平台算子实现加载器

    预留接口，后续在摩尔线程平台开发。
    """

    def __init__(self):
        self._vllm = None

    @property
    def name(self) -> str:
        return "mthreads"

    @property
    def platform(self) -> str:
        return "mthreads"

    def get_device(self) -> torch.device:
        return torch.device("cuda:0")

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def is_available(self) -> bool:
        # TODO: 检测摩尔线程环境
        return False

    def setup(self):
        try:
            import vllm
            self._vllm = vllm
            print(f"  Loaded vllm-mthreads: {vllm.__version__}")
        except ImportError as e:
            print(f"  [WARN] Failed to import vllm (mthreads): {e}")

    def get_impl(
        self,
        op_name: str,
        operator: BaseOperator
    ) -> Tuple[Optional[Callable], Dict[str, str]]:
        """预留: 后续在摩尔线程平台实现"""
        return None, {
            "error": f"MthreadsProvider: {op_name} not implemented yet."
        }

    def teardown(self):
        pass
