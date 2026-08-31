"""Iluvatar Provider (预留)

天数智芯平台最优性能基线Provider。
天数智芯兼容CUDA接口，优先使用vllm kernels。

后续在天数智芯平台环境下继续开发。
"""
from typing import Tuple, Callable, Dict, Optional

import torch

from framework.base_operator import BaseOperator
from .base_provider import BaseProvider
from .registry import register_provider


@register_provider("iluvatar", platform="iluvatar", is_default=True)
class IluvatarProvider(BaseProvider):
    """天数智芯平台算子实现加载器

    预留接口，后续在天数智芯平台开发。
    天数智芯兼容CUDA接口。
    """

    def __init__(self):
        self._vllm = None

    @property
    def name(self) -> str:
        return "iluvatar"

    @property
    def platform(self) -> str:
        return "iluvatar"

    def get_device(self) -> torch.device:
        return torch.device("cuda:0")  # 兼容CUDA接口

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def is_available(self) -> bool:
        # TODO: 检测天数智芯环境
        return False

    def setup(self):
        try:
            import vllm
            self._vllm = vllm
            print(f"  Loaded vllm-iluvatar: {vllm.__version__}")
        except ImportError as e:
            print(f"  [WARN] Failed to import vllm (iluvatar): {e}")

    def get_impl(
        self,
        op_name: str,
        operator: BaseOperator
    ) -> Tuple[Optional[Callable], Dict[str, str]]:
        """预留: 后续在天数智芯平台实现"""
        return None, {
            "error": f"IluvatarProvider: {op_name} not implemented yet."
        }

    def teardown(self):
        pass
