"""Ascend Provider (预留)

昇腾平台最优性能基线Provider。
优先使用vllm-ascend kernels，fallback到torch (torch_npu注册的aten ops)。

后续在昇腾平台环境下继续开发。
"""
from typing import Tuple, Callable, Dict, Optional

import torch

from framework.base_operator import BaseOperator
from .base_provider import BaseProvider
from .registry import register_provider


@register_provider("ascend", platform="ascend", is_default=True)
class AscendProvider(BaseProvider):
    """昇腾平台算子实现加载器（vllm-ascend优先，torch_npu fallback）

    预留接口，后续在昇腾平台开发。
    """

    def __init__(self):
        self._vllm = None
        self._torch_npu = None

    @property
    def name(self) -> str:
        return "ascend"

    @property
    def platform(self) -> str:
        return "ascend"

    def get_device(self) -> torch.device:
        """返回NPU设备"""
        return torch.device("npu:0")

    def synchronize(self) -> None:
        """NPU同步"""
        try:
            import torch_npu
            torch.npu.synchronize()
        except (ImportError, AttributeError):
            pass

    def is_available(self) -> bool:
        """检测昇腾环境是否可用"""
        try:
            import torch_npu
            return torch.npu.is_available()
        except ImportError:
            return False

    def setup(self):
        """延迟import昇腾相关库"""
        try:
            import torch_npu
            self._torch_npu = torch_npu
            print(f"  Loaded torch_npu: {torch_npu.__version__}")
        except ImportError as e:
            print(f"  [WARN] Failed to import torch_npu: {e}")

        try:
            import vllm
            self._vllm = vllm
            print(f"  Loaded vllm-ascend: {vllm.__version__}")
        except ImportError as e:
            print(f"  [WARN] Failed to import vllm (ascend): {e}")

    def get_impl(
        self,
        op_name: str,
        operator: BaseOperator
    ) -> Tuple[Optional[Callable], Dict[str, str]]:
        """获取昇腾算子实现

        TODO: 后续在昇腾平台开发，需要:
        1. 通过 op_name_mapping.yaml 查找平台算子注册名
        2. 优先从 vllm-ascend 加载
        3. fallback 到 torch_npu 注册的 aten ops
        """
        # 预留: 后续实现
        return None, {
            "error": f"AscendProvider: {op_name} not implemented yet. "
                     f"需要在昇腾平台环境下开发。"
        }

    def teardown(self):
        pass
