"""Provider抽象基类"""
from abc import ABC, abstractmethod
from typing import Tuple, Callable, Dict, Any, Optional

import torch

from framework.base_operator import BaseOperator


class BaseProvider(ABC):
    """算子实现来源的抽象基类

    负责根据算子名，返回可调用的实现函数。
    每个Provider代表一个kernel来源（vllm、flagos、平台SDK等）。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """Provider名称: nvidia / ascend / metax / flagos / ..."""
        ...

    @property
    def platform(self) -> str:
        """Provider所属平台: nvidia / ascend / metax / mthreads / iluvatar

        FlagOS等跨平台Provider返回 'all'。
        默认返回 'nvidia' 保持向后兼容。
        """
        return "nvidia"

    @abstractmethod
    def setup(self):
        """初始化环境（延迟import）"""
        ...

    @abstractmethod
    def get_impl(
        self,
        op_name: str,
        operator: BaseOperator
    ) -> Tuple[Optional[Callable], Dict[str, str]]:
        """获取算子实现

        Args:
            op_name: 算子名
            operator: 算子对象（用于获取library等信息）

        Returns:
            (impl_fn, impl_info)
            impl_fn: 可调用函数，签名 fn(**inputs) -> output
            impl_info: 实现信息字典 {"source": "...", "type": "triton/cuda/pytorch"}
            如果该Provider没有该算子实现，返回 (None, {})
        """
        ...

    def get_device(self) -> torch.device:
        """返回该Provider使用的设备

        子类应覆盖此方法以支持不同平台。
        默认返回 cuda:0。
        """
        return torch.device("cuda:0")

    def synchronize(self) -> None:
        """同步设备，确保所有kernel执行完毕

        子类应覆盖此方法以支持不同平台。
        默认调用 torch.cuda.synchronize()。
        """
        torch.cuda.synchronize()

    def is_available(self) -> bool:
        """检测当前环境是否能使用该Provider

        子类应覆盖此方法，检测依赖是否安装、设备是否可用。
        默认返回 True。
        """
        return True

    def teardown(self):
        """清理环境（可选）"""
        pass
