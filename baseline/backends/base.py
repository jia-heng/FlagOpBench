"""后端抽象基类"""

from abc import ABC, abstractmethod
from baseline.framework.timer import BenchmarkTimer


class Backend(ABC):
    """后端抽象基类

    各平台（NVIDIA、昇腾、沐曦等）需实现此接口，
    提供设备管理、计时器创建、算子获取等能力。
    """

    @property
    @abstractmethod
    def name(self) -> str:
        """后端标识，如 'nvidia', 'ascend', 'muxin'"""
        ...

    @property
    @abstractmethod
    def platform_name(self) -> str:
        """平台全称，如 'NVIDIA H20'"""
        ...

    @abstractmethod
    def get_device(self):
        """返回 torch device"""
        ...

    @abstractmethod
    def synchronize(self):
        """设备同步"""
        ...

    @abstractmethod
    def create_timer(self, warmup: int = 10, iters: int = 100) -> BenchmarkTimer:
        """创建平台适配的计时器"""
        ...

    @abstractmethod
    def get_operator(self, op_name: str):
        """获取算子实现

        Args:
            op_name: 算子名称

        Returns:
            算子实例（实现了 BaseOperator 接口）
        """
        ...

    def is_dtype_supported(self, dtype: str) -> bool:
        """检查是否支持指定 dtype

        默认支持常见类型，后端可覆盖此方法。
        """
        return dtype in ("fp32", "float32", "fp16", "float16", "bf16", "bfloat16")

    def collect_env_info(self) -> dict:
        """采集平台环境信息"""
        from baseline.framework.env_collector import EnvCollector
        return EnvCollector().collect()

    def setup(self):
        """后端初始化（可选覆盖）"""
        pass

    def teardown(self):
        """后端清理（可选覆盖）"""
        pass
