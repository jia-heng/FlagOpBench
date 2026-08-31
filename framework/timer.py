"""GPU计时工具

支持多平台的kernel耗时测量。
- CudaTimer: 使用CUDA events (nvidia)
- AscendTimer: 使用torch_npu同步 (预留)
- MetaxTimer: 使用torch_metax同步 (预留)
"""
import torch
from typing import Callable, Dict, List, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass
class TimingResult:
    """计时结果"""
    mean_ms: float
    std_ms: float
    min_ms: float
    max_ms: float
    latencies_ms: List[float]


class BaseTimer(ABC):
    """计时器抽象基类"""

    def __init__(self, warmup: int = 10, repeat: int = 100):
        self.warmup = warmup
        self.repeat = repeat

    @abstractmethod
    def measure(
        self,
        fn: Callable,
        inputs: Dict[str, Any],
    ) -> TimingResult:
        """测量算子执行时间

        Args:
            fn: 被测函数
            inputs: 输入字典

        Returns:
            TimingResult
        """
        ...

    @staticmethod
    def _std(values: List[float]) -> float:
        if len(values) <= 1:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        return variance ** 0.5


class CudaTimer(BaseTimer):
    """CUDA计时器，使用cuda events"""

    def measure(
        self,
        fn: Callable,
        inputs: Dict[str, Any],
    ) -> TimingResult:
        # warmup
        for _ in range(self.warmup):
            fn(**inputs)
        torch.cuda.synchronize()

        # measure
        latencies = []
        for _ in range(self.repeat):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)

            start.record()
            fn(**inputs)
            end.record()

            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))

        return TimingResult(
            mean_ms=sum(latencies) / len(latencies),
            std_ms=self._std(latencies),
            min_ms=min(latencies),
            max_ms=max(latencies),
            latencies_ms=latencies,
        )


class AscendTimer(BaseTimer):
    """昇腾计时器（预留，后续在昇腾平台开发）"""

    def measure(
        self,
        fn: Callable,
        inputs: Dict[str, Any],
    ) -> TimingResult:
        raise NotImplementedError(
            "AscendTimer: 需要在昇腾平台环境下实现，"
            "使用 torch_npu 同步机制进行计时"
        )


class MetaxTimer(CudaTimer):
    """沐曦计时器

    MACA 复用 torch.cuda 接口，torch.cuda.Event(enable_timing=True) 与
    torch.cuda.synchronize() 在沐曦平台可用，因此计时逻辑与 CudaTimer 一致。
    """
    pass


# 保持向后兼容: Timer 作为 CudaTimer 的别名
Timer = CudaTimer


def create_timer(platform: str, warmup: int = 10, repeat: int = 100) -> BaseTimer:
    """根据平台创建对应的计时器

    Args:
        platform: nvidia / ascend / metax / mthreads / iluvatar
        warmup: 预热次数
        repeat: 重复测试次数

    Returns:
        对应平台的Timer实例
    """
    timer_map = {
        "nvidia": CudaTimer,
        "ascend": AscendTimer,
        "metax": MetaxTimer,
        "mthreads": MetaxTimer,  # 预留，后续替换为专用Timer
        "iluvatar": CudaTimer,   # 天数智芯兼容CUDA接口
    }

    timer_cls = timer_map.get(platform)
    if timer_cls is None:
        raise ValueError(
            f"不支持的平台: {platform}，"
            f"可选: {list(timer_map.keys())}"
        )

    return timer_cls(warmup=warmup, repeat=repeat)
