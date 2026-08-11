"""双轨计时器：device event time + wall clock time"""

import time
import statistics
from abc import ABC, abstractmethod


class BenchmarkTimer(ABC):
    """双轨计时：device event time + wall clock time

    子类需实现平台相关的 sync/event 方法。
    """

    def __init__(self, warmup: int = 10, iters: int = 100):
        self.warmup = warmup
        self.iters = iters

    def measure(self, fn) -> dict:
        """执行 warmup + 计时，返回 device_time 和 wall_time 统计"""
        # Warmup
        for _ in range(self.warmup):
            fn()
        self.synchronize()

        device_times = self._measure_device(fn)
        wall_times = self._measure_wall(fn)

        return {
            "device_time": self._compute_stats(device_times),
            "wall_time": self._compute_stats(wall_times),
        }

    def _measure_device(self, fn) -> list:
        """Device event 计时，不含 launch overhead"""
        times = []
        for _ in range(self.iters):
            start_event = self.create_event()
            end_event = self.create_event()
            self.record_event(start_event)
            fn()
            self.record_event(end_event)
            self.synchronize()
            times.append(self.elapsed_time(start_event, end_event))
        return times

    def _measure_wall(self, fn) -> list:
        """Wall clock 计时，含 launch overhead"""
        times = []
        for _ in range(self.iters):
            self.synchronize()
            t0 = time.perf_counter()
            fn()
            self.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms
        return times

    @abstractmethod
    def synchronize(self):
        """设备同步"""
        ...

    @abstractmethod
    def create_event(self):
        """创建计时 event"""
        ...

    @abstractmethod
    def record_event(self, event):
        """记录 event"""
        ...

    @abstractmethod
    def elapsed_time(self, start_event, end_event) -> float:
        """计算两个 event 间的耗时 (ms)"""
        ...

    @staticmethod
    def _compute_stats(times: list) -> dict:
        """计算统计数据"""
        times_sorted = sorted(times)
        n = len(times_sorted)
        return {
            "mean_ms": round(statistics.mean(times_sorted), 4),
            "median_ms": round(statistics.median(times_sorted), 4),
            "min_ms": round(times_sorted[0], 4),
            "max_ms": round(times_sorted[-1], 4),
            "p99_ms": round(times_sorted[min(int(n * 0.99), n - 1)], 4),
            "std_ms": round(statistics.stdev(times_sorted) if n > 1 else 0.0, 4),
        }
