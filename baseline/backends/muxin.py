"""Muxin GPU 后端实现"""

import torch
from baseline.backends.base import Backend
from baseline.framework.timer import BenchmarkTimer
from baseline.operators.registry import get_operator


class MuxinTimer(BenchmarkTimer):
    """Muxin GPU Event 计时器

    Muxin 可能提供类似 CUDA 的 Event API
    如果没有，可以 fallback 到 wall clock time
    """

    def __init__(self, warmup: int = 10, iters: int = 100):
        super().__init__(warmup, iters)
        self._has_event_api = self._check_event_api()

    def _check_event_api(self) -> bool:
        """检查 Muxin 是否提供 Event API"""
        try:
            import torch_muxin
            return hasattr(torch_muxin.muxin, 'Event')
        except (ImportError, AttributeError):
            return False

    def synchronize(self):
        """Muxin 同步"""
        try:
            import torch_muxin
            torch_muxin.muxin.synchronize()
        except ImportError:
            raise RuntimeError(
                "torch_muxin not installed. "
                "Contact Muxin for installation guide."
            )

    def create_event(self):
        """创建 Muxin Event（如果支持）"""
        if not self._has_event_api:
            return None

        import torch_muxin
        return torch_muxin.muxin.Event(enable_timing=True)

    def record_event(self, event):
        """记录 Muxin Event"""
        if event is not None:
            event.record()

    def elapsed_time(self, start_event, end_event) -> float:
        """计算 Muxin Event 间隔时间（毫秒）"""
        if start_event is not None and end_event is not None:
            return start_event.elapsed_time(end_event)
        return 0.0

    def _measure_device(self, fn) -> list:
        """Device event 计时（如果不支持 Event，fallback 到 wall clock）"""
        if self._has_event_api:
            return super()._measure_device(fn)
        else:
            # Fallback: 使用 wall clock time
            print("Warning: Muxin Event API not available, using wall clock time")
            return self._measure_wall(fn)


class MuxinBackend(Backend):
    """Muxin GPU 后端（沐曦）

    支持 Muxin MR60/MR100 等系列 GPU
    使用 torch_muxin 扩展进行算子调用和计时
    """

    def __init__(self, device_id: int = 0):
        self._device_id = device_id
        self._device = None
        self._torch_muxin = None

    @property
    def name(self) -> str:
        return "muxin"

    @property
    def platform_name(self) -> str:
        if self._device is None:
            self.setup()
        # 获取 Muxin GPU 设备名称
        try:
            device_name = self._torch_muxin.muxin.get_device_name(self._device_id)
            return f"Muxin {device_name}"
        except AttributeError:
            return f"Muxin GPU {self._device_id}"

    def get_device(self):
        if self._device is None:
            self.setup()
        return self._device

    def synchronize(self):
        """Muxin GPU 同步"""
        self._torch_muxin.muxin.synchronize(self._device)

    def create_timer(self, warmup: int = 10, iters: int = 100) -> BenchmarkTimer:
        """创建 Muxin 计时器"""
        return MuxinTimer(warmup=warmup, iters=iters)

    def get_operator(self, op_name: str):
        """从注册表获取算子实例"""
        return get_operator(op_name, device=self.get_device())

    def is_dtype_supported(self, dtype: str) -> bool:
        """Muxin GPU 支持的 dtype（需要根据实际文档调整）"""
        supported = {
            "fp32", "float32",
            "fp16", "float16",
            "bf16", "bfloat16",
            # Muxin 可能支持的其他格式
        }
        return dtype in supported

    def setup(self):
        """初始化 Muxin GPU 设备"""
        try:
            import torch_muxin
            self._torch_muxin = torch_muxin
        except ImportError:
            raise RuntimeError(
                "torch_muxin not installed. "
                "Please contact Muxin for installation guide: https://www.metax-tech.com"
            )

        if not torch_muxin.muxin.is_available():
            raise RuntimeError("Muxin GPU is not available")

        device_count = torch_muxin.muxin.device_count()
        if self._device_id >= device_count:
            raise RuntimeError(
                f"Muxin GPU device {self._device_id} not found. "
                f"Available devices: {device_count}"
            )

        # Muxin 可能使用 'muxin:0' 或其他设备标识
        # 需要根据实际 API 调整
        self._device = torch.device(f"muxin:{self._device_id}")

        # 预热设备
        torch.zeros(1, device=self._device)
        torch_muxin.muxin.synchronize()

    def teardown(self):
        """清理 Muxin GPU 缓存"""
        if self._torch_muxin and hasattr(self._torch_muxin.muxin, 'empty_cache'):
            self._torch_muxin.muxin.empty_cache()
