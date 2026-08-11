"""NVIDIA CUDA 后端实现"""

import torch
from baseline.backends.base import Backend
from baseline.framework.timer import BenchmarkTimer
from baseline.operators.registry import get_operator


class CudaTimer(BenchmarkTimer):
    """CUDA Event 计时器"""

    def synchronize(self):
        torch.cuda.synchronize()

    def create_event(self):
        return torch.cuda.Event(enable_timing=True)

    def record_event(self, event):
        event.record()

    def elapsed_time(self, start_event, end_event) -> float:
        return start_event.elapsed_time(end_event)  # ms


class NvidiaBackend(Backend):
    """NVIDIA CUDA 后端

    使用 PyTorch CUDA 接口进行算子调用和计时。
    """

    def __init__(self, device_id: int = 0):
        self._device_id = device_id
        self._device = None

    @property
    def name(self) -> str:
        return "nvidia"

    @property
    def platform_name(self) -> str:
        if self._device is None:
            self.setup()
        return f"NVIDIA {torch.cuda.get_device_name(self._device_id)}"

    def get_device(self):
        if self._device is None:
            self.setup()
        return self._device

    def synchronize(self):
        torch.cuda.synchronize(self._device)

    def create_timer(self, warmup: int = 10, iters: int = 100) -> BenchmarkTimer:
        return CudaTimer(warmup=warmup, iters=iters)

    def get_operator(self, op_name: str):
        """从注册表获取算子实例"""
        return get_operator(op_name, device=self.get_device())

    def is_dtype_supported(self, dtype: str) -> bool:
        """NVIDIA 支持所有常见 dtype，包括 fp8"""
        supported = {
            "fp32", "float32",
            "fp16", "float16",
            "bf16", "bfloat16",
            "fp8", "float8_e4m3fn", "float8_e5m2",
        }
        return dtype in supported

    def setup(self):
        """初始化 CUDA 设备"""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available")
        if self._device_id >= torch.cuda.device_count():
            raise RuntimeError(
                f"Device {self._device_id} not found. "
                f"Available devices: {torch.cuda.device_count()}"
            )
        self._device = torch.device(f"cuda:{self._device_id}")
        # 预热设备（触发 context 初始化）
        torch.zeros(1, device=self._device)

    def teardown(self):
        """清理 CUDA 缓存"""
        torch.cuda.empty_cache()
