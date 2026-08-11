"""Ascend NPU 后端实现"""

import torch
from baseline.backends.base import Backend
from baseline.framework.timer import BenchmarkTimer
from baseline.operators.registry import get_operator


class AscendTimer(BenchmarkTimer):
    """Ascend NPU Event 计时器

    使用 torch_npu 提供的 Event API，接口与 CUDA Event 完全一致
    """

    def synchronize(self):
        """NPU 同步"""
        try:
            import torch_npu
            torch_npu.npu.synchronize()
        except ImportError:
            raise RuntimeError(
                "torch_npu not installed. "
                "Install: pip install torch-npu"
            )

    def create_event(self):
        """创建 NPU Event"""
        import torch_npu
        return torch_npu.npu.Event(enable_timing=True)

    def record_event(self, event):
        """记录 NPU Event"""
        event.record()

    def elapsed_time(self, start_event, end_event) -> float:
        """计算 NPU Event 间隔时间（毫秒）"""
        return start_event.elapsed_time(end_event)


class AscendBackend(Backend):
    """Ascend NPU 后端（华为昇腾）

    支持 Ascend 910A/910B/910C 等系列 NPU
    使用 torch_npu 扩展进行算子调用和计时
    """

    def __init__(self, device_id: int = 0):
        self._device_id = device_id
        self._device = None
        self._torch_npu = None

    @property
    def name(self) -> str:
        return "ascend"

    @property
    def platform_name(self) -> str:
        if self._device is None:
            self.setup()
        # 获取 NPU 设备名称
        return f"Ascend {self._torch_npu.npu.get_device_name(self._device_id)}"

    def get_device(self):
        if self._device is None:
            self.setup()
        return self._device

    def synchronize(self):
        """NPU 同步"""
        self._torch_npu.npu.synchronize(self._device)

    def create_timer(self, warmup: int = 10, iters: int = 100) -> BenchmarkTimer:
        """创建 NPU 计时器"""
        return AscendTimer(warmup=warmup, iters=iters)

    def get_operator(self, op_name: str):
        """从注册表获取算子实例"""
        return get_operator(op_name, device=self.get_device())

    def is_dtype_supported(self, dtype: str) -> bool:
        """Ascend NPU 支持的 dtype"""
        supported = {
            "fp32", "float32",
            "fp16", "float16",
            "bf16", "bfloat16",  # 910B+ 支持 bf16
            # 注意：Ascend 不支持 NVIDIA 的 fp8 格式
        }
        return dtype in supported

    def setup(self):
        """初始化 Ascend NPU 设备"""
        try:
            import torch_npu
            self._torch_npu = torch_npu
        except ImportError:
            raise RuntimeError(
                "torch_npu not installed. "
                "Please install: pip install torch-npu"
            )

        if not torch_npu.npu.is_available():
            raise RuntimeError("Ascend NPU is not available")

        device_count = torch_npu.npu.device_count()
        if self._device_id >= device_count:
            raise RuntimeError(
                f"NPU device {self._device_id} not found. "
                f"Available devices: {device_count}"
            )

        self._device = torch.device(f"npu:{self._device_id}")

        # 预热设备（触发 context 初始化）
        torch.zeros(1, device=self._device)
        torch_npu.npu.synchronize()

    def teardown(self):
        """清理 NPU 缓存"""
        if self._torch_npu:
            self._torch_npu.npu.empty_cache()
