"""算子注册表

使用装饰器注册算子实现，通过名称查找算子。
"""

from abc import ABC, abstractmethod
from typing import Dict, Any

import torch

# 全局算子注册表
_OPERATOR_REGISTRY: Dict[str, type] = {}


class BaseOperator(ABC):
    """算子基类

    每个算子实现需继承此类，提供：
    - forward(): 算子执行
    - compute_flops(): 计算理论 FLOPs
    - compute_bytes(): 计算理论访存量
    - prepare_inputs(): 根据 scenario 参数准备输入 tensors
    """

    def __init__(self, device=None):
        self.device = device or torch.device("cuda")

    @property
    @abstractmethod
    def name(self) -> str:
        """算子名称"""
        ...

    @property
    def level(self) -> str:
        """算子层级: basic / model / third_party"""
        return "basic"

    @abstractmethod
    def forward(self, **kwargs) -> torch.Tensor:
        """算子执行

        Args:
            **kwargs: 算子输入参数

        Returns:
            输出 tensor
        """
        ...

    @abstractmethod
    def compute_flops(self, **params) -> int:
        """计算理论 FLOPs

        Args:
            **params: scenario 参数（shape、dtype 等）

        Returns:
            理论计算量 (FLOPs)
        """
        ...

    @abstractmethod
    def compute_bytes(self, **params) -> int:
        """计算理论访存量

        Args:
            **params: scenario 参数

        Returns:
            理论访存量 (Bytes)
        """
        ...

    @abstractmethod
    def prepare_inputs(self, **params) -> dict:
        """根据 scenario 参数准备输入 tensors

        Args:
            **params: scenario 中的参数 (M, K, N, dtype 等)

        Returns:
            输入字典，可直接传递给 forward()
        """
        ...

    def compute_golden(self, **inputs) -> torch.Tensor:
        """计算 golden reference（默认使用自身 forward）

        子类可覆盖此方法，使用 CPU fp32 或其他方式计算参考值。
        """
        return self.forward(**inputs)

    def get_dtype(self, dtype_str: str) -> torch.dtype:
        """将字符串 dtype 转为 torch.dtype"""
        dtype_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
        }
        return dtype_map.get(dtype_str, torch.bfloat16)

    def dtype_bytes(self, dtype_str: str) -> int:
        """获取 dtype 的字节数"""
        bytes_map = {
            "fp32": 4, "float32": 4,
            "fp16": 2, "float16": 2,
            "bf16": 2, "bfloat16": 2,
            "fp8": 1, "float8_e4m3fn": 1, "float8_e5m2": 1,
            "int8": 1, "int4": 0.5,
        }
        return bytes_map.get(dtype_str, 2)


def register_operator(name: str):
    """算子注册装饰器

    Usage:
        @register_operator("mm")
        class MMOperator(BaseOperator):
            ...
    """
    def decorator(cls):
        if not issubclass(cls, BaseOperator):
            raise TypeError(f"{cls.__name__} must inherit from BaseOperator")
        _OPERATOR_REGISTRY[name] = cls
        return cls
    return decorator


def get_operator(name: str, device=None) -> BaseOperator:
    """从注册表获取算子实例

    Args:
        name: 算子名称
        device: torch device

    Returns:
        算子实例

    Raises:
        KeyError: 算子未注册
    """
    if name not in _OPERATOR_REGISTRY:
        raise KeyError(
            f"Operator '{name}' not registered. "
            f"Available: {list(_OPERATOR_REGISTRY.keys())}"
        )
    return _OPERATOR_REGISTRY[name](device=device)


def list_operators() -> list:
    """列出所有已注册的算子"""
    return sorted(_OPERATOR_REGISTRY.keys())


def import_all_operators():
    """导入所有算子模块以触发注册"""
    import importlib
    import pkgutil
    from pathlib import Path

    # 导入 basic 算子
    basic_dir = Path(__file__).parent / "basic"
    if basic_dir.exists():
        for _, module_name, _ in pkgutil.iter_modules([str(basic_dir)]):
            importlib.import_module(f"baseline.operators.basic.{module_name}")

    # 导入 model 算子
    model_dir = Path(__file__).parent / "model"
    if model_dir.exists():
        for _, module_name, _ in pkgutil.iter_modules([str(model_dir)]):
            importlib.import_module(f"baseline.operators.model.{module_name}")
