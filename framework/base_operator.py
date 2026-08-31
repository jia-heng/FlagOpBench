"""算子基类"""
from abc import ABC, abstractmethod
from typing import Dict, Any
import torch


class BaseOperator(ABC):
    """算子基类

    每个算子实现需提供：
    - prepare_inputs(): 根据workload参数准备输入tensors
    - compute_flops(): 计算理论FLOPs
    - compute_bytes(): 计算理论访存量
    """

    def __init__(self):
        pass

    @property
    @abstractmethod
    def name(self) -> str:
        """算子名称"""
        ...

    @property
    @abstractmethod
    def library(self) -> str:
        """所属库: flaggems / flaggems_vllm / flagattention"""
        ...

    @abstractmethod
    def prepare_inputs(self, **params) -> Dict[str, torch.Tensor]:
        """根据workload参数准备输入tensors

        Args:
            **params: workload中的参数（shape、dtype等）

        Returns:
            输入字典，key为参数名，value为tensor
        """
        ...

    @abstractmethod
    def compute_flops(self, **params) -> int:
        """计算理论FLOPs

        Args:
            **params: workload参数

        Returns:
            理论计算量（FLOPs）
        """
        ...

    @abstractmethod
    def compute_bytes(self, **params) -> int:
        """计算理论访存量

        Args:
            **params: workload参数

        Returns:
            理论访存量（Bytes）
        """
        ...

    def get_dtype(self, dtype_str: str) -> torch.dtype:
        """将字符串dtype转为torch.dtype"""
        dtype_map = {
            "fp32": torch.float32,
            "float32": torch.float32,
            "fp16": torch.float16,
            "float16": torch.float16,
            "bf16": torch.bfloat16,
            "bfloat16": torch.bfloat16,
            "fp8_e4m3": torch.float8_e4m3fn,
            "fp8_e5m2": torch.float8_e5m2,
        }
        return dtype_map.get(dtype_str, torch.bfloat16)

    def dtype_bytes(self, dtype_str: str) -> float:
        """获取dtype的字节数"""
        bytes_map = {
            "fp32": 4, "float32": 4,
            "fp16": 2, "float16": 2,
            "bf16": 2, "bfloat16": 2,
            "fp8": 1, "fp8_e4m3": 1, "fp8_e5m2": 1,
            "int8": 1,
            "int4": 0.5,
        }
        return bytes_map.get(dtype_str, 2)
