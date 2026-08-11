"""Roofline 模型效率分析"""

import yaml
from pathlib import Path


class RooflineAnalyzer:
    """基于 Roofline 模型计算算子效率

    根据算子的计算量(FLOPs)和访存量(Bytes)，结合硬件峰值规格，
    计算实际效率百分比，并判断算子瓶颈类型。
    """

    def __init__(self, hw_specs: dict):
        """
        Args:
            hw_specs: 硬件规格字典，包含 compute 和 memory 子字段
        """
        self.specs = hw_specs

    @classmethod
    def from_yaml(cls, yaml_path: str, platform_key: str):
        """从 YAML 文件加载硬件规格

        Args:
            yaml_path: hardware_specs.yaml 路径
            platform_key: 平台键名，如 'nvidia_h20'
        """
        with open(yaml_path, "r") as f:
            all_specs = yaml.safe_load(f)
        if platform_key not in all_specs:
            raise ValueError(
                f"Platform '{platform_key}' not found. "
                f"Available: {list(all_specs.keys())}"
            )
        return cls(all_specs[platform_key])

    def analyze(self, op_meta: dict, device_time_ms: float) -> dict:
        """分析算子效率

        Args:
            op_meta: 算子元信息，包含:
                - flops: 理论计算量 (FLOPs)
                - bytes: 访存量 (Bytes)
                - dtype: 数据类型
            device_time_ms: 实测 device time (ms)

        Returns:
            dict with efficiency metrics and bound type
        """
        if device_time_ms <= 0:
            return {"error": "invalid device_time_ms"}

        seconds = device_time_ms / 1000.0
        flops = op_meta["flops"]
        mem_bytes = op_meta["bytes"]
        dtype = op_meta.get("dtype", "bf16")

        # 实际达到的算力和带宽
        achieved_tflops = flops / seconds / 1e12
        achieved_bw = mem_bytes / seconds / 1e12  # TB/s

        # 获取峰值
        dtype_key = self._normalize_dtype_key(dtype)
        peak_tflops = self.specs["compute"].get(f"{dtype_key}_tflops", 0)
        peak_bw = self.specs["memory"]["hbm_bandwidth_tb_s"]

        # 效率百分比
        compute_eff = (achieved_tflops / peak_tflops * 100) if peak_tflops > 0 else 0
        memory_eff = (achieved_bw / peak_bw * 100) if peak_bw > 0 else 0

        # 算术强度判断瓶颈
        arithmetic_intensity = flops / mem_bytes if mem_bytes > 0 else float("inf")
        ridge_point = (peak_tflops * 1e12) / (peak_bw * 1e12) if peak_bw > 0 else 0

        bound = "compute" if arithmetic_intensity > ridge_point else "memory"

        return {
            "achieved_tflops": round(achieved_tflops, 2),
            "achieved_bandwidth_tb_s": round(achieved_bw, 3),
            "compute_efficiency_pct": round(compute_eff, 1),
            "memory_efficiency_pct": round(memory_eff, 1),
            "arithmetic_intensity": round(arithmetic_intensity, 1),
            "ridge_point": round(ridge_point, 1),
            "bound": bound,
        }

    @staticmethod
    def _normalize_dtype_key(dtype: str) -> str:
        """将各种 dtype 表示统一为硬件规格 YAML 中的键"""
        mapping = {
            "float32": "fp32",
            "float16": "fp16",
            "bfloat16": "bf16",
            "float8_e4m3fn": "fp8",
            "float8_e5m2": "fp8",
            "fp32": "fp32",
            "fp16": "fp16",
            "bf16": "bf16",
            "fp8": "fp8",
        }
        return mapping.get(dtype, "bf16")
