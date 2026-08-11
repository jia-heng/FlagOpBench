"""精度校验：对比 golden reference 输出"""

import torch
import numpy as np


class Validator:
    """精度校验器

    对比算子输出和 golden reference，计算各类误差指标。
    """

    # 各 dtype 默认容差
    DEFAULT_TOLERANCE = {
        "fp32": 1e-5,
        "float32": 1e-5,
        "fp16": 1e-3,
        "float16": 1e-3,
        "bf16": 1e-2,
        "bfloat16": 1e-2,
        "fp8": 1e-1,
        "float8_e4m3fn": 1e-1,
        "float8_e5m2": 1e-1,
    }

    def __init__(self, tolerance: float = None):
        """
        Args:
            tolerance: 自定义容差。None 则根据 dtype 自动选取。
        """
        self.tolerance = tolerance

    def validate(self, actual, expected, dtype: str = None) -> dict:
        """对比 actual 和 expected，返回精度报告

        Args:
            actual: 实际输出 tensor 或 tuple of tensors
            expected: 期望输出 tensor 或 tuple of tensors (golden reference)
            dtype: 数据类型字符串，用于选择容差

        Returns:
            dict with keys: passed, max_abs_error, max_rel_error,
                           cosine_similarity, tolerance
        """
        # 处理 tuple 输出（如量化算子返回 (quant, scale)）
        if isinstance(actual, tuple) and isinstance(expected, tuple):
            if len(actual) != len(expected):
                return {
                    "passed": False,
                    "error": f"Tuple length mismatch: actual={len(actual)}, expected={len(expected)}",
                    "max_abs_error": float("inf"),
                    "max_rel_error": float("inf"),
                    "cosine_similarity": 0.0,
                }
            # 找到第一个 tensor 元素进行验证
            actual_tensor = None
            expected_tensor = None
            for a, e in zip(actual, expected):
                if isinstance(a, torch.Tensor) and isinstance(e, torch.Tensor):
                    actual_tensor = a
                    expected_tensor = e
                    break

            if actual_tensor is None or expected_tensor is None:
                return {
                    "passed": False,
                    "error": "No tensor found in tuple outputs",
                    "max_abs_error": float("inf"),
                    "max_rel_error": float("inf"),
                    "cosine_similarity": 0.0,
                }

            actual = actual_tensor
            expected = expected_tensor

        # 转为 float32 进行对比
        a = actual.detach().float().cpu()
        e = expected.detach().float().cpu()

        # 确保 shape 一致
        if a.shape != e.shape:
            return {
                "passed": False,
                "error": f"Shape mismatch: actual={a.shape}, expected={e.shape}",
                "max_abs_error": float("inf"),
                "max_rel_error": float("inf"),
                "cosine_similarity": 0.0,
            }

        # 计算误差指标
        abs_diff = torch.abs(a - e)
        max_abs_error = abs_diff.max().item()

        # 相对误差 (避免除零)
        rel_diff = abs_diff / (torch.abs(e) + 1e-12)
        max_rel_error = rel_diff.max().item()

        # 余弦相似度
        a_flat = a.flatten()
        e_flat = e.flatten()
        cos_sim = self._cosine_similarity(a_flat, e_flat)

        # 判断是否通过
        tol = self.tolerance or self._get_tolerance(dtype)
        passed = max_abs_error <= tol or cos_sim >= (1.0 - tol)

        return {
            "passed": passed,
            "max_abs_error": float(f"{max_abs_error:.6e}"),
            "max_rel_error": float(f"{max_rel_error:.6e}"),
            "cosine_similarity": round(cos_sim, 6),
            "tolerance": tol,
        }

    def _get_tolerance(self, dtype: str) -> float:
        """根据 dtype 获取默认容差"""
        if dtype is None:
            return 1e-3
        return self.DEFAULT_TOLERANCE.get(dtype, 1e-2)

    @staticmethod
    def _cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
        """计算余弦相似度"""
        dot = torch.dot(a, b)
        norm_a = torch.norm(a)
        norm_b = torch.norm(b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return (dot / (norm_a * norm_b)).item()
