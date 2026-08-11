"""环境信息采集"""

import platform
import subprocess
import sys


class EnvCollector:
    """采集运行环境信息

    收集 Python、PyTorch、CUDA、GPU 等环境信息，
    用于结果报告中的可追溯性。
    """

    def collect(self) -> dict:
        """采集所有环境信息"""
        info = {
            "python": sys.version.split()[0],
            "os": platform.system(),
            "os_version": platform.release(),
            "hostname": platform.node(),
        }

        # PyTorch 信息
        info.update(self._collect_torch_info())

        # GPU 信息
        info.update(self._collect_gpu_info())

        return info

    def _collect_torch_info(self) -> dict:
        """采集 PyTorch 相关信息"""
        info = {}
        try:
            import torch
            info["pytorch"] = torch.__version__
            info["cuda_available"] = torch.cuda.is_available()
            if torch.cuda.is_available():
                info["cuda_version"] = torch.version.cuda or "N/A"
                info["cudnn_version"] = str(torch.backends.cudnn.version()) if torch.backends.cudnn.is_available() else "N/A"
        except ImportError:
            info["pytorch"] = "not installed"
        return info

    def _collect_gpu_info(self) -> dict:
        """采集 GPU 信息"""
        info = {}
        try:
            import torch
            if torch.cuda.is_available():
                info["gpu_count"] = torch.cuda.device_count()
                info["gpu_model"] = torch.cuda.get_device_name(0)
                props = torch.cuda.get_device_properties(0)
                info["gpu_memory_gb"] = round(props.total_memory / (1024**3), 1)
        except (ImportError, RuntimeError):
            pass

        # 尝试获取 driver 版本
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                info["driver"] = result.stdout.strip().split("\n")[0]
        except (FileNotFoundError, subprocess.TimeoutExpired):
            pass

        # 尝试获取 vLLM 版本
        try:
            import vllm
            info["vllm"] = vllm.__version__
        except ImportError:
            pass

        # Triton 版本
        try:
            import triton
            info["triton"] = triton.__version__
        except ImportError:
            pass

        return info
