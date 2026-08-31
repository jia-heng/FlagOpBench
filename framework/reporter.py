"""结果报告生成"""
import json
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any

import torch


class Reporter:
    """JSON结果报告生成器"""

    def __init__(self, provider_name: str, platform: str = "nvidia",
                 env_info: Dict[str, Any] = None):
        self.provider_name = provider_name
        self.platform = platform
        self.env_info = env_info or {}
        self.results = []

    def add_results(self, results: list):
        """添加BenchmarkResult列表"""
        self.results.extend(results)

    def build_report(self) -> dict:
        """构建完整报告"""
        return {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "provider": self.provider_name,
                "platform": self.platform,
                "num_operators": len(set(r.operator for r in self.results)),
                "num_workloads": len(self.results),
            },
            "environment": self._collect_env(),
            "results": [r.to_dict() for r in self.results],
        }

    def save(self, output_dir: str = None, operator_name: str = None) -> str:
        """保存JSON报告

        输出路径: results/{operator}/{operator}_{provider}.json
        如果 provider 是 flagos，文件名追加 platform: {operator}_flagos_{platform}.json

        Args:
            output_dir: 输出根目录 (默认 results/)
            operator_name: 算子名（用于文件名和子目录名）
        """
        if output_dir is None:
            output_dir = Path(__file__).parent.parent / "results"
        output_dir = Path(output_dir)

        # 从results推断算子名
        if operator_name is None:
            if self.results:
                operator_name = self.results[0].operator
            else:
                operator_name = "unknown"

        # 按算子建子目录: results/{operator}/
        op_dir = output_dir / operator_name
        op_dir.mkdir(parents=True, exist_ok=True)

        # 文件名: flagos 带 platform 后缀，其他直接用 provider_name
        if self.provider_name == "flagos":
            filename = f"{operator_name}_flagos_{self.platform}.json"
        else:
            filename = f"{operator_name}_{self.provider_name}.json"

        filepath = op_dir / filename

        report = self.build_report()
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return str(filepath)

    def print_summary(self):
        """打印摘要"""
        print(f"\n{'='*60}")
        print(f"  Provider: {self.provider_name} | Platform: {self.platform}")
        print(f"  Total workloads: {len(self.results)}")
        print(f"{'='*60}")
        print(f"  {'Operator':<20} {'Workload':<30} {'Mean(ms)':<12} {'GFLOPs':<10}")
        print(f"  {'-'*72}")
        for r in self.results:
            d = r.to_dict()
            perf = d["performance"]
            print(
                f"  {d['operator']:<20} "
                f"{d['workload']:<30} "
                f"{perf['device_time']['mean_ms']:<12.4f} "
                f"{perf['throughput']['gflops']:<10.2f}"
            )

    def _collect_env(self) -> dict:
        """收集环境信息（多平台支持）"""
        env = {
            "python_version": self._get_python_version(),
            "torch_version": torch.__version__,
            "platform": self.platform,
        }

        if self.platform == "nvidia":
            env["cuda_version"] = torch.version.cuda or "N/A"
            if torch.cuda.is_available():
                env["device_name"] = torch.cuda.get_device_name(0)
                env["device_count"] = torch.cuda.device_count()
                props = torch.cuda.get_device_properties(0)
                env["device_memory_gb"] = round(props.total_memory / (1024**3), 1)
        elif self.platform == "ascend":
            env["cann_version"] = self._get_ascend_version()
            env["device_name"] = self._get_ascend_device_name()
        elif self.platform == "metax":
            # 沐曦通过 torch.cuda 接口暴露设备
            if torch.cuda.is_available():
                env["device_name"] = torch.cuda.get_device_name(0)
                env["device_count"] = torch.cuda.device_count()
                props = torch.cuda.get_device_properties(0)
                env["device_memory_gb"] = round(props.total_memory / (1024**3), 1)
            else:
                env["device_name"] = "metax device (torch.cuda unavailable)"
        elif self.platform in ("mthreads", "iluvatar"):
            env["device_name"] = f"{self.platform} device (info pending)"
        else:
            env["device_name"] = "unknown"

        env.update(self.env_info)
        return env

    @staticmethod
    def _get_python_version() -> str:
        import sys
        return f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"

    @staticmethod
    def _get_ascend_version() -> str:
        """获取昇腾CANN版本（预留）"""
        try:
            import torch_npu
            return getattr(torch_npu, "__version__", "N/A")
        except ImportError:
            return "N/A"

    @staticmethod
    def _get_ascend_device_name() -> str:
        """获取昇腾设备名（预留）"""
        try:
            import torch_npu
            if torch.npu.is_available():
                return torch.npu.get_device_name(0)
        except (ImportError, AttributeError):
            pass
        return "N/A"
