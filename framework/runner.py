"""测试执行引擎"""
import yaml
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch

from .registry import get_operator, import_all_operators
from .timer import Timer, BaseTimer, TimingResult, create_timer
from .base_operator import BaseOperator


class BenchmarkResult:
    """单个workload的测试结果"""

    def __init__(
        self,
        operator: str,
        definition: str,
        workload: str,
        provider: str,
        params: Dict[str, Any],
        timing: TimingResult,
        flops: int,
        bytes_accessed: int,
        impl_info: Dict[str, str],
    ):
        self.operator = operator
        self.definition = definition
        self.workload = workload
        self.provider = provider
        self.params = params
        self.timing = timing
        self.flops = flops
        self.bytes_accessed = bytes_accessed
        self.impl_info = impl_info

    def to_dict(self) -> dict:
        mean_ms = self.timing.mean_ms
        return {
            "operator": self.operator,
            "definition": self.definition,
            "workload": self.workload,
            "provider": self.provider,
            "parameters": self.params,
            "performance": {
                "device_time": {
                    "mean_ms": round(mean_ms, 4),
                    "std_ms": round(self.timing.std_ms, 4),
                    "min_ms": round(self.timing.min_ms, 4),
                    "max_ms": round(self.timing.max_ms, 4),
                },
                "throughput": {
                    "gflops": round(self.flops / (mean_ms * 1e6), 2) if mean_ms > 0 else 0,
                    "bandwidth_gb_s": round(self.bytes_accessed / (mean_ms * 1e6), 2) if mean_ms > 0 else 0,
                },
                "theory": {
                    "flops": self.flops,
                    "bytes": self.bytes_accessed,
                    "arithmetic_intensity": round(self.flops / self.bytes_accessed, 2) if self.bytes_accessed > 0 else 0,
                },
            },
            "impl_info": self.impl_info,
        }


class CompareResult:
    """对比结果: 平台基线 vs FlagOS"""

    def __init__(
        self,
        operator: str,
        workload: str,
        params: Dict[str, Any],
        baseline: BenchmarkResult,
        flagos: BenchmarkResult,
    ):
        self.operator = operator
        self.workload = workload
        self.params = params
        self.baseline = baseline
        self.flagos = flagos

    @property
    def speedup(self) -> float:
        """FlagOS相对于基线的加速比 (>1表示FlagOS更快)"""
        if self.flagos.timing.mean_ms <= 0:
            return 0.0
        return self.baseline.timing.mean_ms / self.flagos.timing.mean_ms

    def to_dict(self) -> dict:
        return {
            "operator": self.operator,
            "workload": self.workload,
            "parameters": self.params,
            "baseline": self.baseline.to_dict(),
            "flagos": self.flagos.to_dict(),
            "speedup": round(self.speedup, 4),
        }


class Runner:
    """测试执行引擎"""

    def __init__(self, provider, warmup: int = 10, repeat: int = 100,
                 timer: Optional[BaseTimer] = None):
        """
        Args:
            provider: Provider实例
            warmup: 预热次数
            repeat: 重复测试次数
            timer: 计时器实例（可选，默认根据provider创建CudaTimer）
        """
        self.provider = provider
        if timer is not None:
            self.timer = timer
        else:
            self.timer = Timer(warmup=warmup, repeat=repeat)

    def run_case_file(self, case_path: str) -> List[BenchmarkResult]:
        """运行一个case文件中的所有workload

        支持两种格式:
          1. 单section: {operator, definition_name, const_axes, workloads}
          2. 多section (merged): {operator, sections: [{const_axes, models, workloads}, ...]}
        """
        with open(case_path, "r") as f:
            definition = yaml.safe_load(f)

        op_name = definition["operator"]

        # 获取算子
        operator = get_operator(op_name)

        # 获取impl
        impl_fn, impl_info = self.provider.get_impl(op_name, operator)
        if impl_fn is None:
            print(f"  [SKIP] {op_name}: no impl in provider '{self.provider.name}'")
            return []

        # 判断格式：merged (sections) vs legacy (单 const_axes)
        if "sections" in definition:
            sections = definition["sections"]
        else:
            # 兼容旧格式
            sections = [{
                "const_axes": definition.get("const_axes", {}),
                "models": [],
                "workloads": definition.get("workloads", []),
            }]

        definition_name = definition.get("definition_name", op_name)

        results = []
        for section in sections:
            const_axes = section.get("const_axes", {})
            models_tag = ", ".join(section.get("models", []))
            if models_tag:
                print(f"\n  [section] {op_name} | models: {models_tag}")

            for workload in section.get("workloads", []):
                wl_name = workload["name"]
                var_axes = workload.get("var_axes", {})
                params = {**const_axes, **var_axes}
                if "dtype" in workload:
                    params["dtype"] = workload["dtype"]

                print(f"  [{self.provider.name}] {op_name} / {wl_name} ...", end=" ")

                try:
                    # 准备输入
                    inputs = operator.prepare_inputs(**params)

                    # 用Provider包装调用
                    def run_fn(**kw):
                        return impl_fn(**kw)

                    # 计时
                    timing = self.timer.measure(run_fn, inputs)

                    # 理论指标
                    flops = operator.compute_flops(**params)
                    bytes_accessed = operator.compute_bytes(**params)

                    result = BenchmarkResult(
                        operator=op_name,
                        definition=definition_name,
                        workload=wl_name,
                        provider=self.provider.name,
                        params=params,
                        timing=timing,
                        flops=flops,
                        bytes_accessed=bytes_accessed,
                        impl_info=impl_info,
                    )
                    results.append(result)
                    print(f"{timing.mean_ms:.4f} ms")

                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

        return results

    def run_case_dir(self, case_dir: str) -> List[BenchmarkResult]:
        """运行目录下所有case文件"""
        results = []
        case_path = Path(case_dir)
        yaml_files = sorted(case_path.rglob("*.yaml"))
        yaml_files = [f for f in yaml_files if f.name != "_template.yaml"]

        for f in yaml_files:
            print(f"\n  Case: {f.relative_to(case_path)}")
            results.extend(self.run_case_file(str(f)))

        return results

    def run_compare(
        self,
        case_path: str,
        baseline_provider,
        flagos_provider,
    ) -> List[CompareResult]:
        """对比模式：分别用两个Provider跑同一case，返回对比结果

        Args:
            case_path: case YAML 路径
            baseline_provider: 平台基线Provider
            flagos_provider: FlagOS Provider

        Returns:
            CompareResult列表
        """
        # 用基线Provider跑
        self.provider = baseline_provider
        baseline_results = self.run_case_file(case_path)

        # 用FlagOS Provider跑
        self.provider = flagos_provider
        flagos_results = self.run_case_file(case_path)

        # 按 (operator, workload) 配对
        baseline_map = {
            (r.operator, r.workload): r for r in baseline_results
        }
        flagos_map = {
            (r.operator, r.workload): r for r in flagos_results
        }

        compare_results = []
        for key in baseline_map:
            if key in flagos_map:
                br = baseline_map[key]
                fr = flagos_map[key]
                compare_results.append(CompareResult(
                    operator=br.operator,
                    workload=br.workload,
                    params=br.params,
                    baseline=br,
                    flagos=fr,
                ))

        return compare_results
