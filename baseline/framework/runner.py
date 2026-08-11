"""测试执行引擎

加载 case → 调用后端 → 计时 → 校验 → Roofline 分析 → 输出
"""

import yaml
from pathlib import Path
from typing import List

from baseline.backends.base import Backend
from baseline.framework.validator import Validator
from baseline.framework.roofline import RooflineAnalyzer


class BenchmarkRunner:
    """测试执行引擎

    串联完整测试流程：加载用例、准备输入、精度校验、性能测试、效率分析。
    """

    def __init__(self, backend: Backend, validator: Validator = None,
                 roofline: RooflineAnalyzer = None):
        """
        Args:
            backend: 后端实例
            validator: 精度校验器，None 则使用默认配置
            roofline: Roofline 分析器，None 则跳过效率分析
        """
        self.backend = backend
        self.validator = validator or Validator()
        self.roofline = roofline

    def run_case_file(self, case_path: str) -> List[dict]:
        """执行单个 case 文件

        Args:
            case_path: YAML case 文件路径

        Returns:
            该 case 所有 scenario 的结果列表
        """
        with open(case_path, "r", encoding="utf-8") as f:
            case_config = yaml.safe_load(f)

        return self.run_case(case_config)

    def run_case_dir(self, case_dir: str) -> List[dict]:
        """执行目录下所有 case 文件

        Args:
            case_dir: case 目录路径

        Returns:
            所有 case 的结果列表
        """
        results = []
        case_path = Path(case_dir)
        for yaml_file in sorted(case_path.rglob("*.yaml")):
            print(f"  Running: {yaml_file.name}")
            try:
                results.extend(self.run_case_file(str(yaml_file)))
            except Exception as e:
                print(f"  [ERROR] {yaml_file.name}: {e}")
                results.append({
                    "operator": yaml_file.stem,
                    "scenario": "error",
                    "error": str(e),
                })
        return results

    def run_case(self, case_config: dict) -> List[dict]:
        """执行单个 case 配置

        Args:
            case_config: case 字典，包含 operator, level, scenarios/workloads 等

        Returns:
            scenario/workload 结果列表
        """
        op_name = case_config["operator"]
        warmup = case_config.get("warmup", 10)
        iters = case_config.get("iters", 100)

        # 获取算子实现
        try:
            operator = self.backend.get_operator(op_name)
        except KeyError as e:
            return [{"operator": op_name, "scenario": "error", "error": str(e)}]

        # 创建计时器
        timer = self.backend.create_timer(warmup=warmup, iters=iters)

        results = []

        # 支持两种格式: scenarios (旧格式) 或 workloads (新 Definition 格式)
        test_cases = case_config.get("workloads") or case_config.get("scenarios", [])

        # 如果是 Definition 格式，将 const_axes 合并到每个 workload
        const_axes = case_config.get("const_axes", {})

        for test_case in test_cases:
            # 合并 const_axes 和 workload 参数
            merged_params = {**const_axes, **test_case}
            result = self._run_scenario(operator, timer, merged_params, op_name)
            results.append(result)

        return results

    def _run_scenario(self, operator, timer, scenario: dict,
                      op_name: str) -> dict:
        """执行单个 scenario

        Args:
            operator: 算子实例
            timer: 计时器实例
            scenario: scenario 参数字典
            op_name: 算子名称

        Returns:
            单个 scenario 的完整结果
        """
        scenario_name = scenario.get("name", "unnamed")
        dtype = scenario.get("dtype", "bf16")

        # 检查 dtype 支持
        if not self.backend.is_dtype_supported(dtype):
            return {
                "operator": op_name,
                "scenario": scenario_name,
                "params": scenario,
                "skipped": True,
                "reason": f"dtype '{dtype}' not supported on {self.backend.name}",
            }

        try:
            # 准备输入
            params = {k: v for k, v in scenario.items() if k != "name"}
            inputs = operator.prepare_inputs(**params)

            # 精度校验
            actual_output = operator.forward(**inputs)
            golden_output = operator.compute_golden(**inputs)
            accuracy = self.validator.validate(actual_output, golden_output, dtype)

            # 性能测试
            fn = lambda: operator.forward(**inputs)
            performance = timer.measure(fn)

            # Roofline 分析
            roofline_result = None
            if self.roofline:
                try:
                    op_meta = {
                        "flops": operator.compute_flops(**params),
                        "bytes": operator.compute_bytes(**params),
                        "dtype": dtype,
                    }
                    device_time_ms = performance["device_time"]["mean_ms"]
                    roofline_result = self.roofline.analyze(op_meta, device_time_ms)
                except Exception as e:
                    roofline_result = {"error": str(e)}

            result = {
                "operator": op_name,
                "scenario": scenario_name,
                "params": params,
                "accuracy": accuracy,
                "performance": performance,
            }
            if roofline_result:
                result["roofline"] = roofline_result

            return result

        except Exception as e:
            return {
                "operator": op_name,
                "scenario": scenario_name,
                "params": scenario,
                "error": str(e),
            }
