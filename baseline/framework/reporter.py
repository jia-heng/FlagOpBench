"""结果采集与 JSON 输出"""

import json
import datetime
from pathlib import Path


class Reporter:
    """测试结果采集与输出

    收集所有 case 的测试结果，输出统一格式的 JSON 报告。
    """

    def __init__(self, platform: str, backend_name: str, env_info: dict = None):
        """
        Args:
            platform: 平台标识，如 "NVIDIA H20"
            backend_name: 后端名称，如 "nvidia"
            env_info: 环境信息字典
        """
        self.platform = platform
        self.backend_name = backend_name
        self.env_info = env_info or {}
        self.results = []

    def add_result(self, result: dict):
        """添加单个测试结果"""
        self.results.append(result)

    def add_results(self, results: list):
        """批量添加测试结果"""
        self.results.extend(results)

    def generate_report(self) -> dict:
        """生成完整报告"""
        return {
            "platform": self.platform,
            "backend": self.backend_name,
            "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
            "env": self.env_info,
            "total_operators": len(set(r["operator"] for r in self.results)),
            "total_scenarios": len(self.results),
            "passed": sum(1 for r in self.results if r.get("accuracy", {}).get("passed", False)),
            "failed": sum(1 for r in self.results if not r.get("accuracy", {}).get("passed", True)),
            "results": self.results,
        }

    def save(self, output_path: str = None) -> str:
        """保存报告到 JSON 文件

        Args:
            output_path: 输出路径。None 则自动生成。

        Returns:
            实际保存的文件路径
        """
        report = self.generate_report()

        if output_path is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"results/{self.backend_name}_{timestamp}.json"

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)

        return str(path)

    def print_summary(self):
        """打印测试摘要到终端"""
        report = self.generate_report()
        print(f"\n{'='*60}")
        print(f"  Performance Baseline Report")
        print(f"  Platform: {report['platform']}")
        print(f"  Backend:  {report['backend']}")
        print(f"  Time:     {report['timestamp']}")
        print(f"{'='*60}")
        print(f"  Total operators: {report['total_operators']}")
        print(f"  Total scenarios: {report['total_scenarios']}")
        print(f"  Passed: {report['passed']}  Failed: {report['failed']}")
        print(f"{'='*60}\n")

        # 打印每个结果的摘要
        for r in self.results:
            status = "✓" if r.get("accuracy", {}).get("passed", False) else "✗"
            op = r.get("operator", "unknown")
            scenario = r.get("scenario", "unknown")
            perf = r.get("performance", {})
            device_mean = perf.get("device_time", {}).get("mean_ms", 0)
            roofline = r.get("roofline", {})
            bound = roofline.get("bound", "-")
            eff = roofline.get("compute_efficiency_pct", 0)
            if bound == "memory":
                eff = roofline.get("memory_efficiency_pct", 0)

            print(f"  [{status}] {op}/{scenario}: "
                  f"{device_mean:.4f} ms | {bound}-bound | eff={eff:.1f}%")

        print()
