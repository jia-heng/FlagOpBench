#!/usr/bin/env python3
"""
将旧格式测试用例迁移到 Definition/Workload 格式

旧格式:
  operator: mm
  scenarios:
    - name: "xxx"
      M: 2048
      K: 7168
      N: 18432

新格式:
  operator: mm
  definition: gemm_n18432_k7168
  const_axes:
    N: 18432
    K: 7168
  workloads:
    - name: "xxx"
      M: 2048
"""

import yaml
import sys
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Any


def extract_const_axes_for_gemm(scenarios: List[Dict]) -> Dict[tuple, List[Dict]]:
    """
    对 GEMM 类算子，按 (N, K) 分组
    返回: {(N, K): [scenarios]}
    """
    groups = defaultdict(list)
    for scenario in scenarios:
        N = scenario.get('N')
        K = scenario.get('K')
        if N and K:
            groups[(N, K)].append(scenario)
    return groups


def extract_const_axes_for_bmm(scenarios: List[Dict]) -> Dict[tuple, List[Dict]]:
    """
    对 BMM 类算子，按 (batch, K, N) 分组
    返回: {(batch, K, N): [scenarios]}
    """
    groups = defaultdict(list)
    for scenario in scenarios:
        batch = scenario.get('batch')
        K = scenario.get('K')
        N = scenario.get('N')
        if batch and K and N:
            groups[(batch, K, N)].append(scenario)
    return groups


def extract_const_axes_for_norm(scenarios: List[Dict]) -> Dict[int, List[Dict]]:
    """
    对 Norm 类算子，按 hidden_size 分组
    返回: {hidden_size: [scenarios]}
    """
    groups = defaultdict(list)
    for scenario in scenarios:
        hidden_size = scenario.get('hidden_size')
        if hidden_size:
            groups[hidden_size].append(scenario)
    return groups


def extract_const_axes_for_softmax(scenarios: List[Dict]) -> Dict[int, List[Dict]]:
    """
    对 Softmax 类算子，按 N 分组
    返回: {N: [scenarios]}
    """
    groups = defaultdict(list)
    for scenario in scenarios:
        N = scenario.get('N')
        if N:
            groups[N].append(scenario)
    return groups


def generate_definition_name(operator: str, const_axes: Dict) -> str:
    """生成 Definition 名称"""
    if operator == 'mm':
        N = const_axes['N']
        K = const_axes['K']
        return f"gemm_n{N}_k{K}"
    elif operator == 'bmm':
        batch = const_axes['batch']
        K = const_axes['K']
        N = const_axes['N']
        return f"bmm_b{batch}_k{K}_n{N}"
    elif operator in ['rms_norm', 'layernorm']:
        h = const_axes['hidden_size']
        return f"{operator}_h{h}"
    elif operator == 'softmax':
        N = const_axes['N']
        return f"softmax_n{N}"
    else:
        # 其他算子暂时保持原名
        return operator


def infer_model_from_name(name: str) -> str:
    """从 workload 名称推断模型"""
    if 'deepseek_v3' in name.lower():
        return 'DeepSeek-V3'
    elif 'llama3.1' in name.lower():
        return 'meta-llama/Llama-3.1-8B'
    elif 'qwen' in name.lower():
        return 'Qwen-2.5'
    else:
        return 'unknown'


def infer_phase_from_name(name: str) -> str:
    """从 workload 名称推断推理阶段"""
    name_lower = name.lower()
    if 'decode' in name_lower:
        return 'decode'
    elif 'prefill' in name_lower:
        return 'prefill'
    elif 'batch' in name_lower or 'seq' in name_lower:
        return 'mixed'
    else:
        return 'unknown'


def convert_gemm_to_definitions(old_case: Dict, output_dir: Path):
    """将 mm.yaml 转换为多个 Definition 文件"""
    operator = old_case['operator']
    scenarios = old_case['scenarios']
    warmup = old_case.get('warmup', 10)
    iters = old_case.get('iters', 100)
    level = old_case.get('level', 'basic')

    # 按 (N, K) 分组
    groups = extract_const_axes_for_gemm(scenarios)

    for (N, K), workloads in groups.items():
        definition_name = f"gemm_n{N}_k{K}"

        # 构建新格式
        new_case = {
            'definition': definition_name,
            'operator': operator,
            'level': level,
            'source': 'manual',
            'const_axes': {
                'N': N,
                'K': K,
            },
            'warmup': warmup,
            'iters': iters,
            'workloads': []
        }

        # 转换 workloads
        for w in workloads:
            workload = {
                'name': w['name'],
                'M': w['M'],
                'dtype': w['dtype'],
            }

            # 推断元数据
            model = infer_model_from_name(w['name'])
            phase = infer_phase_from_name(w['name'])

            if model != 'unknown':
                workload['model'] = model
            if phase != 'unknown':
                workload['phase'] = phase

            # 添加 source 描述
            workload['source'] = f"{model}, {w['name']}"

            new_case['workloads'].append(workload)

        # 写入文件
        output_file = output_dir / 'gemm' / f'{definition_name}.yaml'
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            # 写入注释
            f.write(f"# Definition: {definition_name}\n")
            f.write(f"# Operator: {operator}\n")
            f.write(f"# Workloads: {len(workloads)}\n")
            f.write("\n")
            yaml.dump(new_case, f, allow_unicode=True, sort_keys=False)

        print(f"✅ Created: {output_file.relative_to(output_dir.parent.parent)}")
        print(f"   Workloads: {len(workloads)}")


def convert_bmm_to_definitions(old_case: Dict, output_dir: Path):
    """将 bmm.yaml 转换为多个 Definition 文件"""
    operator = old_case['operator']
    scenarios = old_case['scenarios']
    warmup = old_case.get('warmup', 10)
    iters = old_case.get('iters', 100)
    level = old_case.get('level', 'basic')

    # 按 (batch, K, N) 分组
    groups = extract_const_axes_for_bmm(scenarios)

    for (batch, K, N), workloads in groups.items():
        definition_name = f"bmm_b{batch}_k{K}_n{N}"

        new_case = {
            'definition': definition_name,
            'operator': operator,
            'level': level,
            'source': 'manual',
            'const_axes': {
                'batch': batch,
                'K': K,
                'N': N,
            },
            'warmup': warmup,
            'iters': iters,
            'workloads': []
        }

        for w in workloads:
            workload = {
                'name': w['name'],
                'M': w['M'],
                'dtype': w['dtype'],
            }

            model = infer_model_from_name(w['name'])
            phase = infer_phase_from_name(w['name'])

            if model != 'unknown':
                workload['model'] = model
            if phase != 'unknown':
                workload['phase'] = phase

            workload['source'] = f"{model}, {w['name']}"
            new_case['workloads'].append(workload)

        output_file = output_dir / 'gemm' / f'{definition_name}.yaml'
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Definition: {definition_name}\n")
            f.write(f"# Operator: {operator}\n")
            f.write(f"# Workloads: {len(workloads)}\n")
            f.write("\n")
            yaml.dump(new_case, f, allow_unicode=True, sort_keys=False)

        print(f"✅ Created: {output_file.relative_to(output_dir.parent.parent)}")
        print(f"   Workloads: {len(workloads)}")


def convert_norm_to_definitions(old_case: Dict, output_dir: Path):
    """将 norm 类算子转换为 Definition 文件"""
    operator = old_case['operator']
    scenarios = old_case['scenarios']
    warmup = old_case.get('warmup', 10)
    iters = old_case.get('iters', 100)
    level = old_case.get('level', 'basic')

    # 按 hidden_size 分组
    groups = extract_const_axes_for_norm(scenarios)

    for hidden_size, workloads in groups.items():
        definition_name = f"{operator}_h{hidden_size}"

        new_case = {
            'definition': definition_name,
            'operator': operator,
            'level': level,
            'source': 'manual',
            'const_axes': {
                'hidden_size': hidden_size,
            },
            'warmup': warmup,
            'iters': iters,
            'workloads': []
        }

        for w in workloads:
            workload = {
                'name': w['name'],
                'batch_size': w.get('M', w.get('batch_size')),
                'dtype': w['dtype'],
            }

            model = infer_model_from_name(w['name'])
            phase = infer_phase_from_name(w['name'])

            if model != 'unknown':
                workload['model'] = model
            if phase != 'unknown':
                workload['phase'] = phase

            workload['source'] = f"{model}, {w['name']}"
            new_case['workloads'].append(workload)

        output_file = output_dir / 'norm' / f'{definition_name}.yaml'
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Definition: {definition_name}\n")
            f.write(f"# Operator: {operator}\n")
            f.write(f"# Workloads: {len(workloads)}\n")
            f.write("\n")
            yaml.dump(new_case, f, allow_unicode=True, sort_keys=False)

        print(f"✅ Created: {output_file.relative_to(output_dir.parent.parent)}")
        print(f"   Workloads: {len(workloads)}")


def convert_softmax_to_definitions(old_case: Dict, output_dir: Path):
    """将 softmax.yaml 转换为 Definition 文件"""
    operator = old_case['operator']
    scenarios = old_case['scenarios']
    warmup = old_case.get('warmup', 10)
    iters = old_case.get('iters', 100)
    level = old_case.get('level', 'basic')

    # 按 N 分组
    groups = extract_const_axes_for_softmax(scenarios)

    for N, workloads in groups.items():
        definition_name = f"softmax_n{N}"

        new_case = {
            'definition': definition_name,
            'operator': operator,
            'level': level,
            'source': 'manual',
            'const_axes': {
                'N': N,
            },
            'warmup': warmup,
            'iters': iters,
            'workloads': []
        }

        for w in workloads:
            workload = {
                'name': w['name'],
                'M': w['M'],
                'dtype': w['dtype'],
            }

            model = infer_model_from_name(w['name'])
            phase = infer_phase_from_name(w['name'])

            if model != 'unknown':
                workload['model'] = model
            if phase != 'unknown':
                workload['phase'] = phase

            workload['source'] = f"{model}, {w['name']}"
            new_case['workloads'].append(workload)

        output_file = output_dir / 'activation' / f'{definition_name}.yaml'
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w', encoding='utf-8') as f:
            f.write(f"# Definition: {definition_name}\n")
            f.write(f"# Operator: {operator}\n")
            f.write(f"# Workloads: {len(workloads)}\n")
            f.write("\n")
            yaml.dump(new_case, f, allow_unicode=True, sort_keys=False)

        print(f"✅ Created: {output_file.relative_to(output_dir.parent.parent)}")
        print(f"   Workloads: {len(workloads)}")


def main():
    # 路径设置
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    cases_dir = project_root / 'baseline' / 'cases' / 'basic'

    print("=" * 60)
    print("Definition/Workload 格式迁移工具")
    print("=" * 60)
    print()

    # 迁移 mm.yaml
    mm_file = cases_dir / 'mm.yaml'
    if mm_file.exists():
        print(f"📋 处理: {mm_file.name}")
        with open(mm_file, 'r', encoding='utf-8') as f:
            old_case = yaml.safe_load(f)
        convert_gemm_to_definitions(old_case, cases_dir)
        print()

    # 迁移 bmm.yaml
    bmm_file = cases_dir / 'bmm.yaml'
    if bmm_file.exists():
        print(f"📋 处理: {bmm_file.name}")
        with open(bmm_file, 'r', encoding='utf-8') as f:
            old_case = yaml.safe_load(f)
        convert_bmm_to_definitions(old_case, cases_dir)
        print()

    # 迁移 rms_norm.yaml
    rms_norm_file = cases_dir / 'rms_norm.yaml'
    if rms_norm_file.exists():
        print(f"📋 处理: {rms_norm_file.name}")
        with open(rms_norm_file, 'r', encoding='utf-8') as f:
            old_case = yaml.safe_load(f)
        convert_norm_to_definitions(old_case, cases_dir)
        print()

    # 迁移 softmax.yaml
    softmax_file = cases_dir / 'softmax.yaml'
    if softmax_file.exists():
        print(f"📋 处理: {softmax_file.name}")
        with open(softmax_file, 'r', encoding='utf-8') as f:
            old_case = yaml.safe_load(f)
        convert_softmax_to_definitions(old_case, cases_dir)
        print()

    print("=" * 60)
    print("✅ 迁移完成！")
    print()
    print("下一步:")
    print("  1. 验证新生成的 Definition 文件")
    print("  2. 测试运行确保功能一致")
    print("  3. 备份旧文件")
    print("  4. 删除旧格式文件")


if __name__ == '__main__':
    main()
