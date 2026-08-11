#!/usr/bin/env python3
"""
自动采集真实推理 shape - 从实际模型运行中捕获算子调用

使用方法:
    # 方式 1: 使用 sglang
    python scripts/collect_shapes.py --framework sglang --model deepseek-ai/DeepSeek-V3

    # 方式 2: 使用 transformers
    python scripts/collect_shapes.py --framework transformers --model meta-llama/Llama-3.1-8B

    # 方式 3: 自定义代码
    python scripts/collect_shapes.py --custom-script your_inference.py
"""

import torch
import argparse
import json
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Set, Any
import inspect


class ShapeCollector:
    """算子 shape 采集器 - hook PyTorch 算子调用"""

    def __init__(self, output_dir: Path, target_ops: List[str] = None):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 目标算子列表（None = 全部）
        self.target_ops = target_ops or [
            'mm', 'bmm', 'addmm', 'matmul',
            'layer_norm', 'softmax', 'gelu', 'silu',
            'embedding', 'linear',
        ]

        # 采集到的 shapes
        self.shapes: Dict[str, List[Dict]] = defaultdict(list)

        # 去重集合
        self.seen_shapes: Dict[str, Set] = defaultdict(set)

    def _extract_shape_info(self, op_name: str, args, kwargs) -> Dict:
        """提取算子的 shape 信息"""
        info = {"operator": op_name}

        try:
            if op_name in ['mm', 'matmul']:
                if len(args) >= 2:
                    a, b = args[0], args[1]
                    if hasattr(a, 'shape') and hasattr(b, 'shape'):
                        info['M'] = a.shape[0] if len(a.shape) >= 2 else 1
                        info['K'] = a.shape[-1]
                        info['N'] = b.shape[-1]
                        info['dtype'] = str(a.dtype).replace('torch.', '')

            elif op_name == 'bmm':
                if len(args) >= 2:
                    a, b = args[0], args[1]
                    if hasattr(a, 'shape') and hasattr(b, 'shape'):
                        info['B'] = a.shape[0]
                        info['M'] = a.shape[1]
                        info['K'] = a.shape[2]
                        info['N'] = b.shape[2]
                        info['dtype'] = str(a.dtype).replace('torch.', '')

            elif op_name == 'layer_norm':
                if len(args) >= 1:
                    x = args[0]
                    if hasattr(x, 'shape'):
                        info['batch_size'] = x.shape[0] if len(x.shape) >= 2 else 1
                        info['hidden_size'] = x.shape[-1]
                        info['dtype'] = str(x.dtype).replace('torch.', '')

            elif op_name in ['softmax', 'gelu', 'silu']:
                if len(args) >= 1:
                    x = args[0]
                    if hasattr(x, 'shape'):
                        info['shape'] = list(x.shape)
                        info['dtype'] = str(x.dtype).replace('torch.', '')

            elif op_name == 'linear':
                if len(args) >= 2:
                    x, weight = args[0], args[1]
                    if hasattr(x, 'shape') and hasattr(weight, 'shape'):
                        info['batch_size'] = x.shape[0] if len(x.shape) >= 2 else 1
                        info['in_features'] = x.shape[-1]
                        info['out_features'] = weight.shape[0]
                        info['dtype'] = str(x.dtype).replace('torch.', '')

            elif op_name == 'embedding':
                if len(args) >= 2:
                    indices, weight = args[0], args[1]
                    if hasattr(indices, 'shape') and hasattr(weight, 'shape'):
                        info['num_embeddings'] = weight.shape[0]
                        info['embedding_dim'] = weight.shape[1]
                        info['batch_shape'] = list(indices.shape)

        except Exception as e:
            # 如果提取失败，跳过
            pass

        return info

    def _make_shape_key(self, info: Dict) -> str:
        """生成 shape 的唯一 key 用于去重"""
        key_parts = []
        for k in sorted(info.keys()):
            if k != 'operator':
                key_parts.append(f"{k}={info[k]}")
        return "_".join(key_parts)

    def hook_function(self, op_name: str, original_fn):
        """创建 hook 函数"""
        def wrapper(*args, **kwargs):
            # 提取 shape
            info = self._extract_shape_info(op_name, args, kwargs)

            if len(info) > 1:  # 成功提取到信息
                key = self._make_shape_key(info)
                if key not in self.seen_shapes[op_name]:
                    self.seen_shapes[op_name].add(key)
                    self.shapes[op_name].append(info)

            # 调用原始函数
            return original_fn(*args, **kwargs)

        return wrapper

    def install_hooks(self):
        """安装 hooks"""
        self.original_functions = {}

        for op_name in self.target_ops:
            if op_name == 'mm':
                self.original_functions['mm'] = torch.mm
                torch.mm = self.hook_function('mm', torch.mm)

            elif op_name == 'bmm':
                self.original_functions['bmm'] = torch.bmm
                torch.bmm = self.hook_function('bmm', torch.bmm)

            elif op_name == 'matmul':
                self.original_functions['matmul'] = torch.matmul
                torch.matmul = self.hook_function('matmul', torch.matmul)

            elif op_name == 'layer_norm':
                import torch.nn.functional as F
                self.original_functions['layer_norm'] = F.layer_norm
                F.layer_norm = self.hook_function('layer_norm', F.layer_norm)

            elif op_name == 'softmax':
                import torch.nn.functional as F
                self.original_functions['softmax'] = F.softmax
                F.softmax = self.hook_function('softmax', F.softmax)

            # 可以继续添加更多算子...

        print(f"Installed hooks for {len(self.original_functions)} operators")

    def uninstall_hooks(self):
        """卸载 hooks"""
        import torch.nn.functional as F

        if 'mm' in self.original_functions:
            torch.mm = self.original_functions['mm']
        if 'bmm' in self.original_functions:
            torch.bmm = self.original_functions['bmm']
        if 'matmul' in self.original_functions:
            torch.matmul = self.original_functions['matmul']
        if 'layer_norm' in self.original_functions:
            F.layer_norm = self.original_functions['layer_norm']
        if 'softmax' in self.original_functions:
            F.softmax = self.original_functions['softmax']

        print("Uninstalled all hooks")

    def save_results(self):
        """保存采集结果"""
        for op_name, shapes_list in self.shapes.items():
            if not shapes_list:
                continue

            output_file = self.output_dir / f"{op_name}_shapes.json"
            with open(output_file, 'w') as f:
                json.dump({
                    "operator": op_name,
                    "num_unique_shapes": len(shapes_list),
                    "shapes": shapes_list,
                }, f, indent=2)

            print(f"Saved {len(shapes_list)} unique shapes for {op_name} to {output_file}")

    def print_summary(self):
        """打印采集摘要"""
        print("\n" + "=" * 60)
        print("Shape Collection Summary")
        print("=" * 60)

        total_shapes = sum(len(shapes) for shapes in self.shapes.values())
        print(f"Total operators captured: {len(self.shapes)}")
        print(f"Total unique shapes: {total_shapes}")
        print()

        for op_name in sorted(self.shapes.keys()):
            shapes_list = self.shapes[op_name]
            print(f"  {op_name:20s}: {len(shapes_list):4d} unique shapes")

        print("=" * 60)


def collect_from_transformers(model_name: str, collector: ShapeCollector):
    """从 transformers 模型采集"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError:
        print("Error: transformers not installed. Run: pip install transformers")
        return

    print(f"Loading model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    collector.install_hooks()

    # 运行推理
    prompts = [
        "Hello, how are you?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list.",
    ]

    print("Running inference to collect shapes...")
    for prompt in prompts:
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=50)

    collector.uninstall_hooks()
    print("Collection complete!")


def collect_from_sglang(model_name: str, collector: ShapeCollector):
    """从 sglang 模型采集"""
    try:
        import sglang as sgl
    except ImportError:
        print("Error: sglang not installed. Run: pip install sglang")
        return

    print(f"Loading model with sglang: {model_name}")

    collector.install_hooks()

    # 运行推理
    runtime = sgl.Runtime(model_path=model_name)
    sgl.set_default_backend(runtime)

    prompts = [
        "Hello, how are you?",
        "Explain quantum computing in simple terms.",
        "Write a Python function to sort a list.",
    ]

    print("Running inference to collect shapes...")
    for prompt in prompts:
        state = runtime.generate(prompt, max_new_tokens=50)

    collector.uninstall_hooks()
    runtime.shutdown()
    print("Collection complete!")


def collect_from_custom_script(script_path: str, collector: ShapeCollector):
    """从自定义脚本采集"""
    print(f"Running custom script: {script_path}")

    collector.install_hooks()

    # 执行用户脚本
    with open(script_path) as f:
        code = f.read()
        exec(code, {"torch": torch, "collector": collector})

    collector.uninstall_hooks()
    print("Collection complete!")


def main():
    parser = argparse.ArgumentParser(description="Collect operator shapes from real inference")
    parser.add_argument("--framework", choices=['transformers', 'sglang', 'custom'],
                        default='transformers', help="Inference framework")
    parser.add_argument("--model", type=str, help="Model name or path")
    parser.add_argument("--custom-script", type=str, help="Custom Python script to run")
    parser.add_argument("--output-dir", type=str, default="collected_shapes",
                        help="Output directory for collected shapes")
    parser.add_argument("--operators", type=str, nargs="*",
                        help="Target operators (default: all)")

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    collector = ShapeCollector(output_dir, target_ops=args.operators)

    try:
        if args.framework == 'transformers':
            if not args.model:
                print("Error: --model is required for transformers")
                return 1
            collect_from_transformers(args.model, collector)

        elif args.framework == 'sglang':
            if not args.model:
                print("Error: --model is required for sglang")
                return 1
            collect_from_sglang(args.model, collector)

        elif args.framework == 'custom':
            if not args.custom_script:
                print("Error: --custom-script is required for custom mode")
                return 1
            collect_from_custom_script(args.custom_script, collector)

    finally:
        # 确保保存结果
        collector.save_results()
        collector.print_summary()

    return 0


if __name__ == "__main__":
    exit(main())
