# 操作手册

日常使用 Flagtests 的命令速查和操作流程。

## 日常操作

### 跑单个算子的完整流程

```bash
OP=fused_moe

# 生成 case（如果模型或算子配置有变动）
python gen_cases.py --operators $OP

# 跑 FlagOS
python run.py --provider flagos --case cases/generated/merged/${OP}.yaml

# 跑 vLLM baseline
python run.py --provider vllm --case cases/generated/merged/${OP}.yaml

# 对比并保存 JSON
python compare.py --op $OP --save
```

### 全量跑一遍

```bash
# 生成所有 case
python gen_cases.py

# 跑全部 FlagOS
for f in cases/generated/merged/*.yaml; do
    python run.py --provider flagos --case "$f"
done

# 跑全部 vLLM
for f in cases/generated/merged/*.yaml; do
    python run.py --provider vllm --case "$f"
done

# 全量对比
python compare.py --all --save
```

### 只看对比结果（不重新跑 benchmark）

```bash
# 单个算子
python compare.py --op fused_moe

# 全部（使用已有的 results/ JSON）
python compare.py --all
```

---

## 命令参考

### run.py

```bash
python run.py --provider {flagos|vllm} --case <yaml_path> [--warmup 3] [--repeat 10]
```

| 参数 | 说明 |
|------|------|
| `--provider` | `flagos`（加载 FlagOS 算子库）或 `vllm`（加载 baseline） |
| `--case` | Workload YAML 文件路径 |
| `--warmup` | 预热次数（默认 3） |
| `--repeat` | 计时重复次数（默认 10） |

输出：`results/{operator}_{provider}.json`

### compare.py

```bash
python compare.py --op <operator_name> [--save]
python compare.py --all [--save] [--baseline vllm] [--target flagos]
```

| 参数 | 说明 |
|------|------|
| `--op` | 指定算子名（自动查找对应的 flagos/vllm JSON） |
| `--all` | 对比所有有结果的算子 |
| `--save` | 保存 `results/{op}_compare.json` |
| `--baseline` | Baseline provider（默认 vllm） |
| `--target` | Target provider（默认 flagos） |

### gen_cases.py

```bash
python gen_cases.py [--models m1,m2] [--operators op1,op2] [--profile online_serving] [--list]
```

| 参数 | 说明 |
|------|------|
| `--models` | 指定模型（逗号分隔），默认全部 |
| `--operators` | 指定算子（逗号分隔），默认全部 |
| `--profile` | 部署场景，默认 online_serving |
| `--list` | 列出可用的模型和算子 |

输出：`cases/generated/merged/{operator}.yaml`

---

## 添加新算子

### Step 1：算子定义

创建 `operators/{op_name}/operator.py`：

```python
import torch
from framework.base_operator import BaseOperator
from framework.registry import register_operator

@register_operator("your_op_name")
class YourOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "your_op_name"

    @property
    def library(self) -> str:
        # 决定 flagos_provider 从哪个库加载
        # 可选: "flaggems_vllm" / "flaggems" / "flagattention"
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """根据 params 生成输入 tensor，返回 dict 作为算子的 kwargs"""
        M = params.get("M", 1024)
        N = params.get("N", 4096)
        x = torch.randn(M, N, dtype=torch.bfloat16, device="cuda")
        return {"x": x}

    def compute_flops(self, **params):
        """理论 FLOPs（用于计算 GFLOPS/s）"""
        M = params.get("M", 1024)
        N = params.get("N", 4096)
        return 2 * M * N

    def compute_bytes(self, **params):
        """理论访存量（字节，用于计算 bandwidth）"""
        M = params.get("M", 1024)
        N = params.get("N", 4096)
        return (M * N + M * N) * 2  # input + output, bf16=2bytes
```

创建 `operators/{op_name}/__init__.py`（空文件即可）。

### Step 2：注册参数映射

在 `operator_registry.yaml` 添加算子条目，定义从模型配置到算子参数的映射规则：

```yaml
your_op_name:
  library: flaggems_vllm
  function: your_op_name
  const_axes_rule: _rule_your_op
  applicable_models: all
```

### Step 3：Provider 加载

**flagos_provider.py** — 通常自动按 library 导入，无需改动。如果函数名与算子名不同，需在 `impl_map` 中显式指定。

**vllm_provider.py** — 添加 baseline 加载方法：

```python
# 在 impl_map 中注册
"your_op_name": (self._load_your_op, True),

# 实现加载
def _load_your_op(self):
    from some_baseline import baseline_fn
    def wrapper(**kwargs):
        return baseline_fn(**kwargs)
    return wrapper, {"source": "baseline description", "type": "cuda/triton"}
```

### Step 4：生成 Case 并测试

```bash
python gen_cases.py --operators your_op_name
python run.py --provider flagos --case cases/generated/merged/your_op_name.yaml
python run.py --provider vllm --case cases/generated/merged/your_op_name.yaml
python compare.py --op your_op_name --save
```

---

## 添加新模型

1. 在 `model_configs/` 创建 `{model_name}.json`，填入模型架构参数：

```json
{
  "model_name": "your_model",
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "num_key_value_heads": 8,
  "head_dim": 128,
  "intermediate_size": 14336,
  "num_experts": 0,
  "vocab_size": 128256
}
```

2. 重新生成 case：`python gen_cases.py --models your_model`

---

## Triton Allocator 注意事项

在 Triton 3.6+ 环境下，某些 kernel（特别是 TLE 优化路径）需要设置 runtime allocator：

```python
import triton

def alloc_fn(size, align, stream):
    return torch.empty(size, dtype=torch.int8, device="cuda")
triton.set_allocator(alloc_fn)
```

已在需要的 operator 的 `prepare_inputs` 中自动调用。如果新算子遇到 `Kernel requires a runtime memory allocation` 错误，加入此调用即可。

---

## 结果解读

### compare JSON 结构

```json
{
  "operator": "fused_moe",
  "baseline": "vllm",
  "target": "flagos",
  "workloads": [
    {
      "name": "decode_32tokens",
      "flagos_ms": 0.5647,
      "vllm_ms": 1.0221,
      "speedup": 1.812,
      "verdict": "faster"
    }
  ],
  "summary": {
    "avg_speedup": 1.224,
    "geo_mean_speedup": 1.138,
    "faster": 8,
    "slower": 2,
    "on_par": 1
  }
}
```

### 判断标准

| Speedup | 标记 | 含义 |
|---------|------|------|
| > 1.05 | ✓ faster | FlagOS 更快 |
| 0.95 ~ 1.05 | ≈ on par | 持平 |
| < 0.95 | ✗ slower | vLLM 更快 |

---

## 常见问题

| 问题 | 原因 | 解决 |
|------|------|------|
| `No impl for {op}` | flagos_provider 找不到对应函数 | 检查 operator.library 和实际 import 路径 |
| `Kernel requires runtime memory allocation` | Triton 3.6+ 需要 allocator | 在 prepare_inputs 中调用 `_ensure_triton_allocator()` |
| `CUDA illegal memory access` | kernel 参数超出支持范围 | 检查算子在该 seq_len/head_dim 下是否支持 |
| `_load_{op} not found` | vllm_provider 未注册该算子 | 在 vllm_provider.py 的 impl_map 中添加 |
| compare 无数据 | 缺少对应的 flagos/vllm JSON | 先执行 run.py 生成结果 |
| case YAML 为空 | 模型配置缺少该算子需要的字段 | 检查 model_configs 和 operator_registry.yaml 的映射规则 |
