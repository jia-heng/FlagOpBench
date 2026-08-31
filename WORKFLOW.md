# 算子性能测试全流程指南

以 `mhc_pre` 算子为完整示例，说明从算子接入到性能对比的全部步骤。

## 流程概览

```
1. 算子定义 (operators/mhc_pre/operator.py)
2. 注册到 operator_registry.yaml
3. 单模型用例生成
4. 生成测试用例 (gen_cases.py --merged)
5. 运行平台基线 benchmark (run.py --platform nvidia)
6. 运行 FlagOS benchmark (run.py --platform nvidia --impl flagos)
7. 生成对比结果 (scripts/gen_compare_result.py)
```

---

## Step 1: 算子定义

在 `operators/` 下创建目录和实现文件。

```
operators/{op_name}/
├── __init__.py       # 空文件或 from .operator import *
└── operator.py       # 算子定义
```

### operator.py 核心结构

```python
from framework.base_operator import BaseOperator
from framework.registry import register_operator
import torch

@register_operator("your_op_name")   # 注册名必须与目录名和YAML中的name一致
class YourOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "your_op_name"

    @property
    def library(self) -> str:
        """算子所属库，决定provider从哪里加载实现"""
        return "flaggems_vllm"   # flaggems / flaggems_vllm / flagattention

    def prepare_inputs(self, **params):
        """准备算子的输入张量
        
        参数从两个来源合并而来：
        1. const_axes (从model config通过param_mapping规则推导)
        2. var_axes (从profile中当前workload的变化维度)
        
        Args:
            **params: 包含shape、dtype等参数的字典
                常见参数：
                - num_tokens / N: token数量
                - hidden_size / H: 隐藏层维度
                - dtype: 数据类型字符串 (bf16/fp16/fp32)
                - 其他算子特定参数
        
        Returns:
            dict: key为kernel函数参数名，value为torch.Tensor
        """
        # 1. 提取参数（支持多种命名方式）
        N = params.get("N") or params.get("num_tokens")
        H = params.get("H") or params.get("hidden_size")
        dtype = self.get_dtype(params.get("dtype", "bfloat16"))
        
        # 2. 生成输入张量（全部放在CUDA上）
        input_tensor = torch.randn(N, H, dtype=dtype, device="cuda")
        
        # 3. 返回字典，key必须与kernel函数签名匹配
        return {
            "input": input_tensor,
            # ... 其他输入 ...
        }

    def compute_flops(self, **params) -> int:
        """计算理论FLOPs（浮点运算次数）
        
        用于性能分析中的TFLOPS计算：TFLOPS = FLOPs / (time_ms * 1e9)
        
        计算规则：
        - GEMM (M,K) @ (K,N): 2*M*K*N
        - Element-wise (add/mul/etc): N_elements
        - Reduction (sum/max): N_elements
        - Softmax: 3 * N_elements (exp + sum + div)
        
        Args:
            **params: 与prepare_inputs相同的参数
        
        Returns:
            int: 理论FLOPs总数
        """
        N = params.get("N") or params.get("num_tokens")
        H = params.get("H") or params.get("hidden_size")
        
        # 根据算子计算公式累加
        total_flops = N * H * 2  # 示例：一次加法一次乘法
        
        return int(total_flops)

    def compute_bytes(self, **params) -> int:
        """计算理论访存量（字节数）
        
        用于性能分析中的带宽计算：Bandwidth GB/s = Bytes / (time_ms * 1e6)
        
        计算规则：
        - 读：所有输入张量的总字节数
        - 写：所有输出张量的总字节数
        - 访存量 = 读 + 写
        
        注意：
        - GEMM中权重矩阵虽然走cuBLAS缓存，但仍计入（分析worst case）
        - 中间结果如果在kernel内不落地，不计入
        
        Args:
            **params: 与prepare_inputs相同的参数
        
        Returns:
            int: 理论访存量（Bytes）
        """
        N = params.get("N") or params.get("num_tokens")
        H = params.get("H") or params.get("hidden_size")
        elem_bytes = self.dtype_bytes(params.get("dtype", "bfloat16"))
        
        read_bytes = N * H * elem_bytes   # 输入
        write_bytes = N * H * elem_bytes  # 输出
        
        return int(read_bytes + write_bytes)
```

### 关键约定

1. **参数命名兼容性**
   - `prepare_inputs` 中用 `params.get("N") or params.get("num_tokens")` 兼容不同写法
   - 优先使用通用名：`num_tokens`、`hidden_size`、`dtype`
   - 模型特定参数可用简写：`H`、`hc`、`topk`

2. **返回值的 key 必须与 kernel 签名匹配**
   - 先检查 kernel 函数签名（在 vllm 或 flaggems_vllm 中查看）
   - 返回字典的 key 必须与 kernel 参数名完全一致
   - 示例：如果 kernel 是 `def mhc_post(x, residual, post_layer_mix, comb_res_mix)`，则返回字典必须有这4个key

3. **library 属性决定加载路径**
   - `flaggems` → flagos_provider 从 `flag_gems.{op_name}` 加载
   - `flaggems_vllm` → flagos_provider 从 `flaggems_vllm.{op_name}` 加载
   - `flagattention` → flagos_provider 从 `flag_attn.{op_name}` 加载
   - vllm_provider(nvidia_provider) 需单独配置（见 Step 3a）

4. **如果函数名与注册名不同**
   ```python
   @property
   def impl_name(self) -> str:
       return "actual_function_name"  # flagos_provider会用此名称getattr
   ```

5. **辅助方法（继承自 BaseOperator）**
   - `self.get_dtype(dtype_str)`: "bf16" → `torch.bfloat16`
   - `self.dtype_bytes(dtype_str)`: "bf16" → 2

6. **验证方法**
   ```bash
   python -c "from operators.{op_name} import *; print('Import OK')"
   ```

---

## Step 2: 注册到 operator_registry.yaml

在 `operator_registry.yaml` 的 `operators` 列表末尾添加新算子：

```yaml
operators:
  # ... 已有算子 ...
  
  - name: your_op_name           # 必须与Step 1中@register_operator()的名称一致
    library: flaggems_vllm        # 与operator.py中library属性一致
    status: implemented           # planned / implemented / deprecated
    param_mapping:
      const_axes_rule: xxx        # 参数映射规则名（见下文"如何选择"）
      var_axes_key: num_tokens    # 变化维度的key（见下文"可用var_axes_key"）
      applicable: has_moe         # 模型过滤条件（见下文"可用applicable"）
```

### 字段详解

**1. name**
- 算子的唯一标识符
- 必须与 `@register_operator("name")` 和目录名 `operators/name/` 保持一致
- 用于命令行 `--operators name` 和生成的文件名

**2. library**
- 算子实现所在的包
- 必须与 `operator.py` 中的 `library` 属性一致
- 可选值：`flaggems` / `flaggems_vllm` / `flagattention`

**3. status**
- `planned`: 规划中，casegen 跳过
- `implemented`: 已实现，casegen 会生成用例
- `deprecated`: 已废弃

**4. param_mapping.const_axes_rule**
- 从模型配置推导固定参数的规则名
- 规则函数定义在 `casegen/param_mapping.py` 的 `CONST_AXES_RULES` 中
- **如何选择**（见下文"可用 const_axes_rule"）

**5. param_mapping.var_axes_key**
- 从 `profiles/online_serving.yaml` 的 `var_axes` 中选取哪组变化维度
- **如何选择**（见下文"可用 var_axes_key"）

**6. param_mapping.applicable**
- 模型过滤条件，只有满足条件的模型才生成测试用例
- 条件函数定义在 `casegen/param_mapping.py` 的 `APPLICABLE_CONDITIONS` 中
- **如何选择**（见下文"可用 applicable"）

---

### 可用 const_axes_rule

当前已定义的规则（`casegen/param_mapping.py` 中）：

| Rule名 | 返回参数 | 适用算子类型 | 示例 |
|--------|---------|------------|------|
| `moe` | `hidden_size`, `intermediate_size`, `num_experts`, `topk`, `dtype` | MoE 相关 | fused_moe |
| `group_gemm_down` | `K`, `N`, `dtype` | MoE down projection | — |
| `group_gemm_gate_up` | `K`, `N`, `dtype` | MoE gate_up projection | — |
| `flash_mla` | `s_q`, `h_q`, `h_kv`, `d`, `dv`, `block_size` | MLA attention | flash_mla |
| `flash_mla_fp8` | 同上 + `quant=fp8` | MLA FP8量化 | flash_mla_with_kvcache_fp8 |
| `flash_attn` | `num_heads`, `head_dim`, `causal`, `dtype` | 标准 Flash Attention | — |
| `swiglu` | `N`, `dtype` | SwiGLU 激活 | swiglu |
| `silu_and_mul` | `N`, `dtype` | SiLU+Mul 激活 | silu_and_mul_with_clamp |
| `fused_q_kv_rmsnorm` | `hidden_size`, `num_heads`, `head_dim`, `q_lora_rank`, `kv_lora_rank`, `dtype` | MLA QKV归一化 | — |
| `moe_sum` | `hidden_size`, `topk`, `dtype` | MoE 输出求和 | moe_sum |
| `topk_softplus_sqrt` | `num_experts`, `topk`, `dtype` | MoE routing | — |
| `top_k_per_row` | `num_experts`, `topk` | MoE topk选择 | top_k_per_row_decode/prefill |
| `mhc` | `H`, `hc=4`, `dtype` | Multi-Head Cache | mhc_pre, mhc_post |
| `sparse_attn` | `index_topk`, `index_n_heads`, `index_head_dim`, `sliding_window`, `dtype` | 稀疏注意力 | — |
| `pack_unpack_seq` | `index_topk`, `dtype` | 序列打包/解包 | pack_seq_triton, unpack_seq_triton |
| `indexer_cache` | `index_n_heads`, `index_head_dim`, `index_topk`, `dtype` | 索引器缓存 | indexer_k_quant_and_cache |
| `paged_mqa` | `h_q`, `d`, `dv`, `block_size`, `dtype` | Paged MQA | fp8_fp4_paged_mqa_logits |
| `v4_fused_qkv` | `hidden_size`, `num_heads`, `q_lora_rank`, `kv_lora_rank`, `qk_rope_head_dim`, `dtype` | DeepSeek V4 QKV融合 | — |
| `causal_conv1d` | `hidden_size`, `dtype` | 因果卷积 | — |
| `generic` | `dtype` | 通用fallback | — |

**如何选择：**
1. 如果算子类型与上表某一行匹配，直接用对应的 rule 名
2. 如果需要新的参数组合，需要在 `casegen/param_mapping.py` 中添加新规则（见 Step 3）
3. 如果只需要 dtype，用 `generic`

---

### 可用 var_axes_key

从 `profiles/online_serving.yaml` 中选取（当前支持5种）：

| Key | 类型 | 维度数 | 示例值 | 适用场景 |
|-----|------|--------|--------|---------|
| `num_tokens` | 标量列表 | 10 | `[1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096]` | 逐token算子（MoE, FFN, Norm, MHC等） |
| `moe_dispatch` | 字典列表 | 10 | `{num_groups: 8, tokens_per_group: 4}` | MoE dispatch专用 |
| `batch_seq_decode` | 字典列表 | 7 | `{b: 1, max_seq: 2048}` | Decode阶段attention (KV cache) |
| `batch_seq_prefill` | 字典列表 | 6 | `{batch_size: 1, seqlen_q: 512, seqlen_k: 512}` | Prefill阶段attention |
| `batch_seq` | 字典列表 | 7 | `{batch_size: 1, seqlen_q: 1, seqlen_k: 2048}` | 通用batch+seq场景（FA varlen等） |

**如何选择：**
- 算子输入主要维度是 token 数 → `num_tokens`
- MoE dispatch 相关 → `moe_dispatch`
- Attention decode → `batch_seq_decode`
- Attention prefill → `batch_seq_prefill`
- 其他 attention 变体 → `batch_seq`

---

### 可用 applicable

模型过滤条件（`casegen/param_mapping.py` 中）：

| 条件名 | Lambda 函数 | 匹配模型数 | 说明 |
|--------|------------|-----------|------|
| `always` | `lambda m: True` | 全部 (12) | 所有模型都适用 |
| `has_moe` | `lambda m: m.has_moe` | 9 | 有路由专家（`n_routed_experts > 0`） |
| `has_mla` | `lambda m: m.has_mla` | 9 | 有MLA（`kv_lora_rank > 0` 或 DeepSeek-V4风格） |
| `has_sparse_attn` | `lambda m: m.index_topk is not None` | 5 | 有稀疏注意力（indexer） |
| `has_linear_attn` | `lambda m: m.linear_attn_config is not None` | 待确认 | 有线性注意力配置 |

**如何选择：**
- MoE 相关算子 → `has_moe`
- MLA 相关算子 → `has_mla`
- 稀疏注意力相关 → `has_sparse_attn`
- 通用算子（FFN、Norm等） → `always`
- 如需新条件，在 `APPLICABLE_CONDITIONS` 中添加

---

### 验证方法

```bash
python gen_cases.py --list
# 输出应包含新算子名，且status为implemented
```

---

## Step 3: 添加参数映射规则（如需新规则）

如果现有的 `const_axes_rule` 不满足需求，需要在 `casegen/param_mapping.py` 中添加新规则。

### 何时需要新规则

- 算子需要的参数组合在现有规则中不存在
- 示例：新算子需要 `{hidden_size, num_layers, dtype}` 但现有规则都没有 `num_layers`

### 添加步骤

**1. 在 `param_mapping.py` 中定义规则函数**

```python
def _rule_your_new_rule(model: ModelParams) -> dict:
    """规则函数：从ModelParams推导算子所需的固定参数
    
    Args:
        model: ModelParams对象，包含模型配置的所有字段
              常用字段：
              - model.hidden_size
              - model.intermediate_size
              - model.num_attention_heads
              - model.num_key_value_heads
              - model.head_dim
              - model.n_routed_experts
              - model.num_experts_per_tok
              - model.moe_intermediate_size
              - model.q_lora_rank / kv_lora_rank
              - model.index_topk / index_n_heads / index_head_dim
              - model.dtype_short  (bf16/fp16/fp32)
              - model.has_moe (bool property)
              - model.has_mla (bool property)
    
    Returns:
        dict: 参数字典，key为算子期望的参数名
    """
    return {
        "H": model.hidden_size,
        "num_layers": model.num_hidden_layers,
        "dtype": model.dtype_short,
    }
```

**2. 注册到 `CONST_AXES_RULES` 字典**

在文件中找到 `CONST_AXES_RULES` 定义（约在第195行），添加新条目：

```python
CONST_AXES_RULES: dict[str, Callable[[ModelParams], dict]] = {
    # ... 已有规则 ...
    "your_new_rule": _rule_your_new_rule,
}
```

**3. 在 `operator_registry.yaml` 中使用**

```yaml
- name: your_op_name
  param_mapping:
    const_axes_rule: your_new_rule  # 使用新规则
```

### 验证方法

```bash
python -c "
from casegen.param_mapping import get_const_axes
from casegen.model_parser import parse_model_config
from pathlib import Path

# 测试单个模型
model = parse_model_config(Path('model_configs/deepseek_v3.2.json'))
axes = get_const_axes('your_op_name', model)
print(f'Model: {model.name}')
print(f'Const axes: {axes}')
"
```

---

## Step 3a: 配置平台 Provider 加载方法

NV 平台的算子分布在 vLLM 多个子模块中，需要手动添加加载方法。

### 添加步骤

**1. 确定 vLLM 中的算子位置**

```bash
# 在vllm仓库中搜索算子名
cd /data/jianheng/works/Flagos/vllm
grep -r "def your_op_name" --include="*.py"
# 或者
find . -name "*.py" | xargs grep "def your_op_name"
```

假设找到：`vllm/model_executor/layers/your_module.py` 中的 `your_op_name` 函数

**2. 在 `providers/nvidia_provider.py` 中添加加载方法**

找到类似方法的位置，添加：

```python
def _load_your_op_name(self):
    """加载vLLM的your_op_name实现
    
    Returns:
        Tuple[Callable, dict]: (函数对象, 元信息字典)
            元信息包含：source (导入路径), type (实现类型)
    """
    # 检查对应模块是否已import
    if self._vllm_your_module is None:
        return None, {}
    
    # 返回函数和元信息
    return self._vllm_your_module.your_op_name, {
        "source": "vllm.model_executor.layers.your_module.your_op_name",
        "type": "tilelang"  # 或 "cuda" / "triton"
    }
```

**3. 注册到 `impl_map`**

在 `get_impl` 方法的 `impl_map` 字典中添加：

```python
impl_map = {
    # ... 已有算子 ...
    "your_op_name": (self._load_your_op_name, False),  # False表示签名兼容，不需wrapper
}
```

如果算子签名与 flagos 不兼容，需要 wrapper：

```python
"your_op_name": (self._load_your_op_name, True),  # True表示需要wrapper
```

并实现对应的 wrapper 方法（参考 `_load_swiglu`、`_load_silu_and_mul_with_clamp` 等）。

**4. 如果需要新的模块 import**

在 `setup()` 方法中添加：

```python
def setup(self):
    # ... 已有 import ...
    
    # 你的新模块
    try:
        from vllm.model_executor.layers import your_module
        self._vllm_your_module = your_module
    except ImportError:
        pass
    
    print(f"  Loaded vllm modules: ..., your_module={self._vllm_your_module is not None}")
```

### 常见 vLLM 模块位置

| 算子类型 | vLLM 模块路径 | 变量名 |
|---------|--------------|--------|
| MoE 相关 | `vllm._custom_ops` | `self._vllm_ops` |
| MHC | `vllm.model_executor.layers.mhc` | `self._vllm_mhc` |
| 稀疏注意力 | `vllm.model_executor.layers.sparse_attn_indexer` | `self._vllm_sparse_attn` |
| Flash Attention | `vllm.vllm_flash_attn.flash_attn_interface` | `self._vllm_flash_attn` |
| Fused MoE | `vllm.model_executor.layers.fused_moe` | `self._vllm_fused_moe` |
| V1 ops | `vllm.v1.worker.gpu_model_runner._custom_ops` | `self._vllm_v1_ops` |

### 验证方法

```bash
python -c "
from providers.nvidia_provider import NvidiaProvider
from operators.your_op_name import YourOperator

provider = NvidiaProvider()
provider.setup()

op = YourOperator()
impl, info = provider.get_impl('your_op_name', op)

if impl is None:
    print(f'ERROR: {info}')
else:
    print(f'OK: {info}')
"
```

---

## Step 4: 生成测试用例

```bash
# 生成 merged 格式（推荐，一个算子一个文件，跨模型聚合）
python gen_cases.py --operators your_op_name --merged

# 输出: cases/generated/merged/your_op_name.yaml
```

### Merged 格式结构

```yaml
operator: your_op_name
library: flaggems_vllm
sections:
- const_axes: {H: 7168, hc: 4, dtype: bf16}
  models: [deepseek_v3.2, deepseek_v4_pro, kimi_k2.6, kimi_k3]
  workloads:
  - name: deepseek_v3.2_decode_num_tokens1
    source: '[deepseek_v3.2 | deepseek_v4_pro | kimi_k2.6 | kimi_k3] ...'
    var_axes: {num_tokens: 1}
  - name: deepseek_v3.2_decode_num_tokens8
    var_axes: {num_tokens: 8}
  # ... 共10个workload
- const_axes: {H: 4096, hc: 4, dtype: bf16}
  models: [deepseek_v4_flash]
  workloads: [...]
```

### 关键机制

**1. Section 分组逻辑**
- 相同 `const_axes` 的模型合并为一个 section
- 每个 section 的 workload 数量 = `len(profile.var_axes[var_axes_key])`
- 示例：`var_axes_key=num_tokens` → 10 个 workload（1, 8, 32, 64, 128, 256, 512, 1024, 2048, 4096）

**2. Workload 命名规则**
- 格式：`{首个模型名}_{阶段}_{var_axis_name}{value}`
- 阶段自动推断（对 `num_tokens`）：
  - 1-64 → `decode`
  - 128-256 → `mixed`
  - 512+ → `prefill`
- 示例：`deepseek_v3.2_decode_num_tokens1`、`glm_5.2_prefill_num_tokens2048`

**3. Source 字段**
- 标注该 workload 适用的模型列表
- 格式：`[model1 | model2 | ...] {原始来源}, {阶段}, {var_axes描述}`
- 用途：文档化，runner 不依赖此字段

### 验证生成结果

```bash
# 1. 检查文件是否生成
ls -lh cases/generated/merged/your_op_name.yaml

# 2. 查看 section 数量和模型分组
python -c "
import yaml
with open('cases/generated/merged/your_op_name.yaml') as f:
    data = yaml.safe_load(f)
print(f'Operator: {data[\"operator\"]}')
print(f'Library: {data[\"library\"]}')
print(f'Total sections: {len(data[\"sections\"])}')
print()
for i, sec in enumerate(data['sections'], 1):
    print(f'Section {i}:')
    print(f'  const_axes: {sec[\"const_axes\"]}')
    print(f'  models ({len(sec[\"models\"])}): {sec[\"models\"]}')
    print(f'  workloads: {len(sec[\"workloads\"])}')
"

# 3. 验证参数完整性（检查第一个workload能否实例化）
python -c "
import yaml
from operators.your_op_name import YourOperator

with open('cases/generated/merged/your_op_name.yaml') as f:
    data = yaml.safe_load(f)

op = YourOperator()
sec = data['sections'][0]
wl = sec['workloads'][0]

# 合并 const_axes + var_axes
params = {**sec['const_axes'], **wl['var_axes']}
print(f'Testing workload: {wl[\"name\"]}')
print(f'Merged params: {params}')

try:
    inputs = op.prepare_inputs(**params)
    flops = op.compute_flops(**params)
    bytes_val = op.compute_bytes(**params)
    print(f'✅ prepare_inputs OK, {len(inputs)} tensors')
    print(f'✅ compute_flops OK: {flops:,}')
    print(f'✅ compute_bytes OK: {bytes_val:,}')
except Exception as e:
    print(f'❌ Error: {e}')
"
```

### 常见问题排查

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 生成的 section 数为 0 | `applicable` 条件过滤掉了所有模型 | 检查 `operator_registry.yaml` 的 `applicable` 是否正确 |
| 某些模型缺失 | 模型配置中缺少 `const_axes_rule` 需要的字段 | 检查 `model_configs/{model}.json`，或调整 `_rule_xxx` 使用 fallback 值 |
| workload 数量不对 | `var_axes_key` 指向的 profile 条目数量 | 检查 `profiles/online_serving.yaml` 中对应 key 的长度 |
| `prepare_inputs` 报错 | 参数名不匹配 | 确保 `const_axes_rule` 返回的 key 与 `prepare_inputs` 中 `params.get()` 的名称一致 |

---

## Step 5: 运行 benchmark

```bash
# 平台基线 (nvidia vllm kernel，优先 vllm > torch)
python run.py --platform nvidia --case cases/generated/merged/your_op_name.yaml

# FlagOS 实现
python run.py --platform nvidia --impl flagos --case cases/generated/merged/your_op_name.yaml

# 或直接对比模式（只打印表格，不存JSON）
python run.py --platform nvidia --mode compare --case cases/generated/merged/your_op_name.yaml
```

### 输出文件

按算子建子目录，flagos 文件名带 platform 后缀：

```
results/{op}/
├── {op}_nvidia.json              # 平台基线
├── {op}_flagos_nvidia.json       # FlagOS 在该平台的性能
└── {op}_compare_nvidia.json      # 对比结果 (由 gen_compare_result.py 生成)
```

多平台示例：
```
results/swiglu/
├── swiglu_nvidia.json
├── swiglu_flagos_nvidia.json
├── swiglu_compare_nvidia.json
├── swiglu_ascend.json            # 昇腾基线
├── swiglu_flagos_ascend.json     # FlagOS 在昇腾
└── swiglu_compare_ascend.json
```

### 兼容旧用法

```bash
# 以下仍可工作（自动映射到 --platform nvidia）
python run.py --provider vllm --case cases/generated/merged/your_op_name.yaml
python run.py --provider flagos --case cases/generated/merged/your_op_name.yaml
```

### runner 执行流程

对于 merged 格式的 case，runner 逐 section 执行：

1. **读取 const_axes**
   - 从 section 中获取固定参数（如 `{H: 7168, hc: 4, dtype: bf16}`）

2. **逐 workload 执行**
   - 合并参数：`merged_params = {**const_axes, **workload.var_axes}`
   - 调用 `operator.prepare_inputs(**merged_params)` 生成输入张量
   - 通过 provider 获取 kernel 函数：`provider.get_impl(op_name, operator)`
   - Warmup：运行 `--warmup` 次（默认 10），丢弃结果
   - 计时：运行 `--repeat` 次（默认 100），记录每次耗时
   - 计算统计：mean, median, std, min, max

3. **记录结果**
   - 每个 workload 一条记录，包含：
     - 时间统计（ms）
     - 理论 FLOPs / Bytes（调用 `operator.compute_flops/bytes(**merged_params)`）
     - 实测性能指标：TFLOPS, Bandwidth GB/s
     - 实现信息：source, type（从 provider 返回）

### 参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--platform` | nvidia | 目标平台: nvidia / ascend / metax / mthreads / iluvatar |
| `--impl` | — | 实现: 默认用平台基线，`flagos` 用 FlagOS |
| `--mode` | single | `single`(默认) / `compare`(对比 FlagOS vs 基线) |
| `--case` | 必填 | 单个 YAML 文件路径 |
| `--case-dir` | — | 批量运行目录下所有 YAML |
| `--warmup` | 10 | 预热次数（用于 JIT 编译、缓存预热） |
| `--repeat` | 100 | 重复计时次数（取平均值） |
| `--output` | `results/` | 结果输出根目录 |
| `--provider` | — | [兼容旧用法] vllm→平台基线, flagos→FlagOS |

### 常见问题排查

**1. Provider 加载失败**
```
ERROR: No impl for your_op_name in flaggems_vllm
```
- **原因**：`flaggems_vllm.your_op_name` 不存在
- **解决**：检查 `operator.library` 是否正确，或在 operator 中设置 `impl_name` 属性

**2. vLLM TileLang JIT 编译**
```
[INFO] Compiling TileLang kernel for config {H=7168, hc=4, ...} (6.2s)
```
- **原因**：vLLM 某些算子（mhc_pre/post, sparse_attn 等）首次运行新配置时触发 JIT
- **影响**：总运行时间变长（每个新配置 ~6s），但不影响计时结果（warmup 跳过）
- **解决**：正常现象，无需处理

**3. prepare_inputs 参数缺失**
```
KeyError: 'H'
```
- **原因**：`operator.prepare_inputs(**params)` 中访问了不存在的 key
- **解决**：
  - 检查 `const_axes_rule` 是否返回了所需参数
  - 使用 `params.get("H")` 而非 `params["H"]`，提供 fallback 值

**4. Kernel 函数签名不匹配**
```
TypeError: your_op_name() got an unexpected keyword argument 'xxx'
```
- **原因**：`prepare_inputs` 返回的 key 与 kernel 函数参数名不匹配
- **解决**：检查 vllm/flaggems_vllm 源码中的函数签名，调整返回字典的 key

**5. CUDA OOM**
```
torch.cuda.OutOfMemoryError
```
- **原因**：输入张量过大（通常是 prefill 阶段的大 seq_len）
- **解决**：减少 `--repeat` 次数，或跳过超大 workload

### 调试技巧

**单独测试某个 workload：**
```python
import yaml
import torch
from operators.your_op_name import YourOperator
from providers.flagos_provider import FlagOSProvider

# 加载case
with open('cases/generated/merged/your_op_name.yaml') as f:
    data = yaml.safe_load(f)

# 准备算子和provider
op = YourOperator()
provider = FlagOSProvider()
provider.setup()

# 选择第一个section的第一个workload
sec = data['sections'][0]
wl = sec['workloads'][0]
params = {**sec['const_axes'], **wl['var_axes']}

print(f"Testing: {wl['name']}")
print(f"Params: {params}")

# 准备输入
inputs = op.prepare_inputs(**params)
print(f"Inputs: {list(inputs.keys())}")

# 获取kernel
impl, info = provider.get_impl(data['operator'], op)
if impl is None:
    print(f"ERROR: {info}")
else:
    print(f"Impl: {info}")
    # 运行一次
    with torch.no_grad():
        out = impl(**inputs)
    print(f"Output: {out.shape if hasattr(out, 'shape') else type(out)}")
```

---

## Step 6: 生成对比结果

```bash
# 从两份结果JSON生成对比JSON
python scripts/gen_compare_result.py \
    --baseline results/your_op_name/your_op_name_nvidia.json \
    --flagos results/your_op_name/your_op_name_flagos_nvidia.json

# 输出: results/your_op_name/your_op_name_compare_nvidia.json
```

### 对比结果文件结构

`results/{op}/{op}_compare_{platform}.json`:
```json
{
  "metadata": {
    "timestamp": "2026-08-27T06:16:19.846569",
    "type": "compare",
    "platform": "nvidia",
    "baseline_provider": "nvidia",
    "flagos_provider": "flagos",
    "num_workloads": 60,
    "baseline_file": "results/swiglu/swiglu_nvidia.json",
    "flagos_file": "results/swiglu/swiglu_flagos_nvidia.json"
  },
  "environment": { ... },
  "summary": {
    "total": 60,
    "faster": 0,
    "slower": 59,
    "on_par": 1,
    "avg_speedup": 0.5843
  },
  "comparisons": [
    {
      "operator": "swiglu",
      "workload": "deepseek_v3.2_decode_num_tokens1",
      "parameters": {"N": 2048, "dtype": "bf16", "num_tokens": 1},
      "baseline": {
        "provider": "nvidia",
        "mean_ms": 0.0187,
        "std_ms": 0.0026,
        "bandwidth_gb_s": 0.66,
        "gflops": 0.55,
        "impl_source": "torch.ops._C.silu_and_mul (adapted)"
      },
      "flagos": {
        "provider": "flagos",
        "mean_ms": 0.0364,
        "std_ms": 0.0041,
        "bandwidth_gb_s": 0.34,
        "gflops": 0.28,
        "impl_source": "flag_gems.swiglu"
      },
      "speedup": 0.5137,
      "verdict": "slower"
    }
  ]
}
```

### 符号说明

| 符号 | 含义 | 条件 |
|------|------|------|
| `✓` | flagos 更快 | `speedup > 1.05` (快 5% 以上) |
| `✗` | flagos 更慢 | `speedup < 0.95` (慢 5% 以上) |
| `≈` | 性能相当 | `0.95 <= speedup <= 1.05` (差距在 5% 以内) |

### 关键指标

- **Speedup**: `baseline_time / flagos_time`
  - `> 1.0` 表示 flagos 快于平台基线
  - `< 1.0` 表示 flagos 慢于平台基线
  - `= 1.0` 表示性能相同

- **Verdict**
  - `faster`: speedup > 1.05
  - `slower`: speedup < 0.95
  - `on_par`: 0.95 ≤ speedup ≤ 1.05

### 对比结果文件结构

见上方 Step 6 输出格式。

### 分析方法

**1. 查看 summary**
```bash
python -c "
import json
with open('results/your_op_name/your_op_name_compare_nvidia.json') as f:
    data = json.load(f)
s = data['summary']
print(f'Total: {s[\"total\"]} | Faster: {s[\"faster\"]} | Slower: {s[\"slower\"]} | On par: {s[\"on_par\"]}')
print(f'Avg speedup: {s[\"avg_speedup\"]:.4f}x')
"
```

**2. 按阶段分析**
```bash
python -c "
import json, statistics
with open('results/your_op_name/your_op_name_compare_nvidia.json') as f:
    data = json.load(f)

decode = [c['speedup'] for c in data['comparisons'] if 'decode' in c['workload']]
mixed = [c['speedup'] for c in data['comparisons'] if 'mixed' in c['workload']]
prefill = [c['speedup'] for c in data['comparisons'] if 'prefill' in c['workload']]

if decode: print(f'Decode avg: {sum(decode)/len(decode):.3f}x')
if mixed: print(f'Mixed avg: {sum(mixed)/len(mixed):.3f}x')
if prefill: print(f'Prefill avg: {sum(prefill)/len(prefill):.3f}x')
"
```

**3. 找出性能差距最大的 workload**
```bash
python -c "
import json
with open('results/your_op_name/your_op_name_compare_nvidia.json') as f:
    data = json.load(f)

comps = sorted(data['comparisons'], key=lambda x: x['speedup'])
print('Top 5 slowest (flagos vs baseline):')
for c in comps[:5]:
    print(f'  {c[\"workload\"]}: {c[\"speedup\"]:.3f}x')

print('\nTop 5 fastest (flagos vs baseline):')
for c in comps[-5:]:
    print(f'  {c[\"workload\"]}: {c[\"speedup\"]:.3f}x')
"
```

---

## 完整命令清单（一键复制）

### 单算子流程

```bash
cd /data/jianheng/works/Flagos/Flagtests

# 1. 生成 case
python gen_cases.py --operators your_op_name --merged

# 2. 跑平台基线
python run.py --platform nvidia --case cases/generated/merged/your_op_name.yaml

# 3. 跑 FlagOS
python run.py --platform nvidia --impl flagos --case cases/generated/merged/your_op_name.yaml

# 4. 生成对比结果
python scripts/gen_compare_result.py \
    --baseline results/your_op_name/your_op_name_nvidia.json \
    --flagos results/your_op_name/your_op_name_flagos_nvidia.json
```

### 批量跑所有算子

```bash
cd /data/jianheng/works/Flagos/Flagtests

# 1. 生成所有算子的 case
python gen_cases.py --operators all --merged

# 2. 批量跑平台基线
python run.py --platform nvidia --case-dir cases/generated/merged/

# 3. 批量跑 FlagOS
python run.py --platform nvidia --impl flagos --case-dir cases/generated/merged/

# 4. 批量生成对比结果
for op_dir in results/*/; do
    op=$(basename "$op_dir")
    baseline="$op_dir/${op}_nvidia.json"
    flagos="$op_dir/${op}_flagos_nvidia.json"
    if [ -f "$baseline" ] && [ -f "$flagos" ]; then
        python scripts/gen_compare_result.py --baseline "$baseline" --flagos "$flagos"
    fi
done
```

### 其他平台

```bash
# 昇腾平台
python run.py --platform ascend --case cases/generated/merged/your_op_name.yaml
python run.py --platform ascend --impl flagos --case cases/generated/merged/your_op_name.yaml
python scripts/gen_compare_result.py \
    --baseline results/your_op_name/your_op_name_ascend.json \
    --flagos results/your_op_name/your_op_name_flagos_ascend.json
```

---

## 新算子 Checklist

| # | 步骤 | 文件/命令 | 验证方法 | 常见问题 |
|---|------|---------|---------|---------|
| 1 | 创建算子目录和 operator.py | `operators/{op}/operator.py` | `python -c "from operators.{op} import *"` | ImportError: 检查 `__init__.py` 是否正确导入 |
| 2 | 注册到 registry | `operator_registry.yaml` | `python gen_cases.py --list` 看到该算子 | 算子未出现: 检查 YAML 语法，status 是否为 implemented |
| 3 | 添加参数映射规则（如需） | `casegen/param_mapping.py` | `python -c "from casegen.param_mapping import get_const_axes, CONST_AXES_RULES; print(CONST_AXES_RULES.keys())"` | KeyError: 规则名拼写错误或未注册到 CONST_AXES_RULES |
| 3a | 配置 NV Provider | `providers/nvidia_provider.py` | `python -c "from providers.nvidia_provider import NvidiaProvider; p=NvidiaProvider(); p.setup(); print('OK')"` | ModuleNotFoundError: vLLM 模块导入失败，检查 setup() 中的 try-except |
| 4 | 生成 merged case | `python gen_cases.py --operators {op} --merged` | 检查 `cases/generated/merged/{op}.yaml` 存在且 workloads > 0 | 0 workloads: applicable 条件过滤掉了所有模型 |
| 5 | 跑平台基线 | `python run.py --platform nvidia --case cases/generated/merged/{op}.yaml` | exit code 0，`results/{op}/{op}_nvidia.json` 存在 | No impl for {op}: 检查 nvidia_provider.py 的 impl_map |
| 6 | 跑 FlagOS | `python run.py --platform nvidia --impl flagos --case cases/generated/merged/{op}.yaml` | exit code 0，`results/{op}/{op}_flagos_nvidia.json` 存在 | No impl for {op}: 检查 operator.library 和 flaggems_vllm 是否有该函数 |
| 7 | 生成对比 | `python scripts/gen_compare_result.py --baseline results/{op}/{op}_nvidia.json --flagos results/{op}/{op}_flagos_nvidia.json` | `results/{op}/{op}_compare_nvidia.json` 有 speedup 数据 | File not found: 确保 Step 5/6 都成功运行 |
| 8 | 更新进展表 | `WORKFLOW.md` 底部 | 在进展表添加一行记录 avg speedup | — |

---

## 注意事项

### 1. vLLM TileLang JIT 编译

vLLM 的 TileLang 算子（mhc_pre/post、sparse_attn 等）首次运行新配置时会触发即时编译：

```
[INFO] Compiling TileLang kernel for config {...} (6.2s)
```

- **影响**：总运行时间变长（每个新配置 ~6s），但不影响计时结果（warmup 阶段跳过）
- **原因**：TileLang 为每组参数组合（H, hc, dtype 等）生成专用 kernel
- **处理**：正常现象，无需干预

### 2. Dtype 一致性

大部分算子输入为 bfloat16，但某些张量（权重、mix 系数等）可能为 float32 或 fp8：

```python
def prepare_inputs(self, **params):
    dtype = self.get_dtype(params.get("dtype", "bfloat16"))
    
    # 数据张量用 dtype
    x = torch.randn(N, H, dtype=dtype, device="cuda")
    
    # 权重/系数可能固定为 float32
    weights = torch.randn(N, K, dtype=torch.float32, device="cuda")
```

- 参考 vLLM/flaggems_vllm 源码确认每个输入的实际 dtype
- `compute_bytes` 中分别计算不同 dtype 的访存量

### 3. 参数传递链

完整的参数流动路径：

```
model_configs/{model}.json
    ↓ (ModelParams 解析)
casegen/model_parser.py
    ↓ (const_axes_rule 推导)
casegen/param_mapping.py
    ↓ (生成 const_axes)
cases/generated/merged/{op}.yaml
    ↓ (runner 合并 const_axes + var_axes)
framework/runner.py
    ↓ (传递给 operator)
operators/{op}/operator.py::prepare_inputs(**params)
```

- 每个环节的参数名必须保持一致
- 使用 `params.get("H") or params.get("hidden_size")` 容错不同命名
- 调试时打印 `params` 查看实际传入的 key

### 4. 跨模型去重（Merged 格式的核心机制）

`gen_cases.py --merged` 的合并逻辑：

1. 对每个模型，调用 `get_const_axes(op_name, model)` 得到固定参数
2. 相同 `const_axes` 的模型分到同一个 section
3. 每个 section 生成一组 workload（共享 const_axes，仅 var_axes 变化）

**好处**：
- 避免重复测试（4 个模型 H=7168 只跑一次，而非 4 次）
- 减少总测试时间
- section 的 `models` 字段记录适用模型列表

**限制**：
- const_axes 必须完全相同才能合并（包括 dtype）
- 如果某模型缺少某个字段，会产生不同的 const_axes，单独成一个 section

### 5. Provider Dispatch 机制

**Provider 注册 (providers/registry.py)**:
```python
# 使用装饰器注册 provider
@register_provider("nvidia", platform="nvidia", is_default=True)
class NvidiaProvider(BaseProvider): ...

@register_provider("flagos", platform="all")  # 跨平台
class FlagOSProvider(BaseProvider): ...

# 运行时通过 factory 获取
provider = get_provider("nvidia")         # → NvidiaProvider (平台默认)
provider = get_provider("nvidia", "flagos")  # → FlagOSProvider
```

**FlagOS Provider**（简单）：
```python
# flagos_provider.py 通过 operator.library 自动 getattr
lib = operator.library  # "flaggems_vllm"
fn_name = getattr(operator, "impl_name", op_name)  # 默认用 op_name
fn = getattr(flaggems_vllm, fn_name)  # 直接从包获取
```

**NV Provider**（复杂）：
```python
# nvidia_provider.py 需要手动为每个算子写 _load_{op} 方法
# 因为 vLLM 算子分散在多个子模块中：
# - _custom_ops (MoE)
# - layers.mhc (MHC)
# - layers.sparse_attn_indexer (稀疏注意力)
# - vllm_flash_attn (Flash Attention)
# - layers.fused_moe (Fused MoE)
```

### 6. 如何确定函数名与注册名不同

某些情况下，算子在 flaggems_vllm 中的函数名与注册名不同：

```python
# 示例：注册名为 flash_attn_varlen，但实现函数名为 flash_attn_varlen_func
@register_operator("flash_attn_varlen")
class FlashAttnVarlenOperator(BaseOperator):
    
    @property
    def impl_name(self) -> str:
        return "flash_attn_varlen_func"  # flagos_provider 会用此名称
```

- 先在 flaggems_vllm 或 vllm 中搜索函数名
- 如果与注册名不同，设置 `impl_name` 属性

### 7. CUDA 内存管理

大 workload（prefill 阶段，num_tokens=4096）可能导致 OOM：

```python
# 每个 workload 运行后释放缓存（runner 已自动处理）
torch.cuda.empty_cache()
```

如果仍然 OOM：
- 减少 `--repeat` 次数（从 10 降到 5）
- 跳过特别大的 workload（编辑 YAML，删除对应条目）

### 8. 模型配置的多模态嵌套

kimi_k2.6/k3、qwen3.5/3.6/3.8_27b 等多模态模型的文本配置在 `text_config` 子字段下：

```json
{
  "model_type": "kimi",
  "text_config": {
    "hidden_size": 7168,
    "num_attention_heads": 56,
    ...
  },
  "vision_config": {...}
}
```

`model_parser.py` 已自动处理：
```python
data = raw.get("text_config", raw) if "text_config" in raw else raw
```

无需手动处理，但调试时注意这一点。

---

## 调试技巧

### 1. 单步验证算子实现

```python
import torch
from operators.your_op_name import YourOperator

op = YourOperator()

# 模拟一个 workload 的参数
params = {
    "num_tokens": 128,
    "H": 7168,
    "hc": 4,
    "dtype": "bf16"
}

# 测试 prepare_inputs
inputs = op.prepare_inputs(**params)
print(f"Inputs: {inputs.keys()}")
for k, v in inputs.items():
    print(f"  {k}: {v.shape}, {v.dtype}")

# 测试 compute_flops/bytes
flops = op.compute_flops(**params)
bytes_val = op.compute_bytes(**params)
print(f"FLOPs: {flops:,}")
print(f"Bytes: {bytes_val:,}")
```

### 2. 测试 provider 加载

```python
from providers.nvidia_provider import NvidiaProvider
from providers.flagos_provider import FlagOSProvider
from operators.your_op_name import YourOperator

op = YourOperator()

# NV 基线
print("=== Nvidia (baseline) ===")
p_nvidia = NvidiaProvider()
p_nvidia.setup()
impl, info = p_nvidia.get_impl("your_op_name", op)
print(f"Impl: {impl}")
print(f"Info: {info}")

# FlagOS
print("\n=== FlagOS ===")
p_flagos = FlagOSProvider()
p_flagos.setup()
impl, info = p_flagos.get_impl("your_op_name", op)
print(f"Impl: {impl}")
print(f"Info: {info}")
```

### 3. 检查生成的 case 结构

```python
import yaml

with open('cases/generated/merged/your_op_name.yaml') as f:
    data = yaml.safe_load(f)

print(f"Operator: {data['operator']}")
print(f"Sections: {len(data['sections'])}")

for i, sec in enumerate(data['sections'], 1):
    print(f"\nSection {i}:")
    print(f"  const_axes: {sec['const_axes']}")
    print(f"  models: {sec['models']}")
    print(f"  workloads: {len(sec['workloads'])}")
    
    # 检查第一个 workload 的完整参数
    wl = sec['workloads'][0]
    merged = {**sec['const_axes'], **wl['var_axes']}
    print(f"  Example merged params: {merged}")
```

### 4. 对比单个 workload 的性能

```python
import json

with open('results/your_op_name/your_op_name_compare_nvidia.json') as f:
    data = json.load(f)

# 找到特定 workload
target_name = "deepseek_v3.2_decode_num_tokens1"
comp = next(c for c in data['comparisons'] if c['workload'] == target_name)

print(f"Workload: {target_name}")
print(f"Baseline: {comp['baseline']['mean_ms']:.4f} ms ({comp['baseline']['impl_source']})")
print(f"FlagOS:   {comp['flagos']['mean_ms']:.4f} ms ({comp['flagos']['impl_source']})")
print(f"Speedup:  {comp['speedup']:.3f}x ({comp['verdict']})")
```

---

## 算子对比进展

| 算子 | case 生成 | flagos 跑完 | vllm 跑完 | compare | 结论 |
|------|:---------:|:-----------:|:---------:|:-------:|------|
| mhc_pre | ✅ | ✅ | ✅ | ✅ | flagos 慢 (0.62x geo-mean)，小N差2x，大N持平 |
| mhc_post | ✅ | ✅ | ✅ | ✅ | flagos 慢 (0.60x geo-mean)，小N差2x，大N差1.4x |
| swiglu | ✅ | ✅ | ✅ | ✅ | flagos 慢 (0.55x geo-mean)，decode 0.50x，prefill 0.61x，仅1个持平 |
| moe_sum | ✅ | ✅ | ✅ | ✅ | flagos 慢 (0.61x geo-mean)，decode 0.41x，prefill 0.98x（14个更快，59个更慢，7个持平） |
| fused_moe | ✅ | ✅ | ✅ | ✅ | **flagos 快** (1.14x geo-mean)，decode 0.93x，mixed 1.56x，prefill 1.23x（34快/17慢/7平）。kimi_k3 OOM跳过 |
| group_gemm | ✅ | ✅ | ✅ | ✅ | **flagos 快** (1.45x geo-mean)，59快/7慢/1平。Triton vs CUTLASS，除 qwen3.6_35b_a3b(小矩阵) 外全面领先 |
| grouped_topk | ✅ | ✅ | ✅ | ✅ | **vllm 快** (0.28x geo-mean)，0快/50慢/0平。Triton vs CUDA，vllm 快 ~3.5x。routing 算子计算量小，CUDA launch 开销优势大 |
| fused_deepseek_v4_qnorm_rope_kv_rope_quant_insert | ✅ | ✅ | ✅ | ✅ | **vllm 快** (0.66x geo-mean)，0快/58慢/12平。decode 0.50x，prefill 大batch 收敛至~1.0x。Triton 单 kernel vs CUDA 单 kernel，小 batch 固定开销差距明显 |
| flash_mla_with_kvcache | ✅ | ✅ | ✅ | ✅ | **vllm 快** (0.56x geo-mean)，0快/49慢/0平。小batch(b1-b8) 0.22-0.37x，大batch(b64-b128) 0.80-0.87x。flagos 有 ~0.25ms 固定底噪，vLLM 小 batch 能到 0.05ms |
