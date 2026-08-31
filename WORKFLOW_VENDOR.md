# 国产平台适配全流程指南

面向 **在已有算子与已有 case 的前提下，把 FlagOpBench 适配到新硬件平台** 的流程。

与 `WORKFLOW.md` 的分工：

| | `WORKFLOW.md` | 本文件 |
|---|---|---|
| 面向 | 新增**算子** | 新增**平台** |
| 平台 | NVIDIA | 沐曦 / 摩尔 / 昇腾 / 海光 / 昆仑 / 天数 / 燧原 |
| 算子定义 | 需要写 | **复用，不动** |
| operator_registry.yaml | 需要注册 | **复用，不动** |
| 参数映射规则 | 需要加 | **复用，不动** |
| case 生成 | 需要跑 `gen_cases.py` | **跳过**，直接用 NV 上生成好的 `cases/generated/merged/*.yaml` |
| Provider | 加 `_load_xxx` 到 nvidia_provider | **新建/重写厂商 provider** |
| Timer | 复用 CudaTimer | **可能需要实现** |

`cases/` 目录整个是 NV 侧产物，各平台共用同一份，保证跨平台可比。**适配平台时不要重新生成 case**，否则会引入不可比因素。

---

## 流程概览

```
Step 1  环境确认        → 确定设备接口、厂商 vLLM 是否可用
Step 2  实现 Provider   → 核心，决定算子能否加载
Step 3  实现 Timer      → 决定能否计时
Step 4  补 Reporter     → 决定报告里设备信息是否真实
Step 5  注册 CLI 平台名 → 仅新平台需要
Step 6  验证加载        → 产出该平台算子覆盖清单
Step 7  跑 benchmark    → 用 NV 生成好的 case
Step 8  生成对比结果    → FlagOS vs 平台基线
```

---

## Step 1: 环境确认

适配前必须先确定三件事，后面每一步的写法都由它决定。

```bash
# 1) 设备通过哪个接口暴露
python -c "import torch; print('torch', torch.__version__)"
python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.device_count())"

# 2) 厂商专用 torch 插件是否存在（不同厂商换名字）
python -c "import torch_metax" 2>&1 | tail -1    # 沐曦
python -c "import torch_npu"   2>&1 | tail -1    # 昇腾
python -c "import torch_gcu"   2>&1 | tail -1    # 燧原

# 3) 厂商 vLLM 是否可用
python -c "import vllm; print('vllm', vllm.__version__)" 2>&1 | tail -1
python -c "from vllm import _custom_ops; print('custom_ops ok')" 2>&1 | tail -1
```

三类结果对应三种适配策略：

| 情况 | 判断依据 | Provider 策略 |
|---|---|---|
| **CUDA 兼容** | `torch.cuda.is_available()` 为 True | 继承 `NvidiaProvider`，复用整张 impl_map |
| **独立设备接口** | 需 `torch.npu` / `torch.gcu` 等 | 从 `BaseProvider` 起写，算子侧 `device="cuda"` 硬编码也要处理 |
| **无厂商 vLLM** | `import vllm` 失败 | 只能测 FlagOS 侧（`--impl flagos`），基线侧无对照 |

> **易错点**：不要用 `import torch_xxx` 作为 `is_available()` 的判据。多数厂商镜像并不提供该包，设备实际通过 `torch.cuda` 暴露。判据应为设备是否真的可用。

---

## Step 2: 实现 Provider

### 2.1 CUDA 兼容平台（推荐路径）

沐曦、摩尔、海光、天数的 vLLM 都是上游 vLLM 的 fork，模块路径与 NV 一致（`vllm._custom_ops`、`vllm.model_executor.layers.*`）。**直接继承 `NvidiaProvider`**，不必重写 28 个 `_load_xxx`。

```python
"""MetaX Provider"""
import torch

from .nvidia_provider import NvidiaProvider
from .registry import register_provider


@register_provider("metax", platform="metax", is_default=True)
class MetaxProvider(NvidiaProvider):

    @property
    def name(self) -> str:
        return "metax"

    @property
    def platform(self) -> str:
        return "metax"

    def get_device(self) -> torch.device:
        return torch.device("cuda:0")

    def synchronize(self) -> None:
        torch.cuda.synchronize()

    def is_available(self) -> bool:
        return torch.cuda.is_available()

    def setup(self):
        # 平台特有的前置 import 放这里，然后调父类
        super().setup()

    def get_impl(self, op_name, operator):
        impl_fn, impl_info = super().get_impl(op_name, operator)
        if impl_fn is not None:
            impl_info = {**impl_info, "platform": "metax"}
        return impl_fn, impl_info
```

这样做的关键收益：`NvidiaProvider.setup()` 中每个 vLLM 子模块都是独立 `try/except`，厂商 vLLM 缺少的模块会保持为 `None`，对应算子在 Runner 侧自动 `[SKIP]`。**跑一遍就等于自动产出该平台的算子覆盖缺口清单**，不需要人工比对。

### 2.2 模块名不同的情况

若厂商把 kernel 放在自己的模块里（如摩尔的 `vllm_musa._musa_custom_ops`），在 `setup()` 里做属性级合并，让父类的 `_load_xxx` 无感知：

```python
class _MergedOps:
    """按属性合并两个 ops 模块，主模块缺失时回退到备用模块"""

    def __init__(self, primary, fallback):
        self._primary = primary
        self._fallback = fallback

    def __getattr__(self, item):
        if self._primary is not None and hasattr(self._primary, item):
            return getattr(self._primary, item)
        if self._fallback is not None and hasattr(self._fallback, item):
            return getattr(self._fallback, item)
        raise AttributeError(item)


def setup(self):
    super().setup()
    try:
        from vllm_musa import _musa_custom_ops
        self._vllm_ops = _MergedOps(self._vllm_ops, _musa_custom_ops)
    except ImportError:
        pass
```

### 2.3 `torch.ops._C` 未注册

`NvidiaProvider` 中 `_torch_ops_registered` 在 `import vllm` 成功后即置 True，但部分厂商镜像的 `vllm` 缺少 `vllm._C` 扩展，此时 `torch.ops._C.xxx` 会在调用时才失败。改为实检：

```python
def setup(self):
    super().setup()
    self._torch_ops_registered = hasattr(torch.ops, "_C")
    if not self._torch_ops_registered:
        print("  [WARN] torch.ops._C not registered, related operators will be skipped")
```

### 2.4 非 CUDA 兼容平台

若设备走独立接口（昇腾 `npu:0`），除 Provider 外还需注意：**27 个 operator 全部硬编码 `device="cuda"`**，且 `runner.py` 未将 `provider.get_device()` 传给 `prepare_inputs`。这类平台需要额外的 device 注入方案，工作量远大于 CUDA 兼容平台。

---

## Step 3: 实现 Timer

`framework/timer.py` 中厂商 Timer 多为预留桩，`measure()` 直接 `raise NotImplementedError`。**Provider 写好但 Timer 没实现，会在跑到计时时才报错**，且报错信息与 provider 无关，容易误判。

CUDA 兼容平台直接继承 `CudaTimer`：

```python
class MetaxTimer(CudaTimer):
    """沐曦计时器

    MACA 复用 torch.cuda 接口，torch.cuda.Event(enable_timing=True) 与
    torch.cuda.synchronize() 在沐曦平台可用，因此计时逻辑与 CudaTimer 一致。
    """
    pass
```

若平台名不在 `create_timer()` 的 `timer_map` 中，需补一行映射：

```python
timer_map = {
    "nvidia": CudaTimer,
    "ascend": AscendTimer,
    "metax": MetaxTimer,
    "hygon": CudaTimer,      # 新增平台
}
```

---

## Step 4: 补 Reporter 设备信息

`framework/reporter.py` 的 `_collect_env()` 对非 NV 平台默认输出 `"<platform> device (info pending)"`，报告里看不到真实设备。补上采集分支：

```python
elif self.platform == "metax":
    if torch.cuda.is_available():
        env["device_name"] = torch.cuda.get_device_name(0)
        env["device_count"] = torch.cuda.device_count()
        props = torch.cuda.get_device_properties(0)
        env["device_memory_gb"] = round(props.total_memory / (1024**3), 1)
    else:
        env["device_name"] = "metax device (torch.cuda unavailable)"
```

同时建议在 `hardware_specs.yaml` 填该平台峰值算力与带宽，否则 roofline 相关指标为空。

---

## Step 5: 注册 CLI 平台名（仅新平台）

`run.py` 的 `--platform` 是硬编码 `choices`，未列入的平台会被 argparse 直接拒绝，连 provider 都进不去：

```python
parser.add_argument(
    "--platform",
    choices=["nvidia", "ascend", "metax", "mthreads", "iluvatar", "hygon"],
)
```

同时在 `providers/__init__.py` 加一行 import 触发注册：

```python
from . import hygon_provider
```

---

## Step 6: 验证加载（产出算子覆盖清单）

跑 benchmark 前先验证 provider 能注册、能加载。这一步的输出**就是该平台的算子覆盖缺口清单**，可直接作为适配进度依据。

```bash
python -c "
import operators
from framework.registry import import_all_operators, get_operator, list_operators
from providers.registry import get_provider
import_all_operators()

p = get_provider('metax', None)
print('provider:', p.name, '| platform:', p.platform)

ok, fail = [], []
for name in sorted(list_operators()):
    impl, info = p.get_impl(name, get_operator(name)())
    (ok if impl else fail).append((name, info))

print(f'\n可用 {len(ok)} / 缺失 {len(fail)}')
for n, i in ok:   print('  OK  ', n, '->', i.get('source'))
for n, i in fail: print('  MISS', n, '->', i.get('error'))
"
```

排查对照：

| 现象 | 原因 | 处理 |
|---|---|---|
| `invalid choice: 'xxx'` | 平台名未注册进 CLI | Step 5 |
| `Provider 'xxx' 不可用` | `is_available()` 返回 False | Step 2，检查判据是否用了不存在的 `torch_xxx` |
| 全部 MISS，error 为 `not implemented yet` | provider 仍是预留桩 | Step 2 |
| `list_operators()` 数量偏少 | `import_all_operators()` 对 ImportError 静默 pass | 单独 `import operators.xxx.operator` 看真实报错 |
| 加载正常但跑时 `NotImplementedError` | Timer 未实现 | Step 3 |

---

## Step 7: 跑 benchmark

**直接用 NV 上生成好的 case，不要重新生成。**

```bash
# 单算子，平台基线
python run.py --platform metax --case cases/generated/merged/swiglu.yaml

# 单算子，FlagOS 实现
python run.py --platform metax --impl flagos --case cases/generated/merged/swiglu.yaml

# 对比模式（只打印表格，不落 JSON）
python run.py --platform metax --mode compare --case cases/generated/merged/swiglu.yaml

# 全部算子
python run.py --platform metax --case-dir cases/generated/merged/
```

输出：`results/{operator}/{operator}_{provider}.json`，FlagOS 侧为 `{operator}_flagos_{platform}.json`。

---

## Step 8: 生成对比结果

```bash
python scripts/gen_compare_result.py \
    --operator swiglu \
    --platform metax
```

---

## 平台适配 Checklist

- [ ] Step 1 环境确认，判定 CUDA 兼容 / 独立接口 / 无厂商 vLLM
- [ ] Step 2 Provider 实现，`is_available()` 判据正确
- [ ] Step 3 Timer 实现（不能停留在 `NotImplementedError`）
- [ ] Step 4 Reporter 设备信息 + `hardware_specs.yaml`
- [ ] Step 5 CLI 平台名 + `providers/__init__.py` 注册（新平台）
- [ ] Step 6 验证脚本跑通，记录 OK / MISS 清单
- [ ] Step 7 benchmark 出数（**未重新生成 case**）
- [ ] Step 8 对比结果
- [ ] 记录该平台缺失算子及原因

---

## 注意事项

### 1. case 不重新生成

`cases/generated/merged/*.yaml` 是 NV 侧从 12 个模型配置生成的，各平台共用。重新生成会因模型配置或 profile 差异导致 workload 不一致，破坏跨平台可比性。

### 2. 框架不做正确性校验

`Runner` 只测时间，`--mode compare` 也只比性能，全仓无 `allclose` 类校验。**性能数据出来不代表结果正确**，数值正确性需另行验证。

### 3. `library` 字段取值不统一

operator.py 中 `flaggems` 与 `flag_gems` 混用，`operator_registry.yaml` 中 `flagattention` 与 `flag_attention` 并存。`flagos_provider` 按 `library` 做 dispatch 时需注意。

### 4. 平台 provider 的 source 标注

继承 `NvidiaProvider` 后，`impl_info["source"]` 仍是 NV 的模块路径字符串。建议在 `get_impl` 中补 `platform` 字段，避免报告混淆实际来源。

### 5. 目录结构

FlagOpBench 无 `setup.py`，靠 `sys.path.insert` 源码运行，需与 FlagGems / FlagGems-vllm / FlagAttention / vllm 平级：

```
Flagos/
├── FlagAttention      → flag_attn
├── FlagGems           → flag_gems
├── FlagGems-vllm      → flaggems_vllm
├── FlagOpBench
└── vllm               → 厂商版
```
