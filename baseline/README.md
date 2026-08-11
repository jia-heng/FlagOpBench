# FlagOpBench - 性能基线测试平台

在多硬件平台上（NVIDIA、昇腾、沐曦等），对关键算子建立统一性能基线。

## 快速开始

### 环境要求

- Python 3.8+
- PyTorch 2.0+（需对应平台的 CUDA 版本）
- GPU 设备

### 安装依赖

```bash
pip install -r requirements.txt
```

### 运行测试

```bash
# 列出已注册算子
python run.py list

# 单算子测试
python run.py run --backend nvidia --case cases/basic/mm.yaml

# 某一类算子
python run.py run --backend nvidia --case-dir cases/basic/

# 带 Roofline 分析（需指定平台）
python run.py run --backend nvidia --case-dir cases/basic/ --platform nvidia_h20

# 全量测试并保存结果
python run.py run --backend nvidia --case-dir cases/ --output results/nvidia_h20.json
```

### 跨平台对比

```bash
python run.py compare --results results/nvidia_h20.json results/ascend_910b.json
```

### 回归检测

```bash
python run.py regression \
  --baseline results/nvidia_h20/torch2.4.0_20260801.json \
  --current results/nvidia_h20/torch2.5.0_20260815.json \
  --threshold 5
```

## 项目结构

```
baseline/
├── cases/                    # 测试用例 (YAML)
│   ├── basic/                # 基础算子 case
│   └── model/                # 大模型算子 case
├── backends/                 # 后端适配层
│   ├── base.py               # Backend 抽象基类
│   └── nvidia.py             # NVIDIA 后端
├── framework/                # 公共框架
│   ├── timer.py              # 双轨计时器
│   ├── validator.py          # 精度校验
│   ├── reporter.py           # 结果输出
│   ├── runner.py             # 执行引擎
│   ├── roofline.py           # Roofline 分析
│   └── env_collector.py      # 环境采集
├── operators/                # 算子实现
│   ├── registry.py           # 算子注册表
│   ├── basic/                # 基础算子
│   └── model/                # 大模型算子
├── hardware_specs.yaml       # 硬件规格表
├── run.py                    # CLI 入口
└── scripts/                  # 辅助脚本
```

## 添加新算子

1. 在 `operators/basic/` 或 `operators/model/` 下创建 Python 文件
2. 继承 `BaseOperator`，使用 `@register_operator("name")` 注册
3. 实现 `forward()`, `compute_flops()`, `compute_bytes()`, `prepare_inputs()` 方法
4. 在 `cases/` 下创建对应的 YAML 测试用例

```python
from baseline.operators.registry import BaseOperator, register_operator

@register_operator("my_op")
class MyOperator(BaseOperator):
    @property
    def name(self):
        return "my_op"

    def forward(self, x, **kwargs):
        return x * 2

    def compute_flops(self, M, N, **kwargs):
        return M * N

    def compute_bytes(self, M, N, dtype="bf16", **kwargs):
        return 2 * M * N * self.dtype_bytes(dtype)

    def prepare_inputs(self, M, N, dtype="bf16", **kwargs):
        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(M, N, device=self.device, dtype=torch_dtype)
        return {"x": x}
```

## 添加新后端

1. 在 `backends/` 下创建 Python 文件
2. 继承 `Backend` 基类
3. 实现所有抽象方法（get_device, synchronize, create_timer, get_operator）
4. 在 `run.py` 的 `create_backend()` 中注册

## 性能测试建议

1. **锁定 GPU 频率**：`sudo bash scripts/lock_gpu_freq.sh lock`
2. **确认无其他进程占用**：`nvidia-smi`
3. **多次运行取稳定值**：使用 median 而非 mean 作为主指标
4. **注意 thermal throttling**：长时间测试注意温度

## 输出格式

JSON 格式，包含：
- 平台和环境信息
- 每个 scenario 的精度校验结果
- device time 和 wall clock time 统计
- Roofline 效率分析（可选）
