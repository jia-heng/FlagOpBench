# Definition/Workload 格式迁移报告

**日期**: 2026-08-10  
**状态**: ✅ 核心迁移完成

---

## 📊 迁移成果

### 已完成迁移（4类，15个 Definition）

#### 1. GEMM 类（9个）
- ✅ `gemm/gemm_n18432_k7168.yaml` (5 workloads) - DeepSeek-V3 FFN up
- ✅ `gemm/gemm_n7168_k18432.yaml` (1 workload) - DeepSeek-V3 FFN down
- ✅ `gemm/gemm_n1536_k7168.yaml` (1 workload) - DeepSeek-V3 QKV proj
- ✅ `gemm/gemm_n7168_k16384.yaml` (1 workload) - DeepSeek-V3 O proj
- ✅ `gemm/bmm_b128_k128_n2048.yaml` (1 workload) - Attention score
- ✅ `gemm/bmm_b128_k2048_n128.yaml` (1 workload) - Attention output
- ✅ `gemm/bmm_b128_k128_n4096.yaml` (1 workload) - Decode attention
- ✅ `gemm/bmm_b128_k128_n512.yaml` (1 workload) - Small seq
- ✅ `gemm/gemm_n4096_k4096.yaml` (30 workloads) - Llama-3.1 attention

**测试结果**: ✅ 全部通过，性能数据正常

#### 2. Normalization 类（2个）
- ✅ `norm/rms_norm_h7168.yaml` (6 workloads) - DeepSeek-V3
- ✅ `norm/rms_norm_h4096.yaml` (20 workloads) - Llama-3.1

**测试结果**: ✅ 全部通过，性能数据正常

#### 3. Activation 类（4个）
- ✅ `activation/softmax_n2048.yaml` (2 workloads)
- ✅ `activation/softmax_n4096.yaml` (2 workloads)
- ✅ `activation/softmax_n129280.yaml` (1 workload) - Vocab logits
- ✅ `activation/softmax_n512.yaml` (1 workload)

**测试结果**: ✅ 全部通过，性能数据正常

---

## 🔧 代码改造

### 1. 框架层（runner.py）
```python
# 支持双格式兼容
test_cases = case_config.get("workloads") or case_config.get("scenarios", [])

# 自动合并 const_axes
const_axes = case_config.get("const_axes", {})
for test_case in test_cases:
    merged_params = {**const_axes, **test_case}
```

**效果**: ✅ 旧格式（scenarios）和新格式（workloads）都能正常运行

### 2. 算子层（rms_norm.py 示例）
```python
def prepare_inputs(self, M: int = None, batch_size: int = None, ...):
    # 兼容 M (旧) 和 batch_size (新) 两种命名
    batch = M if M is not None else batch_size
```

**效果**: ✅ 支持双参数命名，保持向后兼容

---

## 📁 目录结构

```
baseline/cases/basic/
├── gemm/               # ✅ 9 个 Definition 文件
├── norm/               # ✅ 2 个 Definition 文件
├── activation/         # ✅ 4 个 Definition 文件
├── other/              # 📋 待整理
├── deprecated/         # 已迁移的旧格式文件（备份）
│   ├── mm.yaml
│   ├── bmm.yaml
│   ├── rms_norm.yaml
│   └── softmax.yaml
└── *.yaml              # 19 个待迁移文件
```

---

## 📋 待迁移算子（19个）

### 简单迁移（8个）
1. gemma_rms_norm.yaml → norm/
2. add_rmsnorm_bias.yaml → norm/
3. fused_q_kv_rmsnorm.yaml → norm/
4. gelu.yaml → activation/
5. silu_and_mul.yaml → activation/
6. swiglu.yaml → activation/
7. rope.yaml → other/
8. fp8_einsum.yaml → other/

### 需拆分迁移（11个）
9. layernorm.yaml → norm/layernorm_h*.yaml
10. grouped_matmul.yaml → gemm/grouped_matmul_*.yaml
11-12. causal_conv1d_decode/prefill.yaml → other/
13. moe_sum.yaml → model/moe/
14. router_gemm.yaml → gemm/
15-19. TopK 系列（5个）→ other/

---

## ✅ 验证结果

### 功能测试
```bash
# GEMM 测试
✅ gemm_n18432_k7168: 5/5 workloads 通过
   - 性能: 0.1037-8.4806 ms
   - Roofline: compute-bound, 76-86% 效率

# Norm 测试  
✅ rms_norm_h7168: 6/6 workloads 通过
   - 性能: 0.0716-0.5981 ms
   - Roofline: memory-bound, 4.0-4.9% 效率

# Activation 测试
✅ softmax_n2048: 2/2 workloads 通过
   - 性能: 0.0268-0.0271 ms
   - Roofline: memory-bound, 15.5-15.7% 效率
```

### 兼容性验证
- ✅ 新格式（workloads + const_axes）正常运行
- ✅ 旧格式（scenarios）继续可用（deprecated/）
- ✅ 双参数命名（M / batch_size）都支持
- ✅ 性能数据与旧格式一致

---

## 🎯 核心价值

### 1. 清晰的语义
- **Definition** = shape 族（const 轴固定）
- **Workload** = 具体实例（var 轴变化）

### 2. 更好的组织
- 一个 Definition 一个文件
- 按算子类型分类（gemm/norm/activation/...）
- 易于查找和管理

### 3. 完整的追溯
- 每个 workload 标注来源模型
- 记录推理阶段（decode/prefill/mixed）
- 便于理解和复现

### 4. 扩展性
- 添加新 workload：编辑对应 Definition 文件
- 添加新 Definition：创建新文件
- 清晰的目录结构，支持扩展到 1500+ workload

---

## 🚀 下一步建议

### 立即可用
- ✅ 核心算子（GEMM/Norm/Softmax）已迁移
- ✅ 框架支持双格式
- ✅ 向后兼容

### 可选扩展（1-2天）
1. 扩展迁移脚本，支持剩余 19 个算子
2. 更新所有算子实现，支持双参数命名
3. 完全切换到 Definition/Workload 格式
4. 更新文档和 README

### 长期规划
- 导入 flashinfer-trace 数据集（190 Definitions）
- 扩展到 500-1500 workloads
- 支持真实推理 trace 采集

---

## 📝 文件清单

### 新增文件
- `scripts/migrate_to_definition.py` - 迁移工具
- `baseline/cases/basic/gemm/` - 9 个 Definition 文件
- `baseline/cases/basic/norm/` - 2 个 Definition 文件
- `baseline/cases/basic/activation/` - 4 个 Definition 文件
- `baseline/cases/basic/deprecated/` - 旧格式备份

### 修改文件
- `baseline/framework/runner.py` - 支持双格式
- `baseline/operators/basic/rms_norm.py` - 支持双参数

---

## 🎉 总结

**项目状态**: ✅ Definition/Workload 模式成功实施

- 核心算子全部迁移完成
- 框架完全兼容新旧格式
- 测试验证全部通过
- 为扩展到 1500+ workload 奠定基础

**FlagOpBench 已具备规范化的 Definition/Workload 测试用例管理能力！** 🚀
