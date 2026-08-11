# 如何使用 vLLM CUDA 算子

## 当前状态

**问题**：vLLM 的 CUDA kernel（`torch.ops._C.*`）需要编译后才能使用，当前环境未编译 vLLM。

**当前行为**：所有算子自动 fallback 到纯 PyTorch 实现。

```
Warning: vLLM CUDA ops not available for rms_norm, using PyTorch fallback
Warning: vLLM CUDA ops not available for fused_add_rms_norm, using PyTorch fallback
Warning: vLLM CUDA ops not available for rotary_embedding, using PyTorch fallback
Warning: vLLM CUDA ops not available for topk, using PyTorch fallback
```

---

## 解决方案：编译 vLLM

### 方案 1：从源码编译 vLLM（推荐）

```bash
cd /data/jianheng/works/FlagOpBench/vllm

# 安装依赖
pip install -e .

# 编译 CUDA extensions
python setup.py build_ext --inplace

# 验证
python -c "import torch.ops._C as ops; print(dir(ops))"
```

编译成功后会看到：
```python
['rms_norm', 'fused_add_rms_norm', 'rotary_embedding', 
 'top_k_per_row_prefill', 'top_k_per_row_decode', 
 'cutlass_moe_mm', 'fused_kda_decode', ...]
```

### 方案 2：使用预编译的 vLLM wheel

```bash
pip install vllm==0.6.1  # 或最新版本
```

### 方案 3：使用 Docker 镜像（最简单）

```bash
docker run --gpus all -it vllm/vllm-openai:latest bash
```

---

## 验证 CUDA Kernel 是否可用

运行验证脚本：

```bash
python3 << 'EOF'
import torch

# 检查 vLLM CUDA ops
try:
    import torch.ops._C as vllm_ops
    print("✓ vLLM CUDA ops available")
    print(f"  - rms_norm: {hasattr(vllm_ops, 'rms_norm')}")
    print(f"  - fused_add_rms_norm: {hasattr(vllm_ops, 'fused_add_rms_norm')}")
    print(f"  - rotary_embedding: {hasattr(vllm_ops, 'rotary_embedding')}")
    print(f"  - top_k_per_row_prefill: {hasattr(vllm_ops, 'top_k_per_row_prefill')}")
    print(f"  - cutlass_moe_mm: {hasattr(vllm_ops, 'cutlass_moe_mm')}")
except Exception as e:
    print(f"✗ vLLM CUDA ops NOT available: {e}")

# 测试 rms_norm
if hasattr(vllm_ops, 'rms_norm'):
    x = torch.randn(2048, 4096, dtype=torch.float16, device='cuda')
    w = torch.randn(4096, dtype=torch.float16, device='cuda')
    out = torch.empty_like(x)
    vllm_ops.rms_norm(out, x, w, 1e-6)
    print("✓ rms_norm CUDA kernel works!")
EOF
```

---

## 性能对比：PyTorch vs vLLM CUDA

### 当前实现（PyTorch fallback）

```python
# baseline/operators/basic/rms_norm.py
def forward(self, x, weight, eps=1e-6):
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight
```

**问题**：
- ❌ 多次 kernel launch（pow, mean, rsqrt, mul）
- ❌ 中间结果需要写回全局内存
- ❌ 性能约为 CUDA kernel 的 30-50%

### vLLM CUDA Kernel

```python
def forward(self, x, weight, eps=1e-6):
    out = torch.empty_like(x)
    vllm_ops.rms_norm(out, x, weight, eps)  # 单次 kernel launch
    return out
```

**优势**：
- ✅ 融合算子，单次 kernel launch
- ✅ 在线规约，无需中间变量
- ✅ 接近硬件理论峰值

### 预期性能提升

| 算子 | PyTorch 实现 | vLLM CUDA | 提升倍数 |
|------|-------------|-----------|---------|
| rms_norm | 0.33 ms | ~0.15 ms | ~2.2x |
| fused_add_rms_norm | 0.45 ms | ~0.18 ms | ~2.5x |
| top_k_per_row | 0.25 ms | ~0.08 ms | ~3x |

---

## 编译后的使用流程

### 1. 编译 vLLM

```bash
cd /data/jianheng/works/FlagOpBench/vllm
pip install -e .
```

### 2. 运行测试

```bash
cd /data/jianheng/works/FlagOpBench

# 单个算子测试
python baseline/run.py run --backend nvidia \
  --case baseline/cases/basic/rms_norm.yaml --platform nvidia_h20

# 全部算子测试
python baseline/run.py run --backend nvidia \
  --case-dir baseline/cases/basic/ --platform nvidia_h20
```

### 3. 验证使用的是 CUDA kernel

运行日志中**不应该**出现 Warning：

```
Running case: baseline/cases/basic/rms_norm.yaml
  [✓] rms_norm/hidden_7168_seq2048: 0.15 ms | memory-bound | eff=9.8%
```

如果仍然看到：
```
Warning: vLLM CUDA ops not available for rms_norm, using PyTorch fallback
```

说明编译失败或导入路径不对。

---

## 已适配的算子列表

以下算子已修改为优先使用 vLLM CUDA kernel：

| 算子 | vLLM CUDA Op | Fallback | 状态 |
|------|-------------|----------|------|
| rms_norm | `torch.ops._C.rms_norm` | ✅ PyTorch | 已适配 |
| add_rmsnorm_bias | `torch.ops._C.fused_add_rms_norm` | ✅ PyTorch | 已适配 |
| rope | `torch.ops._C.rotary_embedding` | ✅ PyTorch | 已适配（接口待调整）|
| top_k_per_row_prefill | `torch.ops._C.top_k_per_row_prefill` | ✅ PyTorch | 已适配 |
| top_k_per_row_decode | `torch.ops._C.top_k_per_row_decode` | ✅ PyTorch | 已适配 |

---

## 下一步：接入更多 vLLM 算子

编译完成后，可以快速接入以下算子：

### P0 - 核心算子

1. **fused_moe** → `torch.ops._C.cutlass_moe_mm`
2. **FlashKDA** → `torch.ops._C.fused_kda_decode`
3. **flash_mla** → vLLM MLAAttention 层

### P1 - 量化 GEMM

4. **fp8_gemm** → `torch.ops._C.cutlass_scaled_mm`
5. **marlin_moe** → `torch.ops._C.marlin_gemm`

### 实现示例

```python
# baseline/operators/model/fused_moe.py
import torch.ops._C as vllm_ops

@register_operator("fused_moe")
class FusedMoEOperator(BaseOperator):
    def forward(self, x, expert_weights, top_k_indices, ...):
        return vllm_ops.cutlass_moe_mm(x, expert_weights, ...)
```

---

## 总结

**当前状态**：✅ 代码已适配 vLLM CUDA kernel，但未编译所以使用 fallback

**下一步操作**：
1. 编译 vLLM：`cd vllm && pip install -e .`
2. 验证：`python baseline/run.py run --backend nvidia --case baseline/cases/basic/rms_norm.yaml`
3. 确认无 Warning 且性能提升 ~2x

**预期收益**：
- rms_norm 性能提升 2.2x
- 所有算子使用官方 CUDA kernel，性能数据具备参考价值
