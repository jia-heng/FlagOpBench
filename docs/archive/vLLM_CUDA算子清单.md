# vLLM CUDA Kernel 实现的算子清单

## 说明

以下算子在 vLLM 中通过 CUDA/C++ kernel 实现，并注册为 `torch.ops._C.*`，这些是**官方 SDK 级别的实现**，具有参考意义。

## 已在 vLLM CUDA kernel 中实现的算子

### ✅ 已在我们框架中实现的（可对比）

| # | 算子名称 | vLLM CUDA Op | 我们的实现 | 对比价值 |
|---|---------|-------------|-----------|---------|
| 50 | rms_norm | `torch.ops._C.rms_norm` | ✅ 纯 PyTorch | 高 - 对比官方优化 |
| 21 | AddRmsNormBias | `torch.ops._C.fused_add_rms_norm` | ✅ 纯 PyTorch | 高 - 融合算子对比 |
| 51 | RoPE | `torch.ops._C.rotary_embedding` | ✅ 纯 PyTorch | 高 |
| 2 | silu_and_mul_with_clamp | `torch.ops._C.silu_and_mul_per_block_quant` | ✅ 纯 PyTorch | 中 - 量化版本 |

### 🔧 需要接入的高价值算子

#### MoE 系列

| # | 算子名称 | vLLM CUDA Op | 文件位置 |
|---|---------|-------------|---------|
| 14 | fused_moe | `torch.ops._C.cutlass_moe_mm` | `csrc/moe/` |
| 47 | MoeAlignBlockSize | `torch.ops._C.moe_align_block_size` | `csrc/moe/` |
| 4 | moe_sum | `torch.ops._C.moe_sum` | `csrc/moe/` |

#### TopK 系列

| # | 算子名称 | vLLM CUDA Op | 文件位置 |
|---|---------|-------------|---------|
| 8 | top_k_per_row_prefill | `torch.ops._C.top_k_per_row_prefill` | `csrc/` |
| 18 | top_k_per_row_decode | `torch.ops._C.top_k_per_row_decode` | `csrc/` |

#### KV Cache 系列

| # | 算子名称 | vLLM CUDA Op | 文件位置 |
|---|---------|-------------|---------|
| 28 | FlashKDA | `torch.ops._C.fused_kda_decode` | `csrc/flashkda_registration.cpp` |
| 24 | flash_mla_with_kvcache | `torch.ops._C.concat_and_cache_mla` | `csrc/` |

#### Norm + RoPE 融合

| # | 算子名称 | vLLM CUDA Op | 文件位置 |
|---|---------|-------------|---------|
| 15 | fused_q_kv_rmsnorm | `torch.ops._C.fused_qk_norm_rope` | `csrc/libtorch_stable/` |
| 48 | per_token_group_fp8_quant | `torch.ops._C.rms_norm_dynamic_per_token_quant` | `csrc/` |

#### 量化 GEMM

| # | 算子名称 | vLLM CUDA Op | 文件位置 |
|---|---------|-------------|---------|
| 38 | gemm w8a8 | `torch.ops._C.cutlass_scaled_mm` | `csrc/cutlass_extensions/` |
| 30 | fp8_einsum | `torch.ops._C.cutlass_scaled_mm` | `csrc/cutlass_extensions/` |
| 32-35 | fused_marlin_moe 系列 | `torch.ops._C.marlin_gemm` / `cutlass_moe_mm` | `csrc/quantization/` |

#### Mamba 系列

| # | 算子名称 | vLLM CUDA Op | 文件位置 |
|---|---------|-------------|---------|
| 22/23 | CausalConv1D | `torch.ops._C.causal_conv1d_*` | `csrc/` |
| 36 | GDN | `torch.ops._C.chunk_gated_delta_rule_cpu` | `csrc/cpu/` |

## 我们当前实现方式的问题

### 当前实现（纯 PyTorch）

```python
# baseline/operators/basic/rms_norm.py
def forward(self, x, weight, eps=1e-6):
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return x * weight
```

**问题**：
- ❌ 不是官方 SDK 实现
- ❌ 性能不具备参考价值（PyTorch 调度开销大）
- ❌ 无法反映硬件真实能力

### 应该的实现方式（调用 vLLM CUDA kernel）

```python
# baseline/operators/basic/rms_norm.py
import torch.ops._C as vllm_ops

def forward(self, x, weight, eps=1e-6):
    out = torch.empty_like(x)
    vllm_ops.rms_norm(out, x, weight, eps)  # 直接调用 CUDA kernel
    return out
```

**优势**：
- ✅ 官方优化的 CUDA kernel
- ✅ 性能具备参考价值
- ✅ 真实反映硬件能力

## 修改建议

### Phase 1: 替换已实现的算子为 vLLM CUDA 版本

将以下算子从纯 PyTorch 实现改为调用 vLLM CUDA kernel：

1. **rms_norm** → `torch.ops._C.rms_norm`
2. **add_rmsnorm_bias** → `torch.ops._C.fused_add_rms_norm`
3. **rope** → `torch.ops._C.rotary_embedding`
4. **top_k_per_row_prefill/decode** → `torch.ops._C.top_k_per_row_*`

### Phase 2: 接入核心 MoE 算子

5. **fused_moe** → `torch.ops._C.cutlass_moe_mm`
6. **moe_align_block_size** → `torch.ops._C.moe_align_block_size`

### Phase 3: 接入 Attention 相关

7. **flash_mla** → 需要通过 vLLM 的 MLAAttention 层
8. **FlashKDA** → `torch.ops._C.fused_kda_decode`

## 实现示例

### 修改后的 rms_norm.py

```python
"""RMS Norm 算子 - 使用 vLLM CUDA kernel"""

import torch
from baseline.operators.registry import BaseOperator, register_operator

# 导入 vLLM 编译好的 CUDA ops
try:
    import vllm._custom_ops as vllm_ops
    HAS_VLLM_OPS = True
except ImportError:
    HAS_VLLM_OPS = False
    print("Warning: vLLM ops not available, falling back to PyTorch")


@register_operator("rms_norm")
class RMSNormOperator(BaseOperator):
    """RMS Normalization - vLLM CUDA kernel 版本"""

    @property
    def name(self) -> str:
        return "rms_norm"

    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                eps: float = 1e-6, **kwargs) -> torch.Tensor:
        if HAS_VLLM_OPS:
            # 调用 vLLM 官方 CUDA kernel
            out = torch.empty_like(x)
            vllm_ops.rms_norm(out, x, weight, eps)
            return out
        else:
            # Fallback: 纯 PyTorch 实现
            input_dtype = x.dtype
            x = x.float()
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)
            return (x * weight.float()).to(input_dtype)
    
    # compute_flops, compute_bytes, prepare_inputs 保持不变
    ...
```

## 优先级

### P0 - 立即修改（已实现的算子）

1. rms_norm
2. fused_add_rms_norm (add_rmsnorm_bias)
3. rotary_embedding (rope)
4. top_k_per_row_prefill/decode

### P1 - 接入核心算子（未实现）

5. cutlass_moe_mm (fused_moe)
6. moe_align_block_size
7. fused_kda_decode (FlashKDA)

### P2 - 完善覆盖（量化/特殊场景）

8. cutlass_scaled_mm (fp8 gemm)
9. marlin_gemm (量化)
10. 其他融合算子

## 总结

**当前问题**：我们用纯 PyTorch 实现了 23 个算子，但这些**不是官方 SDK 实现**，性能数据参考价值有限。

**解决方案**：
1. **替换现有实现** - 将已实现的 4 个算子改为调用 vLLM CUDA kernel
2. **接入官方实现** - 新增约 20 个 vLLM 已有的 CUDA kernel 算子
3. **保持 fallback** - 对于没有 CUDA kernel 的，保留 PyTorch 实现作为 fallback

**预期收益**：
- ✅ 性能数据具备参考价值
- ✅ 反映官方 SDK 真实能力
- ✅ 覆盖率提升至 ~40 个算子（73%）
