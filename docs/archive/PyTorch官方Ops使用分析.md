# PyTorch 官方 Ops 使用情况分析

## 当前算子实现分析

### ✅ 已使用官方 Ops 的算子（12 个）

| 算子 | 官方 Op | 文件 | 状态 |
|------|---------|------|------|
| mm | `torch.mm()` | `mm.py` | ✅ 完全官方 |
| bmm | `torch.bmm()` | `bmm.py` | ✅ 完全官方 |
| layernorm | `torch.nn.functional.layer_norm()` | `layernorm.py` | ✅ 完全官方 |
| softmax | `torch.nn.functional.softmax()` | `softmax.py` | ✅ 完全官方 |
| gelu | `torch.nn.functional.gelu()` | `gelu.py` | ✅ 完全官方 |
| silu_and_mul | `torch.nn.functional.silu()` | `silu_and_mul.py` | ✅ 完全官方 |
| swiglu | `torch.nn.functional.silu()` | `swiglu.py` | ✅ 完全官方 |
| fp8_einsum | `torch.einsum()` | `fp8_einsum.py` | ✅ 完全官方 |
| causal_conv1d | `torch.nn.functional.conv1d()` | `causal_conv1d.py` | ✅ 完全官方 |
| topk | `torch.topk()` | `topk.py` | ✅ 完全官方 |
| topk_softplus_sqrt | `torch.topk()` + `F.softplus()` | `topk_softplus_sqrt.py` | ✅ 完全官方 |
| rms_norm | `torch.ops._C.rms_norm` (vLLM) | `rms_norm.py` | ✅ 官方 CUDA kernel |
| add_rmsnorm_bias | `torch.ops._C.fused_add_rms_norm` (vLLM) | `add_rmsnorm_bias.py` | ✅ 官方 CUDA kernel |

### ⚠️ 使用手动实现的算子（7 个）

| 算子 | 当前实现 | 应改用的官方 Op | 优先级 |
|------|---------|----------------|--------|
| **gemma_rms_norm** | 手动 `x.pow(2).mean()` | `torch.nn.functional.rms_norm()` 或 vLLM | 🔴 高 |
| **fused_q_kv_rmsnorm** | 手动 RMSNorm | `torch.nn.functional.rms_norm()` | 🔴 高 |
| **rope** | 手动旋转 | `torch.ops._C.rotary_embedding` (vLLM) | 🟡 中 |
| **router_gemm** | 手动 `torch.mm()` + cast | `torch.mm()` 或 `torch.matmul()` | 🟢 低（已基本官方）|
| **grouped_matmul** | 手动循环 + matmul | `torch.nn.functional.scaled_dot_product_attention` 或分组 ops | 🟡 中 |
| **moe_sum** | 手动 broadcast + sum | `torch.einsum()` 或 `torch.bmm()` | 🟡 中 |

---

## 需要修改的算子详情

### 1. gemma_rms_norm.py（高优先级）

**当前实现**（手动）:
```python
def forward(self, x, weight, eps=1e-6):
    input_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * (weight.float() + 1.0)).to(input_dtype)
```

**应改为官方 Op**:
```python
# PyTorch 2.4+ 提供了 rms_norm
def forward(self, x, weight, eps=1e-6):
    # Gemma 特殊: weight + 1
    adjusted_weight = weight + 1.0
    return torch.nn.functional.rms_norm(x, [x.shape[-1]], adjusted_weight, eps)
```

或者使用 vLLM:
```python
def forward(self, x, weight, eps=1e-6):
    if HAS_VLLM_OPS:
        adjusted_weight = weight + 1.0
        out = torch.empty_like(x)
        vllm_ops.rms_norm(out, x, adjusted_weight, eps)
        return out
    else:
        # fallback
```

---

### 2. fused_q_kv_rmsnorm.py（高优先级）

**当前实现**（手动）:
```python
def _rmsnorm(self, x, weight, eps):
    input_dtype = x.dtype
    x = x.float()
    variance = x.pow(2).mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (x * weight.float()).to(input_dtype)
```

**应改为官方 Op**:
```python
def forward(self, q, kv, q_weight, kv_weight, eps=1e-6):
    # 使用 PyTorch 官方 rms_norm 或 vLLM
    if HAS_VLLM_OPS:
        q_out = torch.empty_like(q)
        kv_out = torch.empty_like(kv)
        vllm_ops.rms_norm(q_out, q, q_weight, eps)
        vllm_ops.rms_norm(kv_out, kv, kv_weight, eps)
        return torch.cat([q_out, kv_out], dim=-1)
    else:
        # PyTorch 2.4+
        q_normed = F.rms_norm(q, [q.shape[-1]], q_weight, eps)
        kv_normed = F.rms_norm(kv, [kv.shape[-1]], kv_weight, eps)
        return torch.cat([q_normed, kv_normed], dim=-1)
```

---

### 3. rope.py（中优先级）

**当前实现**（手动旋转）:
```python
def _rotate_half(self, x):
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)

def forward(self, q, cos, sin):
    q_rot = q * cos + self._rotate_half(q) * sin
    return q_rot
```

**应改为 vLLM CUDA kernel**:

需要调研 vLLM 的 `rotary_embedding` 接口，它需要额外参数。可能需要：
```python
# vLLM 接口需要: positions, head_size 等参数
vllm_ops.rotary_embedding(
    positions,  # (batch, seq_len)
    query,      # (batch, seq_len, num_heads, head_dim)
    key,        # (batch, seq_len, num_kv_heads, head_dim)
    head_size,
    cos_sin_cache,
    is_neox=True
)
```

如果接口不匹配，保持手动实现但添加注释说明。

---

### 4. moe_sum.py（中优先级）

**当前实现**（手动 broadcast）:
```python
def forward(self, expert_outputs, weights):
    # expert_outputs: (num_experts, num_tokens, hidden_size)
    # weights: (num_tokens, num_experts)
    w = weights.t().unsqueeze(-1)  # (num_experts, num_tokens, 1)
    return (expert_outputs * w).sum(dim=0)
```

**应改为 einsum**:
```python
def forward(self, expert_outputs, weights):
    # 更简洁且可能被优化
    return torch.einsum('enh,ne->nh', expert_outputs, weights)
```

或者使用 bmm:
```python
def forward(self, expert_outputs, weights):
    # weights: (num_tokens, num_experts) -> (1, num_tokens, num_experts)
    # expert_outputs: (num_experts, num_tokens, hidden) 
    # 需要转置为 (num_tokens, num_experts, hidden)
    expert_outputs_t = expert_outputs.permute(1, 0, 2)  # (tokens, experts, hidden)
    weights_expanded = weights.unsqueeze(-1)  # (tokens, experts, 1)
    return (expert_outputs_t * weights_expanded).sum(dim=1)  # (tokens, hidden)
```

---

### 5. grouped_matmul.py（中优先级）

**当前实现**（手动循环）:
```python
def forward(self, x, weights, expert_ids):
    num_tokens = x.shape[0]
    expert_size = weights.shape[2]
    output = torch.zeros(num_tokens, expert_size, ...)
    
    # 循环每个 expert（慢！）
    for expert_id in range(num_experts):
        mask = expert_ids == expert_id
        ...
```

**应改为批量操作**:
```python
def forward(self, x, weights, expert_ids):
    # 使用 torch.index_select 或 gather 避免循环
    # weights[expert_ids] 可以批量索引
    selected_weights = weights[expert_ids]  # (num_tokens, hidden_size, expert_size)
    # 使用 bmm 批量计算
    output = torch.bmm(
        x.unsqueeze(1),  # (num_tokens, 1, hidden_size)
        selected_weights  # (num_tokens, hidden_size, expert_size)
    ).squeeze(1)
    return output
```

---

### 6. router_gemm.py（低优先级）

**当前实现**（已接近官方）:
```python
def forward(self, x, router_weight):
    x_fp32 = x.float()
    w_fp32 = router_weight.t().float()
    logits = torch.mm(x_fp32, w_fp32)
    return logits
```

**可以简化为**:
```python
def forward(self, x, router_weight):
    # torch.matmul 自动处理 dtype 和转置
    return torch.matmul(x.float(), router_weight.t().float())
```

实际上已经够官方了，只是加了 dtype cast，这是合理的。

---

## PyTorch 2.4+ 新增的官方 Ops

### 1. RMSNorm (torch.nn.functional.rms_norm)

PyTorch 2.4+ 新增了官方 RMSNorm 实现:
```python
torch.nn.functional.rms_norm(
    input,
    normalized_shape,
    weight=None,
    eps=1e-5
)
```

**检查是否可用**:
```python
import torch
import torch.nn.functional as F

# 检查版本
print(f"PyTorch version: {torch.__version__}")

# 检查是否有 rms_norm
if hasattr(F, 'rms_norm'):
    print("✅ torch.nn.functional.rms_norm 可用")
else:
    print("❌ torch.nn.functional.rms_norm 不可用，需要升级或使用 vLLM")
```

如果当前 PyTorch 版本不支持，继续使用 vLLM 的 `torch.ops._C.rms_norm`。

---

## 修改优先级

### 🔴 高优先级（立即修改）

1. **gemma_rms_norm.py** - 改用 `F.rms_norm()` 或 vLLM
2. **fused_q_kv_rmsnorm.py** - 改用 `F.rms_norm()` 或 vLLM

### 🟡 中优先级（编译完成后）

3. **moe_sum.py** - 改用 `torch.einsum()`
4. **grouped_matmul.py** - 改用 `torch.bmm()` + indexing
5. **rope.py** - 适配 vLLM `rotary_embedding` 接口

### 🟢 低优先级（可选优化）

6. **router_gemm.py** - 已基本官方，可保持现状

---

## 行动计划

### 第 1 步: 检查 PyTorch 版本是否支持 rms_norm

```bash
python -c "import torch; import torch.nn.functional as F; print(f'PyTorch {torch.__version__}'); print(f'rms_norm: {hasattr(F, \"rms_norm\")}')"
```

### 第 2 步: 修改 gemma_rms_norm.py

- 如果 PyTorch 支持 `F.rms_norm()`，使用它
- 否则改用 vLLM `torch.ops._C.rms_norm`（编译后）

### 第 3 步: 修改 fused_q_kv_rmsnorm.py

- 同上

### 第 4 步: 修改 moe_sum.py

- 改用 `torch.einsum('enh,ne->nh', expert_outputs, weights)`

### 第 5 步: 修改 grouped_matmul.py

- 改用 `torch.bmm()` 避免循环

---

## 总结

**当前状态**:
- ✅ 12/19 个算子已使用官方 Ops
- ⚠️ 7/19 个算子有手动实现成分

**目标**:
- 全部改用 PyTorch 官方 Ops 或 vLLM CUDA kernel
- 避免手动实现的 `.pow(2).mean()` 等操作

**收益**:
- 更好的性能（官方 Ops 有融合优化）
- 更好的数值稳定性
- 符合用户要求的"官方实现"标准
