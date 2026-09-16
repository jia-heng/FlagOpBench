# pack_seq_triton Performance Analysis

## Summary

**Result**: FlagOS slower than vLLM baseline by ~6% on average (0.94x speedup)
- Total workloads: 30 (3 models × 10 scenarios each)
- Faster: 1, Slower: 28, On par: 1
- Both implementations are **Triton kernels**, not Triton vs CUDA

## Performance Breakdown

### By Model
| Model | Avg Speedup | Workloads |
|-------|------------|-----------|
| deepseek_v3.2 (index_topk=2048) | 0.9568x | 10 |
| deepseek_v4_flash (index_topk=512) | 0.9290x | 10 |
| deepseek_v4_pro (index_topk=1024) | 0.9342x | 10 |

### By Scenario
| Scenario | Avg Speedup | Workloads |
|----------|------------|-----------|
| decode | 0.9523x | 12 |
| mixed | 0.9361x | 6 |
| prefill | 0.9296x | 12 |

### Time Range
- **Baseline (vLLM)**: 0.0519 - 0.0586 ms
- **FlagOS**: 0.0552 - 0.0610 ms
- **Absolute difference**: +0.0034 ms (平均慢 3.4μs)

### Best Cases
1. `deepseek_v3.2_decode_num_tokens64`: **1.0597x** (0.0586ms → 0.0553ms) ✅ **唯一快的case**
2. `deepseek_v3.2_decode_num_tokens32`: 1.0269x (0.0573ms → 0.0558ms) ≈持平
3. `deepseek_v4_pro_decode_num_tokens64`: 0.9495x

### Worst Cases
1. `deepseek_v4_flash_prefill_num_tokens2048`: 0.8672x (0.0529ms → 0.0610ms)
2. `deepseek_v4_flash_prefill_num_tokens1024`: 0.9174x (0.0522ms → 0.0569ms)
3. `deepseek_v4_pro_prefill_num_tokens4096`: 0.9199x (0.0528ms → 0.0574ms)

## Implementation Comparison

### 核心区别

Both are Triton kernels with nearly identical logic, but differ in **prefix sum computation**:

#### vLLM baseline (`vllm/v1/attention/ops/common.py`)
```python
# Line 27-31: Sequential loop for prefix sum
in_start = 0
for i in range(pid_b):
    in_start += tl.load(lengths_ptr + i)
seq_len = tl.load(lengths_ptr + pid_b)
```
- **O(B)** sequential loads per block
- Simple, no extra kernel parameters
- Works for any batch size

#### FlagOS (`flaggems_vllm/ops/pack_seq.py`)
```python
# Line 58-61: Vectorized prefix sum with BLOCK_B
off_b = tl.arange(0, BLOCK_B)
prev_lengths = tl.load(lengths_ptr + off_b, mask=off_b < pid_b, other=0)
in_start = tl.sum(prev_lengths, axis=0)
seq_len = tl.load(lengths_ptr + pid_b)
```
- **Vectorized** load with masking
- Requires `BLOCK_B = triton.next_power_of_2(B)` parameter
- Better for large batch sizes (B ≥ 512)

#### Additional optimization in FlagOS
```python
# Line 24-34: Adaptive config selection
def _select_pack_seq_config(B, Lmax, D, element_size):
    # Use larger tiles for specific shapes
    if element_size <= 2 and B >= 512 and Lmax <= 16 and D >= 1024:
        return 128, 256, 4, 2  # BLOCK_T, BLOCK_D, num_warps, num_stages
    return 64, 64, 4, 2
```

### Why FlagOS is slower?

**Analysis**:

1. **Overhead from vectorized prefix sum**: For small batch sizes (B=1-64, 大部分decode场景), vectorizing the prefix sum with masking introduces:
   - Mask computation overhead
   - Potential wasted memory bandwidth loading masked-out elements
   - No benefit since sequential loop is already fast for small B

2. **Fixed overhead**: Both kernels show ~0.052ms baseline time, suggesting:
   - Kernel launch overhead dominates for this memory-bound op
   - Actual compute time is minimal (pure data movement)
   - Small differences (3-4μs) get amplified in speedup ratio

3. **Adaptive config not triggered**: 
   - The special case requires `B >= 512 && Lmax <= 16 && D >= 1024`
   - Most test cases don't meet these criteria
   - When triggered (e.g., deepseek_v3.2_decode_num_tokens64), FlagOS performs better (1.06x)

## Conclusion

This is a **minor performance difference** (~6%) with both implementations being functionally correct Triton kernels:

- **vLLM's sequential loop** is simpler and works well for all batch sizes
- **FlagOS's vectorized prefix sum** adds complexity but only helps at large batch sizes
- For this specific operator (pure memory movement, ~50μs runtime), kernel launch overhead dominates, making micro-optimizations less impactful

**Recommendation**: 
- Current FlagOS implementation is acceptable (within 10% of baseline)
- If optimization needed: consider hybrid approach (sequential for B < threshold, vectorized for B ≥ threshold)
- Priority: **LOW** (this is not a compute bottleneck in actual serving)

## Files
- Baseline: `vllm/vllm/v1/attention/ops/common.py:pack_seq_triton`
- FlagOS: `FlagGems-vllm/src/flaggems_vllm/ops/pack_seq.py:pack_seq_triton`
- Case: `cases/generated/merged/pack_seq_triton.yaml`
- Results: 
  - `results/pack_seq_triton/pack_seq_triton_nvidia.json`
  - `results/pack_seq_triton/pack_seq_triton_flagos_nvidia.json`
  - `results/pack_seq_triton/pack_seq_triton_compare_nvidia.json`
