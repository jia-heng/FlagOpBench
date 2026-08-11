#!/usr/bin/env python3
"""
智能去重过滤器 - 实现设计文档中的去重策略

包含两类过滤器:
1. UniqueAxesFilter - 基础算子去重（GEMM, BMM, LayerNorm 等）
2. SemanticFilter - 复杂算子去重（Attention, MoE 等）
"""

from typing import List, Dict, Callable, Set, Tuple, Any
from collections import defaultdict


class UniqueAxesFilter:
    """基础算子去重：每个唯一 axes 组合保留 k 个（通常 k=1）

    适用算子: GEMM, BMM, LayerNorm, Softmax, Add 等计算密集型算子

    逻辑:
        对于每个唯一的 (var_axes) 组合，只保留 k 个 workload
        例如 GEMM: (M, N, K) = (256, 4096, 4096) 只保留 1 个

    示例:
        filter = UniqueAxesFilter(var_axes=['M', 'N', 'K'], keep_first_k=1)
        filtered = filter.filter(workloads)
    """

    def __init__(self, var_axes: List[str], keep_first_k: int = 1):
        """
        Args:
            var_axes: 需要去重的轴名称列表
            keep_first_k: 每个唯一组合保留几个（通常为 1）
        """
        self.var_axes = var_axes
        self.keep_first_k = keep_first_k
        self.seen: Dict[Tuple, int] = defaultdict(int)

    def _get_key(self, workload: Dict) -> Tuple:
        """提取 workload 的去重 key"""
        # 只使用 var_axes，按字母序排序确保一致性
        values = []
        for axis in sorted(self.var_axes):
            if axis in workload:
                values.append((axis, workload[axis]))
        return tuple(values)

    def should_keep(self, workload: Dict) -> bool:
        """判断是否保留该 workload"""
        key = self._get_key(workload)
        count = self.seen[key]

        if count < self.keep_first_k:
            self.seen[key] += 1
            return True
        return False

    def filter(self, workloads: List[Dict]) -> List[Dict]:
        """过滤 workload 列表"""
        self.seen.clear()
        return [wl for wl in workloads if self.should_keep(wl)]

    def stats(self) -> Dict:
        """返回去重统计信息"""
        return {
            "unique_combinations": len(self.seen),
            "total_kept": sum(self.seen.values()),
        }


class SemanticFilter:
    """复杂算子去重：按语义派生指标去重

    适用算子: Attention, MoE 等结构复杂的算子

    核心思想:
        某些算子的性能主要受语义特征影响，而非原始轴的绝对值
        例如 Attention: 性能取决于 avg_kv_len，而非 batch_size 绝对值

        batch=8, avg_kv_len=73 和 batch=16, avg_kv_len=73 性能相近 → 去重
        batch=8, avg_kv_len=73 和 batch=8, avg_kv_len=512 性能差异大 → 都保留

    示例:
        def attention_key(wl):
            avg_kv_len = round(wl['num_kv_indices'] / wl['batch_size'])
            return (wl['num_qo_heads'], wl['num_kv_heads'], avg_kv_len)

        filter = SemanticFilter(semantic_fn=attention_key)
        filtered = filter.filter(workloads)
    """

    def __init__(self, semantic_fn: Callable[[Dict], Tuple], keep_first_k: int = 1):
        """
        Args:
            semantic_fn: 计算语义 key 的函数，输入 workload，返回 tuple
            keep_first_k: 每个唯一组合保留几个
        """
        self.semantic_fn = semantic_fn
        self.keep_first_k = keep_first_k
        self.seen: Dict[Tuple, int] = defaultdict(int)

    def should_keep(self, workload: Dict) -> bool:
        """判断是否保留该 workload"""
        try:
            key = self.semantic_fn(workload)
            count = self.seen[key]

            if count < self.keep_first_k:
                self.seen[key] += 1
                return True
            return False
        except (KeyError, ZeroDivisionError, TypeError) as e:
            # 如果无法计算语义 key，默认保留
            print(f"Warning: Failed to compute semantic key for {workload.get('name', 'unnamed')}: {e}")
            return True

    def filter(self, workloads: List[Dict]) -> List[Dict]:
        """过滤 workload 列表"""
        self.seen.clear()
        return [wl for wl in workloads if self.should_keep(wl)]

    def stats(self) -> Dict:
        """返回去重统计信息"""
        return {
            "unique_semantic_keys": len(self.seen),
            "total_kept": sum(self.seen.values()),
        }


# ============================================================================
# 预定义的语义函数 - 常用算子的语义特征提取
# ============================================================================

def attention_semantic_key(workload: Dict) -> Tuple:
    """Attention 算子的语义 key

    关键发现: 性能主要受平均序列长度影响

    返回: (num_qo_heads, num_kv_heads, head_dim, avg_kv_len)
    """
    batch_size = workload.get('batch_size', 1)
    num_kv_indices = workload.get('num_kv_indices', 0)

    # 计算平均 KV 序列长度
    avg_kv_len = round(num_kv_indices / batch_size) if batch_size > 0 else 0

    return (
        workload.get('num_qo_heads', 32),
        workload.get('num_kv_heads', 8),
        workload.get('head_dim', 128),
        avg_kv_len,
    )


def moe_semantic_key(workload: Dict) -> Tuple:
    """MoE 算子的语义 key

    关键发现: 性能主要受平均每个 expert 的 token 数影响

    返回: (num_experts, hidden_size, avg_tokens_per_expert)
    """
    num_experts = workload.get('num_experts', 8)
    num_tokens = workload.get('num_tokens', 0)

    # 假设 token 均匀分布到 experts
    avg_tokens_per_expert = round(num_tokens / num_experts) if num_experts > 0 else 0

    return (
        num_experts,
        workload.get('hidden_size', 4096),
        avg_tokens_per_expert,
    )


def sampling_semantic_key(workload: Dict) -> Tuple:
    """Sampling 算子的语义 key

    关键发现: 性能主要受 vocab_size 和 batch_size 影响

    返回: (vocab_size, batch_size)
    """
    return (
        workload.get('vocab_size', 32000),
        workload.get('batch_size', 1),
    )


# ============================================================================
# 工厂函数 - 根据算子类型自动创建过滤器
# ============================================================================

def create_filter(operator: str, **kwargs):
    """根据算子类型创建合适的过滤器

    Args:
        operator: 算子名称
        **kwargs: 传递给过滤器的额外参数

    Returns:
        Filter 实例

    示例:
        filter = create_filter('mm')
        filter = create_filter('paged_attention_decode')
    """
    # 基础算子 - 使用 UniqueAxesFilter
    if operator in ['mm', 'bmm']:
        return UniqueAxesFilter(var_axes=['M', 'N', 'K'], **kwargs)

    elif operator in ['layernorm', 'rms_norm', 'gemma_rms_norm']:
        return UniqueAxesFilter(var_axes=['batch_size', 'hidden_size'], **kwargs)

    elif operator in ['softmax', 'gelu', 'silu_and_mul']:
        return UniqueAxesFilter(var_axes=['batch_size', 'seq_len'], **kwargs)

    # Attention 算子 - 使用 SemanticFilter
    elif 'attention' in operator or 'decode' in operator or 'prefill' in operator:
        return SemanticFilter(semantic_fn=attention_semantic_key, **kwargs)

    # MoE 算子
    elif 'moe' in operator:
        return SemanticFilter(semantic_fn=moe_semantic_key, **kwargs)

    # Sampling 算子
    elif 'top' in operator or 'sampling' in operator:
        return SemanticFilter(semantic_fn=sampling_semantic_key, **kwargs)

    # 默认：不去重
    else:
        print(f"Warning: No filter defined for operator '{operator}', keeping all workloads")
        return NoOpFilter()


class NoOpFilter:
    """空操作过滤器 - 保留所有 workload"""

    def filter(self, workloads: List[Dict]) -> List[Dict]:
        return workloads

    def stats(self) -> Dict:
        return {"note": "No filtering applied"}


# ============================================================================
# 使用示例
# ============================================================================

if __name__ == "__main__":
    # 示例 1: 基础算子去重（GEMM）
    print("Example 1: GEMM deduplication")
    gemm_workloads = [
        {"name": "m256_n4096_k4096", "M": 256, "N": 4096, "K": 4096, "dtype": "fp16"},
        {"name": "m256_n4096_k4096_2", "M": 256, "N": 4096, "K": 4096, "dtype": "bf16"},  # 重复
        {"name": "m248_n4096_k4096", "M": 248, "N": 4096, "K": 4096, "dtype": "fp16"},
    ]

    filter = create_filter('mm', keep_first_k=1)
    filtered = filter.filter(gemm_workloads)
    print(f"Before: {len(gemm_workloads)}, After: {len(filtered)}")
    print(f"Stats: {filter.stats()}")
    print()

    # 示例 2: Attention 语义去重
    print("Example 2: Attention semantic deduplication")
    attention_workloads = [
        {"name": "bs8_kvlen73", "batch_size": 8, "num_kv_indices": 584, "num_qo_heads": 32, "num_kv_heads": 8, "head_dim": 128},  # avg_kv_len=73
        {"name": "bs16_kvlen73", "batch_size": 16, "num_kv_indices": 1168, "num_qo_heads": 32, "num_kv_heads": 8, "head_dim": 128},  # avg_kv_len=73, 应该去重
        {"name": "bs8_kvlen512", "batch_size": 8, "num_kv_indices": 4096, "num_qo_heads": 32, "num_kv_heads": 8, "head_dim": 128},  # avg_kv_len=512, 保留
    ]

    filter = create_filter('paged_attention_decode', keep_first_k=1)
    filtered = filter.filter(attention_workloads)
    print(f"Before: {len(attention_workloads)}, After: {len(filtered)}")
    print(f"Kept: {[wl['name'] for wl in filtered]}")
    print(f"Stats: {filter.stats()}")
