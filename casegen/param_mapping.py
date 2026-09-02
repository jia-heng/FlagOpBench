"""算子参数映射规则

从 operator_registry.yaml 加载算子列表，结合 const_axes_rule 推导参数。
"""

import yaml
from pathlib import Path
from typing import Callable

from .model_parser import ModelParams


# ============================================================
# const_axes 规则函数: 每种 rule 定义如何从 ModelParams 推导固定参数
# ============================================================

def _rule_moe(model: ModelParams) -> dict:
    return {
        "hidden_size": model.hidden_size,
        "intermediate_size": model.moe_intermediate_size,
        "num_experts": model.n_routed_experts,
        "topk": model.num_experts_per_tok,
        "dtype": model.dtype_short,
    }


def _rule_group_gemm_down(model: ModelParams) -> dict:
    """down projection: K=moe_intermediate_size, N=hidden_size"""
    return {
        "K": model.moe_intermediate_size,
        "N": model.hidden_size,
        "dtype": model.dtype_short,
    }


def _rule_group_gemm_gate_up(model: ModelParams) -> dict:
    """gate_up projection: K=hidden_size, N=2*moe_intermediate_size"""
    return {
        "K": model.hidden_size,
        "N": 2 * model.moe_intermediate_size if model.moe_intermediate_size else 0,
        "dtype": model.dtype_short,
    }


def _rule_flash_mla(model: ModelParams) -> dict:
    return {
        "s_q": 1,
        "h_q": model.num_attention_heads,
        "h_kv": model.num_key_value_heads,
        "d": model.mla_d,
        "dv": model.mla_dv,
        "block_size": 64,
    }


def _rule_flash_mla_fp8(model: ModelParams) -> dict:
    axes = _rule_flash_mla(model)
    axes["quant"] = "fp8"
    return axes


def _rule_flash_attn(model: ModelParams) -> dict:
    return {
        "num_heads": model.num_attention_heads,
        "head_dim": model.head_dim,
        "causal": True,
        "dtype": model.dtype_short,
    }


def _rule_swiglu(model: ModelParams) -> dict:
    return {
        "N": model.ffn_size,
        "dtype": model.dtype_short,
    }


def _rule_silu_and_mul(model: ModelParams) -> dict:
    return {
        "N": model.ffn_size,
        "dtype": model.dtype_short,
    }


def _rule_fused_q_kv_rmsnorm(model: ModelParams) -> dict:
    axes = {
        "hidden_size": model.hidden_size,
        "num_heads": model.num_attention_heads,
        "head_dim": model.head_dim,
        "dtype": model.dtype_short,
    }
    if model.q_lora_rank:
        axes["q_lora_rank"] = model.q_lora_rank
    if model.kv_lora_rank:
        axes["kv_lora_rank"] = model.kv_lora_rank
    return axes


def _rule_moe_sum(model: ModelParams) -> dict:
    return {
        "hidden_size": model.hidden_size,
        "topk": model.num_experts_per_tok,
        "dtype": model.dtype_short,
    }


def _rule_grouped_topk(model: ModelParams) -> dict:
    # scoring_func: "sigmoid" -> 1, else -> 0
    sf = 1 if model.scoring_func == "sigmoid" else 0
    return {
        "num_experts": model.n_routed_experts,
        "n_group": model.n_group,
        "topk_group": model.topk_group,
        "topk": model.num_experts_per_tok,
        "routed_scaling_factor": model.routed_scaling_factor or 1.0,
        "scoring_func": sf,
    }


def _rule_topk_softplus_sqrt(model: ModelParams) -> dict:
    return {
        "num_experts": model.n_routed_experts,
        "topk": model.num_experts_per_tok,
        "dtype": model.dtype_short,
    }


def _rule_top_k_per_row(model: ModelParams) -> dict:
    return {
        "num_experts": model.n_routed_experts,
        "topk": model.num_experts_per_tok,
    }


def _rule_mhc(model: ModelParams) -> dict:
    """Multi-Head Cache (MHC) pre/post 算子
    operator 期望: N (from var_axes), H, hc
    """
    return {
        "H": model.hidden_size,
        "hc": 4,  # MHC head cache 数，通常为 4
        "dtype": model.dtype_short,
    }


def _rule_sparse_attn(model: ModelParams) -> dict:
    return {
        "index_topk": model.index_topk,
        "index_n_heads": model.index_n_heads,
        "index_head_dim": model.index_head_dim,
        "sliding_window": model.sliding_window,
        "dtype": model.dtype_short,
    }


def _rule_pack_unpack_seq(model: ModelParams) -> dict:
    return {
        "index_topk": model.index_topk,
        "dtype": model.dtype_short,
    }


def _rule_indexer_cache(model: ModelParams) -> dict:
    return {
        "index_n_heads": model.index_n_heads,
        "index_head_dim": model.index_head_dim,
        "index_topk": model.index_topk,
        "dtype": model.dtype_short,
    }


def _rule_paged_mqa(model: ModelParams) -> dict:
    return {
        "h_q": model.num_attention_heads,
        "d": model.mla_d,
        "dv": model.mla_dv,
        "block_size": 64,
        "dtype": model.dtype_short,
    }


def _rule_v4_fused_qkv(model: ModelParams) -> dict:
    return {
        "hidden_size": model.hidden_size,
        "num_heads": model.num_attention_heads,
        "q_lora_rank": model.q_lora_rank,
        "kv_lora_rank": model.kv_lora_rank,
        "qk_rope_head_dim": model.qk_rope_head_dim,
        "dtype": model.dtype_short,
    }


def _rule_gdn(model: ModelParams) -> dict:
    """GDN (chunk_gated_delta_rule) 线性注意力算子

    支持两种 linear_attn_config 格式：
    - kimi_k3: {num_heads, head_dim} (Q/K/V 共享)
    - qwen3.8: {num_key_heads, num_value_heads, key_head_dim, value_head_dim} (分离)
    """
    lac = model.linear_attn_config
    if lac is None:
        return {"num_tokens": 2048, "num_heads": 96, "head_dim_k": 128, "dtype": model.dtype_short}

    # kimi_k3 格式
    if "num_heads" in lac:
        num_heads = lac["num_heads"]
        head_dim = lac["head_dim"]
        return {
            "num_heads": num_heads,
            "num_value_heads": num_heads,
            "head_dim_k": head_dim,
            "head_dim_v": head_dim,
            "batch_size": 1,
            "dtype": model.dtype_short,
        }

    # qwen3.8 格式
    return {
        "num_heads": lac.get("num_key_heads", 16),
        "num_value_heads": lac.get("num_value_heads", 128),
        "head_dim_k": lac.get("key_head_dim", 128),
        "head_dim_v": lac.get("value_head_dim", 128),
        "batch_size": 1,
        "dtype": model.dtype_short,
    }


def _rule_causal_conv1d(model: ModelParams) -> dict:
    return {
        "hidden_size": model.hidden_size,
        "dtype": model.dtype_short,
    }


def _rule_topk(model: ModelParams) -> dict:
    """TopK 算子: 通用 top-k 选择

    典型用例: token 采样从 vocab_size 维度选择 top-k
    参数:
        N: 选择维度大小（类似 vocab_size，固定为 128000）
        k: 选择的元素数量（固定为 50，典型采样值）
        dtype: 数据类型
    """
    return {
        "N": 128000,  # 类似 vocab_size 的典型值
        "k": 50,      # top-50 采样
        "dtype": model.dtype_short,
    }


def _rule_mm(model: ModelParams) -> dict:
    """矩阵乘法: K=hidden_size, N=hidden_size (方阵投影), dtype"""
    return {
        "K": model.hidden_size,
        "N": model.hidden_size,
        "dtype": model.dtype_short,
    }


def _rule_elementwise(model: ModelParams) -> dict:
    """逐元素算子 (add, sub 等): N=hidden_size, dtype"""
    return {
        "N": model.hidden_size,
        "alpha": 1,
        "dtype": model.dtype_short,
    }


def _rule_generic(model: ModelParams) -> dict:
    """通用 fallback: 只输出 dtype"""
    return {
        "dtype": model.dtype_short,
    }


# rule name -> function 的映射
CONST_AXES_RULES: dict[str, Callable[[ModelParams], dict]] = {
    "moe": _rule_moe,
    "group_gemm_down": _rule_group_gemm_down,
    "group_gemm_gate_up": _rule_group_gemm_gate_up,
    "flash_mla": _rule_flash_mla,
    "flash_mla_fp8": _rule_flash_mla_fp8,
    "flash_attn": _rule_flash_attn,
    "swiglu": _rule_swiglu,
    "silu_and_mul": _rule_silu_and_mul,
    "fused_q_kv_rmsnorm": _rule_fused_q_kv_rmsnorm,
    "moe_sum": _rule_moe_sum,
    "grouped_topk": _rule_grouped_topk,
    "topk_softplus_sqrt": _rule_topk_softplus_sqrt,
    "top_k_per_row": _rule_top_k_per_row,
    "topk": _rule_topk,
    "mhc": _rule_mhc,
    "sparse_attn": _rule_sparse_attn,
    "pack_unpack_seq": _rule_pack_unpack_seq,
    "indexer_cache": _rule_indexer_cache,
    "paged_mqa": _rule_paged_mqa,
    "v4_fused_qkv": _rule_v4_fused_qkv,
    "causal_conv1d": _rule_causal_conv1d,
    "gdn": _rule_gdn,
    "elementwise": _rule_elementwise,
    "mm": _rule_mm,
    "generic": _rule_generic,
}


# ============================================================
# 适用条件
# ============================================================

APPLICABLE_CONDITIONS: dict[str, Callable[[ModelParams], bool]] = {
    "always": lambda m: True,
    "has_moe": lambda m: m.has_moe,
    "has_mla": lambda m: m.has_mla,
    "has_sparse_attn": lambda m: m.index_topk is not None,
    "has_linear_attn": lambda m: m.linear_attn_config is not None,
    "has_grouped_topk": lambda m: m.has_moe and m.n_group is not None and m.n_group >= 1,
}


# ============================================================
# Registry loading
# ============================================================

def _default_registry_path() -> Path:
    return Path(__file__).resolve().parent.parent / "operator_registry.yaml"


def load_operator_registry(path: Path | None = None) -> dict:
    """从 operator_registry.yaml 加载算子注册表，返回 OPERATOR_REGISTRY 格式的 dict"""
    if path is None:
        path = _default_registry_path()

    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    registry = {}
    for op in data.get("operators", []):
        name = op["name"]
        mapping = op.get("param_mapping")
        if mapping is None:
            # planned 但无映射的算子跳过
            continue

        rule_name = mapping.get("const_axes_rule", "generic")
        var_key = mapping.get("var_axes_key", "num_tokens")
        applicable_name = mapping.get("applicable", "always")

        # 查找对应的函数
        const_fn = CONST_AXES_RULES.get(rule_name, _rule_generic)
        applicable_fn = APPLICABLE_CONDITIONS.get(applicable_name, lambda m: True)

        # 规范化 library 名
        lib = op.get("library", "flaggems_vllm")
        lib_normalized = {
            "FlagGems-vllm": "flaggems_vllm",
            "FlagGems": "flag_gems",
            "FlagAttention": "flag_attention",
            "flaggems_vllm": "flaggems_vllm",
            "flag_gems": "flag_gems",
            "flag_attention": "flag_attention",
        }.get(lib, lib)

        registry[name] = {
            "const_axes_fn": const_fn,
            "var_axes_key": var_key,
            "applicable": applicable_fn,
            "library": lib_normalized,
            "status": op.get("status", "planned"),
        }

    return registry


# 模块级别的 registry，首次 import 时加载
OPERATOR_REGISTRY: dict = {}


def _init_registry():
    global OPERATOR_REGISTRY
    try:
        OPERATOR_REGISTRY = load_operator_registry()
    except FileNotFoundError:
        # fallback: 空 registry
        OPERATOR_REGISTRY = {}


_init_registry()


# ============================================================
# Public API
# ============================================================

def get_applicable_operators(model: ModelParams, status: str | None = "implemented") -> list[str]:
    """返回对给定模型适用的算子列表

    Args:
        model: 模型参数
        status: 过滤状态, None 表示不过滤
    """
    results = []
    for op_name, spec in OPERATOR_REGISTRY.items():
        if status and spec.get("status") != status:
            continue
        if spec["applicable"](model):
            results.append(op_name)
    return results


def get_const_axes(op_name: str, model: ModelParams) -> dict | None:
    """获取算子在给定模型下的 const_axes，不适用返回 None"""
    spec = OPERATOR_REGISTRY.get(op_name)
    if spec is None:
        return None
    if not spec["applicable"](model):
        return None
    return spec["const_axes_fn"](model)


def get_var_axes_key(op_name: str) -> str | None:
    """获取算子对应的 var_axes 场景 key"""
    spec = OPERATOR_REGISTRY.get(op_name)
    return spec["var_axes_key"] if spec else None


def reload_registry(path: Path | None = None):
    """重新加载 registry (用于测试或动态更新)"""
    global OPERATOR_REGISTRY
    OPERATOR_REGISTRY = load_operator_registry(path)
