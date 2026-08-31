"""解析 model_configs/ 下的 JSON 为统一 ModelParams 结构"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class ModelParams:
    """模型架构参数的统一表示"""

    name: str                               # 文件名 stem, e.g. "deepseek_v4_pro"
    architecture: str = ""                  # model_type, e.g. "deepseek_v4"
    hidden_size: int = 0
    intermediate_size: int = 0              # dense FFN intermediate
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 128
    num_hidden_layers: int = 0

    # MLA
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None

    # MoE
    n_routed_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    n_shared_experts: Optional[int] = None
    n_group: Optional[int] = None               # grouped_topk 分组数
    topk_group: Optional[int] = None            # 从多少个 group 中选 expert
    routed_scaling_factor: Optional[float] = None
    scoring_func: Optional[str] = None          # "softmax" or "sigmoid"

    # Sparse attention (indexer)
    index_topk: Optional[int] = None
    index_n_heads: Optional[int] = None
    index_head_dim: Optional[int] = None
    sliding_window: Optional[int] = None

    # Quantization & dtype
    dtype: str = "bfloat16"
    expert_dtype: Optional[str] = None

    # Linear attention (Kimi K3, Qwen3.5 MoE)
    linear_attn_config: Optional[dict] = field(default=None, repr=False)

    @property
    def dtype_short(self) -> str:
        """bf16 / fp16 / fp32 短名"""
        mapping = {"bfloat16": "bf16", "float16": "fp16", "float32": "fp32"}
        return mapping.get(self.dtype, self.dtype)

    @property
    def mla_d(self) -> Optional[int]:
        """MLA 的 KV cache head 维度 = kv_lora_rank + qk_rope_head_dim
        这是 flash_mla kernel 的 d 参数（缓存维度）。
        """
        rope = self.qk_rope_head_dim or 0
        if self.kv_lora_rank and rope:
            return self.kv_lora_rank + rope
        # DeepSeek-V4 style: head_dim 本身是 latent dim (无显式 kv_lora_rank)
        if self.has_mla and rope:
            return self.head_dim + rope
        return None

    @property
    def mla_dv(self) -> Optional[int]:
        """MLA 的 value 维度"""
        if self.v_head_dim:
            return self.v_head_dim
        if self.kv_lora_rank:
            return self.kv_lora_rank
        # DeepSeek-V4 style: head_dim 就是 dv
        if self.has_mla:
            return self.head_dim
        return None

    @property
    def has_moe(self) -> bool:
        return self.n_routed_experts is not None and self.n_routed_experts > 0

    @property
    def has_mla(self) -> bool:
        """判断是否使用 MLA (Multi-head Latent Attention)"""
        # 显式有 kv_lora_rank
        if self.kv_lora_rank is not None and self.kv_lora_rank > 0:
            return True
        # DeepSeek-V4 style: q_lora_rank + num_kv_heads=1 + 大 head_dim
        if (self.q_lora_rank is not None and self.q_lora_rank > 0
                and self.num_key_value_heads == 1
                and self.head_dim >= 256):
            return True
        return False

    @property
    def ffn_size(self) -> int:
        """用于 SwiGLU/SiLU 的 N 维度"""
        if self.moe_intermediate_size:
            return self.moe_intermediate_size
        return self.intermediate_size // 2 if self.intermediate_size else 0


def _get_field(data: dict, *keys, default=None):
    """尝试多个 key，返回第一个存在的值"""
    for k in keys:
        if k in data and data[k] is not None:
            return data[k]
    return default


def parse_model_config(path: Path) -> ModelParams:
    """解析单个模型 config JSON"""
    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    # 有些模型参数在 text_config 子字段下（多模态模型）
    data = raw.get("text_config", raw) if "text_config" in raw else raw

    name = path.stem

    # 推导 head_dim
    head_dim = _get_field(data, "head_dim")
    if head_dim is None:
        # MLA 模型: head_dim = qk_nope_head_dim + qk_rope_head_dim
        nope = _get_field(data, "qk_nope_head_dim")
        rope = _get_field(data, "qk_rope_head_dim")
        if nope and rope:
            head_dim = nope + rope
        else:
            hidden = _get_field(data, "hidden_size", default=0)
            n_heads = _get_field(data, "num_attention_heads", default=1)
            head_dim = hidden // n_heads if n_heads else 128

    params = ModelParams(
        name=name,
        architecture=_get_field(data, "model_type", default="unknown"),
        hidden_size=_get_field(data, "hidden_size", default=0),
        intermediate_size=_get_field(data, "intermediate_size", default=0),
        num_attention_heads=_get_field(data, "num_attention_heads", default=0),
        num_key_value_heads=_get_field(data, "num_key_value_heads", default=0),
        head_dim=head_dim,
        num_hidden_layers=_get_field(data, "num_hidden_layers", default=0),
        # MLA
        q_lora_rank=_get_field(data, "q_lora_rank"),
        kv_lora_rank=_get_field(data, "kv_lora_rank"),
        qk_nope_head_dim=_get_field(data, "qk_nope_head_dim"),
        qk_rope_head_dim=_get_field(data, "qk_rope_head_dim"),
        v_head_dim=_get_field(data, "v_head_dim"),
        # MoE
        n_routed_experts=_get_field(data, "n_routed_experts", "num_experts"),
        num_experts_per_tok=_get_field(data, "num_experts_per_tok", "num_experts_per_token"),
        moe_intermediate_size=_get_field(data, "moe_intermediate_size"),
        n_shared_experts=_get_field(data, "n_shared_experts", "num_shared_experts"),
        n_group=_get_field(data, "n_group", "num_expert_group", "num_expert_groups"),
        topk_group=_get_field(data, "topk_group"),
        routed_scaling_factor=_get_field(data, "routed_scaling_factor"),
        scoring_func=_get_field(data, "moe_router_activation_func", "scoring_func"),
        # Sparse attention
        index_topk=_get_field(data, "index_topk"),
        index_n_heads=_get_field(data, "index_n_heads"),
        index_head_dim=_get_field(data, "index_head_dim"),
        sliding_window=_get_field(data, "sliding_window"),
        # Dtype
        dtype=_get_field(data, "dtype", "torch_dtype", default="bfloat16"),
        expert_dtype=_get_field(data, "expert_dtype"),
        # Linear attention
        linear_attn_config=_get_field(data, "linear_attn_config"),
    )

    # qwen3.8 格式: 顶层 linear_* 字段，自动合成 linear_attn_config
    if params.linear_attn_config is None and "linear_key_head_dim" in data:
        params.linear_attn_config = {
            "num_key_heads": data.get("linear_num_key_heads"),
            "num_value_heads": data.get("linear_num_value_heads"),
            "key_head_dim": data.get("linear_key_head_dim"),
            "value_head_dim": data.get("linear_value_head_dim"),
        }

    return params


def load_all_models(config_dir: Path) -> list[ModelParams]:
    """加载目录下所有模型 config"""
    models = []
    for p in sorted(config_dir.glob("*.json")):
        try:
            models.append(parse_model_config(p))
        except (json.JSONDecodeError, KeyError) as e:
            print(f"Warning: failed to parse {p.name}: {e}")
    return models
