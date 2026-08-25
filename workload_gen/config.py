"""
模型配置和推理场景定义
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json


@dataclass
class ModelConfig:
    """统一的模型配置抽象"""

    # 基础参数
    model_name: str
    model_type: str  # "llama", "mixtral", "qwen_moe", "deepseek_v3", "glm", "gemma"
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    intermediate_size: int
    num_hidden_layers: int
    vocab_size: int

    # RoPE 参数
    rope_theta: float = 10000.0
    max_position_embeddings: int = 4096

    # MoE 参数（可选）
    num_experts: Optional[int] = None
    num_experts_per_tok: Optional[int] = None
    moe_intermediate_size: Optional[int] = None

    # MLA 参数（可选，DeepSeek-V3）
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None

    # 其他
    rms_norm_eps: float = 1e-6

    @classmethod
    def from_json(cls, path: str) -> "ModelConfig":
        """从 JSON 配置文件加载"""
        with open(path) as f:
            data = json.load(f)

        return cls(
            model_name=data.get("model_name", Path(path).stem),
            model_type=data.get("model_type", "llama"),
            hidden_size=data["hidden_size"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data.get("num_key_value_heads", data["num_attention_heads"]),
            intermediate_size=data.get("intermediate_size", data["hidden_size"] * 4),
            num_hidden_layers=data.get("num_hidden_layers", 32),
            vocab_size=data.get("vocab_size", 32000),
            rope_theta=data.get("rope_theta", 10000.0),
            max_position_embeddings=data.get("max_position_embeddings", 4096),
            num_experts=data.get("num_experts"),
            num_experts_per_tok=data.get("num_experts_per_tok"),
            moe_intermediate_size=data.get("moe_intermediate_size"),
            q_lora_rank=data.get("q_lora_rank"),
            kv_lora_rank=data.get("kv_lora_rank"),
            qk_rope_head_dim=data.get("qk_rope_head_dim"),
            v_head_dim=data.get("v_head_dim"),
            rms_norm_eps=data.get("rms_norm_eps", 1e-6),
        )

    @property
    def head_dim(self) -> int:
        """计算 head dimension"""
        return self.hidden_size // self.num_attention_heads

    @property
    def is_moe(self) -> bool:
        """判断是否为 MoE 模型"""
        return self.num_experts is not None and self.num_experts > 1

    @property
    def is_mla(self) -> bool:
        """判断是否为 MLA 架构（DeepSeek-V3）"""
        return self.q_lora_rank is not None


@dataclass
class InferenceScenario:
    """推理场景参数"""

    phase: str          # "decode" | "prefill"
    batch_size: int     # 并发请求数
    seq_len: int        # 当前生成/处理的 token 数
    kv_len: int = 0     # KV cache 长度（decode 阶段非零）

    @property
    def num_tokens(self) -> int:
        """总 token 数"""
        return self.batch_size * self.seq_len

    @classmethod
    def standard_scenarios(cls) -> list["InferenceScenario"]:
        """返回标准测试场景集合"""
        scenarios = []

        # Decode 场景（batch 优先）
        for batch in [1, 4, 8, 16, 32, 64]:
            scenarios.append(cls(
                phase="decode",
                batch_size=batch,
                seq_len=1,
                kv_len=2048  # 假设 2K context
            ))

        # Prefill 场景（seq_len 优先）
        for seq_len in [128, 256, 512, 1024, 2048, 4096, 8192]:
            scenarios.append(cls(
                phase="prefill",
                batch_size=1,
                seq_len=seq_len,
                kv_len=0
            ))

        return scenarios

    def __str__(self) -> str:
        if self.phase == "decode":
            return f"decode_b{self.batch_size}_kv{self.kv_len}"
        else:
            return f"prefill_s{self.seq_len}"


@dataclass
class OperatorWorkload:
    """单个算子的 workload 记录"""

    op_name: str              # 算子名称（baseline 注册表中的名称）
    axes: dict[str, int]      # shape 参数
    const_params: dict        # 常量参数（dtype, eps 等）
    source: str               # 来源标识（用于 YAML 注释）
    phase: str = "mixed"      # decode / prefill / mixed

    def to_yaml_workload(self, name: str) -> dict:
        """转换为 YAML workload 格式"""
        workload = {
            "name": name,
            **self.axes,
            **self.const_params,
            "phase": self.phase,
            "source": self.source,
        }
        return workload


@dataclass
class WorkloadSet:
    """同一算子的 workload 集合（对应一个 YAML 文件）"""

    op_name: str
    model_name: str
    description: str
    workloads: list[OperatorWorkload] = field(default_factory=list)

    def add_workload(self, workload: OperatorWorkload):
        """添加 workload"""
        if workload.op_name != self.op_name:
            raise ValueError(f"Workload op_name mismatch: {workload.op_name} != {self.op_name}")
        self.workloads.append(workload)

    def to_yaml_dict(self) -> dict:
        """转换为 YAML 字典"""
        # 提取所有 const_params 作为 definition.const_axes
        const_axes = {}
        if self.workloads:
            # 假设所有 workload 的 const_params 一致
            const_axes = self.workloads[0].const_params.copy()

        # 提取所有 axes 作为 definition.var_axes
        var_axes_values = {}
        for wl in self.workloads:
            for key, value in wl.axes.items():
                if key not in var_axes_values:
                    var_axes_values[key] = set()
                var_axes_values[key].add(value)

        var_axes = {k: sorted(v) for k, v in var_axes_values.items()}

        # 生成 workloads 列表
        yaml_workloads = []
        for i, wl in enumerate(self.workloads):
            name = f"{self.model_name}_{wl.phase}_{i+1}"
            yaml_workloads.append(wl.to_yaml_workload(name))

        return {
            "operator": self.op_name,
            "description": self.description,
            "source": "traced",
            "definition": {
                "const_axes": const_axes,
                "var_axes": var_axes,
            },
            "workloads": yaml_workloads,
        }
