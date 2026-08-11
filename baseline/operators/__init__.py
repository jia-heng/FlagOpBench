"""Operators package

Auto-import all operator modules to trigger registration
"""

# Basic operators
from baseline.operators.basic import bmm
from baseline.operators.basic import mm
from baseline.operators.basic import grouped_matmul
from baseline.operators.basic import layernorm
from baseline.operators.basic import rms_norm
from baseline.operators.basic import gemma_rms_norm
from baseline.operators.basic import add_rmsnorm_bias
from baseline.operators.basic import fused_q_kv_rmsnorm
from baseline.operators.basic import gelu
from baseline.operators.basic import softmax
from baseline.operators.basic import silu_and_mul
from baseline.operators.basic import swiglu
from baseline.operators.basic import rope
from baseline.operators.basic import causal_conv1d
from baseline.operators.basic import topk
from baseline.operators.basic import topk_softplus_sqrt
from baseline.operators.basic import moe_sum
from baseline.operators.basic import router_gemm
from baseline.operators.basic import fp8_einsum
from baseline.operators.basic import silu_and_mul_with_clamp

# Attention operators
from baseline.operators.attention import flashattention
from baseline.operators.attention import flash_mla
from baseline.operators.attention import sparse_attention
from baseline.operators.attention import flash_linear_attention

# MoE operators
from baseline.operators.moe import fused_moe
from baseline.operators.moe import moe_align_block_size

# Quantization operators
from baseline.operators.quantization import gemm_w8a8
from baseline.operators.quantization import per_token_group_fp8_quant
from baseline.operators.quantization import fused_marlin_moe
from baseline.operators.quantization import fused_inv_rope_fp8_quant

# Norm operators
from baseline.operators.norm import kv_rms_norm_rope_cache
