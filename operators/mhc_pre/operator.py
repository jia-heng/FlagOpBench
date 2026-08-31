"""MHC Pre算子

Multi-Head Cache Pre-processing。用于MLA（Multi-head Latent Attention）的预处理。

功能:
    将residual通过GEMM+fused kernel计算出post_mix、comb_mix和layer_input。
    包含: GEMM投影、sqrsum归一化、sinkhorn迭代、加权求和等步骤。

签名:
    mhc_pre(residual, fn, hc_scale, hc_base, rms_eps, hc_pre_eps,
            hc_sinkhorn_eps, hc_post_mult_value, sinkhorn_repeat, n_splits=1)
        -> (post_mix, comb_mix, layer_input)

输入:
    residual: (N, hc, H)       — hc个残差流 (bfloat16)
    fn: (hc_mult3, hc*H)      — 投影矩阵 (float32), hc_mult3 = hc*2 + hc*hc
    hc_scale: (3,)            — 三组logits的缩放参数 (post/comb/layer)
    hc_base: (hc_mult3,)      — 基础偏移参数
    rms_eps: float             — RMS归一化的epsilon
    hc_pre_eps: float          — pre处理的epsilon
    hc_sinkhorn_eps: float     — sinkhorn迭代的epsilon
    hc_post_mult_value: float  — 后处理乘法值
    sinkhorn_repeat: int       — sinkhorn迭代次数

输出:
    post_mix: (N, hc, 1)      — 层混合权重 (float32)
    comb_mix: (N, hc, hc)     — 残差组合权重 (float32)
    layer_input: (N, H)       — 归一化后的层输入 (bfloat16)
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("mhc_pre")
class MHCPreOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "mhc_pre"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            N/num_tokens: token数（batch * seq_len）
            H: hidden_size
            hc: head cache数（通常为4）
            rms_eps: RMS归一化的epsilon
            hc_pre_eps: pre处理epsilon
            hc_sinkhorn_eps: sinkhorn迭代epsilon
            hc_post_mult_value: 后处理乘法值
            sinkhorn_repeat: sinkhorn迭代次数
            dtype: residual的数据类型（必须为bfloat16）
        """
        N = params.get("N") or params.get("num_tokens")
        H = params.get("H") or params.get("hidden_size")
        hc = params.get("hc", 4)
        rms_eps = params.get("rms_eps", 1e-5)
        hc_pre_eps = params.get("hc_pre_eps", 1e-3)
        hc_sinkhorn_eps = params.get("hc_sinkhorn_eps", 1e-3)
        hc_post_mult_value = params.get("hc_post_mult_value", 1.0)
        sinkhorn_repeat = params.get("sinkhorn_repeat", 5)

        # residual必须为bfloat16
        residual = torch.randn(N, hc, H, dtype=torch.bfloat16, device="cuda")

        # fn: (hc_mult3, hc*H), hc_mult3 = hc*2 + hc*hc
        hc_mult3 = hc * 2 + hc * hc
        hc_hidden_size = hc * H
        fn = torch.randn(hc_mult3, hc_hidden_size, dtype=torch.float32, device="cuda")

        # hc_scale: (3,) — 三组logits(post/comb/layer)各一个scale
        # hc_base: (hc_mult3,) — 逐元素偏移
        hc_scale = torch.randn(3, dtype=torch.float32, device="cuda")
        hc_base = torch.randn(hc_mult3, dtype=torch.float32, device="cuda")

        return {
            "residual": residual,
            "fn": fn,
            "hc_scale": hc_scale,
            "hc_base": hc_base,
            "rms_eps": rms_eps,
            "hc_pre_eps": hc_pre_eps,
            "hc_sinkhorn_eps": hc_sinkhorn_eps,
            "hc_post_mult_value": hc_post_mult_value,
            "sinkhorn_repeat": sinkhorn_repeat,
        }

    def compute_flops(self, **params):
        """理论FLOPs

        主要计算:
          1. GEMM: x_flat(N, hc*H) @ fn.T(hc*H, hc_mult3) -> (N, hc_mult3)
             FLOPs: 2 * N * hc*H * hc_mult3
          2. Fused kernel（sqrsum, norm, sinkhorn, weighted sum）近似:
             - RMS norm: N * H * 3 (sqr + sum + div)
             - Sinkhorn: N * hc * hc * sinkhorn_repeat * 4
             - Weighted sum: N * hc * H
        """
        N = params.get("N") or params.get("num_tokens")
        H = params.get("H") or params.get("hidden_size")
        hc = params.get("hc", 4)
        sinkhorn_repeat = params.get("sinkhorn_repeat", 5)

        hc_mult3 = hc * 2 + hc * hc
        hc_hidden_size = hc * H

        # GEMM dominates
        gemm_flops = 2 * N * hc_hidden_size * hc_mult3

        # Fused kernel approximate
        fused_flops = (
            N * H * 3                               # RMS norm
            + N * hc * hc * sinkhorn_repeat * 4     # sinkhorn iterations
            + N * hc * H                            # weighted sum
        )

        return int(gemm_flops + fused_flops)

    def compute_bytes(self, **params):
        """理论访存量

        读:
          residual: N * hc * H (bfloat16)
          fn: hc_mult3 * hc*H (float32) — 实际走cuBLAS缓存,但仍计入
          hc_scale: hc_mult3 (float32)
          hc_base: hc_mult3 (float32)
        写:
          post_mix: N * hc (float32)
          comb_mix: N * hc * hc (float32)
          layer_input: N * H (bfloat16)
        """
        N = params.get("N") or params.get("num_tokens")
        H = params.get("H") or params.get("hidden_size")
        hc = params.get("hc", 4)

        hc_mult3 = hc * 2 + hc * hc
        hc_hidden_size = hc * H

        read_bytes = (
            N * hc * H * 2                      # residual (bfloat16)
            + hc_mult3 * hc_hidden_size * 4     # fn (float32)
            + hc_mult3 * 4                      # hc_scale (float32)
            + hc_mult3 * 4                      # hc_base (float32)
        )
        write_bytes = (
            N * hc * 4                          # post_mix (float32)
            + N * hc * hc * 4                   # comb_mix (float32)
            + N * H * 2                         # layer_input (bfloat16)
        )

        return int(read_bytes + write_bytes)
