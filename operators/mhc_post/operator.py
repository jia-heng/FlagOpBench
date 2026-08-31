"""MHC Post算子

Multi-Head Cache Post-processing。用于MLA（Multi-head Latent Attention）的后处理。

计算:
    out[n, i, h] = post_layer_mix[n, i] * x[n, h]
                 + sum_j(comb_res_mix[n, j, i] * residual[n, j, h])

输入:
    x: (N, H)              — 当前层输出
    residual: (N, hc, H)   — hc个残差流
    post_layer_mix: (N, hc) — 层混合权重 (float32)
    comb_res_mix: (N, hc, hc) — 残差组合权重 (float32)

输出:
    out: (N, hc, H)
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("mhc_post")
class MHCPostOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "mhc_post"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            N/num_tokens: token数（batch * seq_len）
            H: hidden_size
            hc: head cache数（通常为4）
            dtype: x和residual的数据类型
        """
        N = params.get("N") or params.get("num_tokens")
        H = params["H"]
        hc = params.get("hc", 4)
        dtype = self.get_dtype(params.get("dtype", "bfloat16"))

        x = torch.randn(N, H, dtype=dtype, device="cuda")
        residual = torch.randn(N, hc, H, dtype=dtype, device="cuda")
        post_layer_mix = torch.randn(N, hc, dtype=torch.float32, device="cuda")
        comb_res_mix = torch.randn(N, hc, hc, dtype=torch.float32, device="cuda")

        return {
            "x": x,
            "residual": residual,
            "post_layer_mix": post_layer_mix,
            "comb_res_mix": comb_res_mix,
        }

    def compute_flops(self, **params):
        """理论FLOPs

        对于每个(n, i, h):
          - post_layer_mix[n,i] * x[n,h]: 1 mul
          - sum_j(comb_res_mix[n,j,i] * residual[n,j,h]): hc mul + (hc-1) add
          - 加法: 1 add
        总: N * hc * H * (2*hc + 1)
        """
        N = params.get("N") or params.get("num_tokens")
        H = params["H"]
        hc = params.get("hc", 4)
        return N * hc * H * (2 * hc + 1)

    def compute_bytes(self, **params):
        """理论访存量

        读:
          x: N * H (dtype)
          residual: N * hc * H (dtype)
          post_layer_mix: N * hc (float32)
          comb_res_mix: N * hc * hc (float32)
        写:
          out: N * hc * H (dtype)
        """
        N = params.get("N") or params.get("num_tokens")
        H = params["H"]
        hc = params.get("hc", 4)
        elem_bytes = self.dtype_bytes(params.get("dtype", "bfloat16"))

        read_bytes = (
            N * H * elem_bytes              # x
            + N * hc * H * elem_bytes       # residual
            + N * hc * 4                    # post_layer_mix (float32)
            + N * hc * hc * 4              # comb_res_mix (float32)
        )
        write_bytes = N * hc * H * elem_bytes  # out

        return int(read_bytes + write_bytes)
