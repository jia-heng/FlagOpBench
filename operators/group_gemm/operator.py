"""Group GEMM 算子 (Grouped Matrix Multiplication)

基于Triton的分组矩阵乘法实现，用于MoE等场景中对多个不同大小的矩阵乘法进行批量计算。

签名:
    group_mm(A, B, offs) -> C

输入:
    A: (M_total, K) bf16          — 拼接的输入矩阵（所有 group 的行拼接在一起）
    B: (num_groups, K, N) bf16    — 各 group 的权重矩阵
    offs: (num_groups,) int32     — 各 group 的累积行偏移量

输出:
    C: (M_total, N) bf16

说明:
    offs[i] 表示前 i+1 个 group 一共有多少行。
    Group i 的行范围是 [offs[i-1], offs[i])（offs[-1]=0）。
    等价于: for i in range(num_groups): C[start:end] = A[start:end] @ B[i]
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("group_gemm")
class GroupGemmOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "group_gemm"

    @property
    def library(self) -> str:
        return "flag_gems"

    @property
    def impl_name(self) -> str:
        """实际函数名"""
        return "group_mm"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            num_groups: group 数量（即 expert 数）
            tokens_per_group: 每个 group 的 token 数（均匀分配时）
            K: 输入维度
            N: 输出维度
            dtype: 数据类型
        """
        num_groups = params.get("num_groups", 8)
        tokens_per_group = params.get("tokens_per_group", 64)
        K = params.get("K", 7168)
        N = params.get("N", 2048)
        dtype_str = params.get("dtype", "bf16")
        dtype = self.get_dtype(dtype_str)

        M_total = num_groups * tokens_per_group

        # A: (M_total, K) — 所有 group 的行拼接
        A = torch.randn(M_total, K, dtype=dtype, device="cuda")

        # B: (num_groups, K, N) — 各 group 的权重
        B = torch.randn(num_groups, K, N, dtype=dtype, device="cuda")

        # offs: cumulative row offsets, e.g. [64, 128, 192, 256, ...]
        offs = torch.arange(
            1, num_groups + 1, device="cuda", dtype=torch.int32
        ) * tokens_per_group

        return {
            "A": A,
            "B": B,
            "offs": offs,
        }

    def compute_flops(self, **params):
        """理论FLOPs

        每个 group: 2 * M_i * K * N (矩阵乘法)
        均匀分配时: num_groups * 2 * tokens_per_group * K * N
        """
        num_groups = params.get("num_groups", 8)
        tokens_per_group = params.get("tokens_per_group", 64)
        K = params.get("K", 7168)
        N = params.get("N", 2048)

        return num_groups * 2 * tokens_per_group * K * N

    def compute_bytes(self, **params):
        """理论访存量

        读:
          A: M_total * K * elem_bytes
          B: num_groups * K * N * elem_bytes
        写:
          C: M_total * N * elem_bytes
        """
        num_groups = params.get("num_groups", 8)
        tokens_per_group = params.get("tokens_per_group", 64)
        K = params.get("K", 7168)
        N = params.get("N", 2048)
        dtype_str = params.get("dtype", "bf16")
        elem_bytes = self.dtype_bytes(dtype_str)

        M_total = num_groups * tokens_per_group

        read_bytes = (
            M_total * K * elem_bytes       # A
            + num_groups * K * N * elem_bytes  # B
        )
        write_bytes = M_total * N * elem_bytes  # C

        return int(read_bytes + write_bytes)
