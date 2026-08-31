"""Unpack Seq Triton算子

将padded batch tensor (B, Lmax, D) 解包为拼接的变长序列 (N, D)。
pack_seq_triton的逆操作。

签名:
    unpack_seq_triton(packed_tensor, lengths, block_t=64, block_d=64)
        -> output

输入:
    packed_tensor: (B, Lmax, D) 或 (B, Lmax, ...) — padded batch
    lengths: (B,) int32 — 每个序列的长度，sum(lengths)=N
    block_t: int        — time维tile大小
    block_d: int        — feature维tile大小

输出:
    output: (N, D) 或 (N, ...)
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("unpack_seq_triton")
class UnpackSeqTritonOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "unpack_seq_triton"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """准备输入

        Args:
            batch_size: batch大小
            seq_len: 最大序列长度 (Lmax)
            hidden_size: feature维度
            dtype: 数据类型
        """
        batch_size = params["batch_size"]
        seq_len = params["seq_len"]
        hidden_size = params.get("hidden_size", 512)
        dtype_str = params.get("dtype", "bf16")
        dtype = self.get_dtype(dtype_str)

        # lengths: 每个序列长度
        lengths = torch.randint(
            max(1, seq_len // 2), seq_len + 1, (batch_size,),
            dtype=torch.int32, device="cuda"
        )

        Lmax = int(lengths.max().item())

        # packed_tensor: (B, Lmax, hidden_size)
        packed_tensor = torch.randn(
            batch_size, Lmax, hidden_size,
            dtype=dtype, device="cuda"
        )

        return {
            "packed_tensor": packed_tensor,
            "lengths": lengths,
        }

    def compute_flops(self, **params):
        """理论FLOPs — 纯数据搬运，几乎无计算"""
        return 0

    def compute_bytes(self, **params):
        """理论访存量

        读:
          packed_tensor: B * Lmax * hidden_size * elem_bytes (实际只读有效部分)
          lengths: B * 4
        写:
          out: N * hidden_size * elem_bytes
        """
        batch_size = params["batch_size"]
        seq_len = params["seq_len"]
        hidden_size = params.get("hidden_size", 512)
        dtype_str = params.get("dtype", "bf16")
        elem_bytes = self.dtype_bytes(dtype_str)

        N = batch_size * seq_len  # 近似
        # 实际只读/写有效token
        read_bytes = N * hidden_size * elem_bytes + batch_size * 4
        write_bytes = N * hidden_size * elem_bytes

        return int(read_bytes + write_bytes)
