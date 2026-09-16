"""Pack Seq Triton算子

将变长序列(N, D)打包为(B, Lmax, D)的padded batch tensor。
未填满的位置用pad_value填充。

签名:
    pack_seq_triton(x, lengths, pad_value=-inf, block_t=64, block_d=64)
        -> packed_tensor

输入:
    x: (N, D) 或 (N, ...) — 拼接的变长序列
    lengths: (B,) int32   — 每个序列的长度，sum(lengths)=N
    pad_value: float|int  — padding值，默认-inf
    block_t: int          — time维tile大小
    block_d: int          — feature维tile大小

输出:
    packed_tensor: (B, Lmax, D) 或 (B, Lmax, ...)
"""
import torch

from framework.base_operator import BaseOperator
from framework.registry import register_operator


@register_operator("pack_seq_triton")
class PackSeqTritonOperator(BaseOperator):

    @property
    def name(self) -> str:
        return "pack_seq_triton"

    @property
    def library(self) -> str:
        return "flaggems_vllm"

    def prepare_inputs(self, **params):
        """准备输入

        Args (casegen):
            num_tokens: 总token数
            index_topk: hidden维度（sparse attn的topk）
            dtype: 数据类型

        Args (direct):
            batch_size: batch大小
            seq_len: 平均序列长度
            hidden_size: feature维度
        """
        # 参数映射：casegen → operator
        if "num_tokens" in params:
            num_tokens = params["num_tokens"]
            # 推导 batch_size: 假设 decode 场景 batch≈num_tokens, prefill 场景 batch≈1-16
            # 使用启发式：num_tokens <= 64 → decode (batch=num_tokens, seq=1)
            #             num_tokens > 64  → mixed/prefill (batch=16, seq=num_tokens//16)
            if num_tokens <= 64:
                batch_size = num_tokens
                seq_len = 1
            else:
                batch_size = min(16, num_tokens)
                seq_len = num_tokens // batch_size
            hidden_size = params.get("index_topk", 512)
        else:
            batch_size = params["batch_size"]
            seq_len = params["seq_len"]
            hidden_size = params.get("hidden_size", 512)

        dtype_str = params.get("dtype", "bf16")
        dtype = self.get_dtype(dtype_str)

        # lengths: 每个序列长度，围绕seq_len随机波动
        lengths = torch.randint(
            max(1, seq_len // 2), seq_len + 1, (batch_size,),
            dtype=torch.int32, device="cuda"
        )
        N = int(lengths.sum().item())

        # x: (N, hidden_size)
        x = torch.randn(N, hidden_size, dtype=dtype, device="cuda")

        return {
            "x": x,
            "lengths": lengths,
        }

    def compute_flops(self, **params):
        """理论FLOPs — 纯数据搬运，几乎无计算"""
        return 0

    def compute_bytes(self, **params):
        """理论访存量

        读:
          x: N * hidden_size * elem_bytes
          lengths: batch_size * 4
        写:
          out: batch_size * Lmax * hidden_size * elem_bytes
        """
        # 参数映射：与 prepare_inputs 保持一致
        if "num_tokens" in params:
            num_tokens = params["num_tokens"]
            if num_tokens <= 64:
                batch_size = num_tokens
                seq_len = 1
            else:
                batch_size = min(16, num_tokens)
                seq_len = num_tokens // batch_size
            hidden_size = params.get("index_topk", 512)
        else:
            batch_size = params["batch_size"]
            seq_len = params["seq_len"]
            hidden_size = params.get("hidden_size", 512)

        dtype_str = params.get("dtype", "bf16")
        elem_bytes = self.dtype_bytes(dtype_str)

        N = batch_size * seq_len  # 近似
        Lmax = seq_len  # 近似

        read_bytes = N * hidden_size * elem_bytes + batch_size * 4
        write_bytes = batch_size * Lmax * hidden_size * elem_bytes

        return int(read_bytes + write_bytes)
