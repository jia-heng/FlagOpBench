from framework.base_operator import BaseOperator
from framework.registry import register_operator
import torch


@register_operator("topk")
class TopKOperator(BaseOperator):
    """
    通用 topk 算子

    用途:
    - 从输入张量的最后一个维度中选择前 k 个最大值
    - 常见场景: token sampling (vocab logits), expert selection, beam search

    FlagOS 实现: flag_gems.topk
    平台基线: torch.topk

    参数:
    - num_tokens: batch size (变化维度)
    - N: 选择维度大小 (例如 vocab_size)
    - k: 选择前 k 个元素
    - dtype: 数据类型
    """

    @property
    def name(self) -> str:
        return "topk"

    @property
    def library(self) -> str:
        return "flag_gems"

    def prepare_inputs(self, **params):
        """准备 topk 算子的输入

        Args:
            num_tokens: batch size
            N: dimension to select from (e.g., vocab_size)
            k: number of elements to select
            dtype: data type

        Returns:
            dict: {
                "x": (num_tokens, N) tensor,  # flag_gems uses 'x', torch.topk uses positional
                "k": int,
                "dim": -1,
                "largest": True,
                "sorted": True
            }
        """
        num_tokens = params.get("num_tokens") or params.get("N")
        N = params.get("N", 128000)  # dimension to select from
        k = params.get("k", 50)  # number of elements to pick
        dtype = self.get_dtype(params.get("dtype", "bfloat16"))

        # 生成输入: (num_tokens, N) 形状的张量
        input_tensor = torch.randn(num_tokens, N, dtype=dtype, device="cuda")

        return {
            "x": input_tensor,  # flag_gems uses 'x'
            "k": k,
            "dim": -1,  # 在最后一个维度上做 topk
            "largest": True,  # 选择最大的 k 个
            "sorted": True,  # 返回排序结果
        }

    def compute_flops(self, **params) -> int:
        """计算 topk 的理论 FLOPs

        topk 主要是比较操作，不是标准的浮点运算
        近似估计: O(N * log(k)) 次比较，每次比较算 1 FLOP

        Args:
            num_tokens: batch size
            N: dimension size
            k: number to select

        Returns:
            int: 理论 FLOPs
        """
        import math

        num_tokens = params.get("num_tokens") or params.get("N")
        N = params.get("N", 128000)
        k = params.get("k", 50)

        # Heap-based selection: O(N * log(k)) comparisons per row
        flops_per_row = N * max(1, int(math.log2(k)))
        total_flops = num_tokens * flops_per_row

        return int(total_flops)

    def compute_bytes(self, **params) -> int:
        """计算 topk 的理论访存量

        读:
        - input: (num_tokens, N) * elem_bytes

        写:
        - values: (num_tokens, k) * elem_bytes
        - indices: (num_tokens, k) * 4 (int32)

        Args:
            num_tokens: batch size
            N: dimension size
            k: number to select
            dtype: data type

        Returns:
            int: 理论访存量 (Bytes)
        """
        num_tokens = params.get("num_tokens") or params.get("N")
        N = params.get("N", 128000)
        k = params.get("k", 50)
        elem_bytes = self.dtype_bytes(params.get("dtype", "bfloat16"))

        # 读: 输入张量
        read_bytes = num_tokens * N * elem_bytes

        # 写: values + indices
        write_bytes = num_tokens * k * (elem_bytes + 4)  # 4 bytes for int32 indices

        return int(read_bytes + write_bytes)
