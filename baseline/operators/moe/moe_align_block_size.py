"""MoE Align Block Size 算子

对齐 MoE block size，提高 GPU 利用率

可用实现:
1. PyTorch 手动实现 (torch.cat padding)
2. vLLM CUDA kernel (torch.ops._C.moe_align_block_size)

性能对比 (待实测后更新):
- PyTorch padding: 简单 cat + zeros，无额外 kernel launch 开销
- vLLM CUDA kernel: 单 kernel 完成 padding + reorder，减少 memory copy

基线选择策略:
  此算子计算量极小 (FLOPs ≈ 0)，主要是 memory 操作。
  PyTorch cat 实现足够高效，且无外部依赖。
  如果实测 vLLM kernel 有 > 5% 提升（大 batch 场景），则切换。
"""

import torch
from baseline.operators.registry import BaseOperator, register_operator

# 尝试导入 vLLM CUDA ops
try:
    import torch.ops._C as vllm_ops
    HAS_VLLM_MOE_ALIGN = hasattr(vllm_ops, 'moe_align_block_size')
except (ImportError, AttributeError):
    HAS_VLLM_MOE_ALIGN = False


@register_operator("moe_align_block_size")
class MoeAlignBlockSizeOperator(BaseOperator):
    """MoE Align Block Size: 对齐 block size 提高 GPU 利用率

    Input:
        topk_ids: (num_tokens, top_k) - expert IDs
        block_size: int - alignment size (通常 64)
    Output:
        aligned_topk_ids: (aligned_num_tokens, top_k)
        num_valid_tokens: int
    """

    @property
    def name(self) -> str:
        return "moe_align_block_size"

    def forward(self, topk_ids: torch.Tensor, block_size: int = 64,
                **kwargs) -> tuple:
        """MoE Align Block Size 前向

        可用实现:
        1. PyTorch cat padding - 简单高效 ⭐ 当前基线
        2. vLLM CUDA kernel - 单 kernel 完成

        基线选择: PyTorch padding（此算子 compute ≈ 0，memory 操作 cat 已足够）
        TODO: 实测大 batch (>4096 tokens) 场景下 vLLM kernel 是否有优势
        """
        if HAS_VLLM_MOE_ALIGN:
            # 备选: vLLM CUDA kernel (大 batch 可能更优)
            # TODO: 实测后如果性能提升 > 5% 则切换为默认
            pass  # 暂不使用，等待实测

        num_tokens = topk_ids.shape[0]
        top_k = topk_ids.shape[1]

        # 基线: PyTorch padding
        padded_size = ((num_tokens + block_size - 1) // block_size) * block_size

        if padded_size == num_tokens:
            return topk_ids, num_tokens

        padding_size = padded_size - num_tokens
        padding = torch.zeros(padding_size, top_k, dtype=topk_ids.dtype,
                            device=topk_ids.device)
        aligned_topk_ids = torch.cat([topk_ids, padding], dim=0)

        return aligned_topk_ids, num_tokens

    def compute_flops(self, num_tokens: int, top_k: int = 2,
                      block_size: int = 64, **kwargs) -> int:
        """几乎没有计算，主要是 memory copy"""
        return 0

    def compute_bytes(self, num_tokens: int, top_k: int = 2,
                      block_size: int = 64, **kwargs) -> int:
        """访存 = 读 topk_ids + 写 aligned_topk_ids"""
        padded_size = ((num_tokens + block_size - 1) // block_size) * block_size
        read_ids = num_tokens * top_k * 4  # int32
        write_ids = padded_size * top_k * 4
        return read_ids + write_ids

    def prepare_inputs(self, num_tokens: int, num_experts: int = 8,
                       top_k: int = 2, block_size: int = 64, **kwargs) -> dict:
        # 生成随机 expert IDs
        topk_ids = torch.randint(0, num_experts, (num_tokens, top_k),
                                device=self.device, dtype=torch.int32)
        return {"topk_ids": topk_ids, "block_size": block_size}

    def compute_golden(self, topk_ids: torch.Tensor, block_size: int = 64,
                       **kwargs) -> tuple:
        """Golden reference"""
        topk_ids_cpu = topk_ids.cpu()
        num_tokens = topk_ids_cpu.shape[0]
        top_k = topk_ids_cpu.shape[1]

        padded_size = ((num_tokens + block_size - 1) // block_size) * block_size

        if padded_size == num_tokens:
            return topk_ids, num_tokens

        padding_size = padded_size - num_tokens
        padding = torch.zeros(padding_size, top_k, dtype=topk_ids_cpu.dtype)
        aligned_topk_ids = torch.cat([topk_ids_cpu, padding], dim=0)

        return aligned_topk_ids.to(topk_ids.device), num_tokens
