"""RMS Normalization 算子

Root Mean Square Layer Normalization

可用实现:
1. PyTorch 官方 F.rms_norm (PyTorch 2.4+)
2. vLLM CUDA kernel (torch.ops._C.rms_norm)
3. 手动实现 (fallback)

性能对比 (待实测后更新):
- PyTorch F.rms_norm: 官方 fused kernel，性能优秀
- vLLM rms_norm: 定制 CUDA kernel，针对推理场景优化
- 手动实现: 多步 op 组合，存在中间 tensor 开销

基线选择策略:
  PyTorch 2.4+ F.rms_norm 内部已 fused，与 vLLM kernel 性能接近。
  优先使用 PyTorch 官方（兼容性好）。
  如果实测 vLLM kernel 性能提升 > 5%，则切换为 vLLM 实现。
"""

import torch
import torch.nn.functional as F
from baseline.operators.registry import BaseOperator, register_operator

# 尝试导入 vLLM CUDA ops
try:
    import torch.ops._C as vllm_ops
    HAS_VLLM_OPS = hasattr(vllm_ops, 'rms_norm')
except (ImportError, AttributeError):
    HAS_VLLM_OPS = False


@register_operator("rms_norm")
class RMSNormOperator(BaseOperator):
    """RMSNorm: y = x / sqrt(mean(x^2) + eps) * weight

    Input: (M, hidden_size)
    使用 vLLM 官方 CUDA kernel 实现
    """

    @property
    def name(self) -> str:
        return "rms_norm"

    def forward(self, x: torch.Tensor, weight: torch.Tensor,
                eps: float = 1e-6, **kwargs) -> torch.Tensor:
        """RMS Normalization forward pass

        可用实现:
        1. PyTorch F.rms_norm - fused kernel，兼容性好
        2. vLLM CUDA kernel - 推理场景定制优化
        3. 手动实现 - 多步 ops 组合

        基线选择: PyTorch F.rms_norm（性能与 vLLM 接近，兼容性更优）
        TODO: 实测对比 PyTorch vs vLLM kernel 性能，差异 > 5% 则切换
        """
        if hasattr(F, 'rms_norm'):
            # 基线选择: PyTorch 官方 rms_norm (PyTorch 2.4+, fused kernel)
            return F.rms_norm(x, [x.shape[-1]], weight, eps)
        elif HAS_VLLM_OPS:
            # 备选: vLLM CUDA kernel (in-place, 推理优化)
            out = torch.empty_like(x)
            vllm_ops.rms_norm(out, x, weight, eps)
            return out
        else:
            # Fallback: 纯 PyTorch 实现 (性能较差，多步 ops 无 fusion)
            input_dtype = x.dtype
            x = x.float()
            variance = x.pow(2).mean(-1, keepdim=True)
            x = x * torch.rsqrt(variance + eps)
            return (x * weight.float()).to(input_dtype)

    def compute_flops(self, M: int = None, batch_size: int = None,
                      hidden_size: int = None, **kwargs) -> int:
        """RMSNorm FLOPs ≈ 4 * M * hidden_size (square, mean, rsqrt, mul)"""
        batch = M if M is not None else batch_size
        return 4 * batch * hidden_size

    def compute_bytes(self, M: int = None, batch_size: int = None,
                      hidden_size: int = None, dtype: str = "bf16", **kwargs) -> int:
        """访存 = 读 x + 读 weight + 写 output"""
        batch = M if M is not None else batch_size
        elem_bytes = self.dtype_bytes(dtype)
        read_x = batch * hidden_size * elem_bytes
        read_weight = hidden_size * elem_bytes
        write_out = batch * hidden_size * elem_bytes
        return read_x + read_weight + write_out

    def prepare_inputs(self, M: int = None, batch_size: int = None, num_tokens: int = None,
                       hidden_size: int = None, dtype: str = "bf16",
                       eps: float = 1e-6, **kwargs) -> dict:
        """准备输入 - 支持 M、batch_size 或 num_tokens 参数"""
        # 兼容三种命名: num_tokens (新) > M (旧) > batch_size (Definition 格式)
        batch = num_tokens if num_tokens is not None else (M if M is not None else batch_size)
        if batch is None:
            raise ValueError("Must specify either 'num_tokens', 'M' or 'batch_size'")

        torch_dtype = self.get_dtype(dtype)
        x = torch.randn(batch, hidden_size, device=self.device, dtype=torch_dtype)
        weight = torch.ones(hidden_size, device=self.device, dtype=torch_dtype)
        return {"x": x, "weight": weight, "eps": eps}

    def compute_golden(self, x: torch.Tensor, weight: torch.Tensor,
                       eps: float = 1e-6, **kwargs) -> torch.Tensor:
        x_fp32 = x.float().cpu()
        w_fp32 = weight.float().cpu()
        variance = x_fp32.pow(2).mean(-1, keepdim=True)
        x_norm = x_fp32 * torch.rsqrt(variance + eps)
        result = x_norm * w_fp32
        return result.to(x.dtype).to(x.device)
