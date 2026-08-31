"""vLLM Provider (向后兼容别名)

已迁移为 nvidia_provider.py 中的 NvidiaProvider。
保留此文件以兼容 `from providers.vllm_provider import VLLMProvider` 的旧用法。
"""
from .nvidia_provider import NvidiaProvider


# 向后兼容: VLLMProvider 是 NvidiaProvider 的别名
class VLLMProvider(NvidiaProvider):
    """兼容旧接口 — 实际使用NvidiaProvider"""

    @property
    def name(self) -> str:
        # 保持旧的输出名，方便已有结果文件格式兼容
        return "vllm"
