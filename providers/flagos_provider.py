"""FlagOS Provider

FlagOS跨平台算子实现加载器。
加载FlagGems/FlagGems-vllm/FlagAttention，支持所有平台。
"""
from typing import Tuple, Callable, Dict, Optional

import torch

from framework.base_operator import BaseOperator
from .base_provider import BaseProvider
from .registry import register_provider


@register_provider("flagos", platform="all")
class FlagOSProvider(BaseProvider):
    """FlagOS算子实现加载器（跨平台，被测对象）"""

    def __init__(self):
        self._flaggems = None
        self._flaggems_vllm = None
        self._flagattention = None

    @property
    def name(self) -> str:
        return "flagos"

    @property
    def platform(self) -> str:
        return "all"

    def get_device(self) -> torch.device:
        """FlagOS支持多平台，根据当前环境返回设备"""
        if torch.cuda.is_available():
            return torch.device("cuda:0")
        # 预留: 后续支持其他平台设备
        return torch.device("cuda:0")

    def synchronize(self) -> None:
        """同步当前设备"""
        if torch.cuda.is_available():
            torch.cuda.synchronize()

    def is_available(self) -> bool:
        """检查FlagOS相关库是否可用"""
        try:
            import flag_gems
            return True
        except ImportError:
            pass
        try:
            import flaggems_vllm
            return True
        except ImportError:
            pass
        try:
            import flag_attn
            return True
        except ImportError:
            pass
        return False

    def setup(self):
        """延迟import FlagOS相关库"""
        try:
            import flag_gems
            self._flaggems = flag_gems
            print(f"  Loaded flag_gems: {flag_gems.__version__ if hasattr(flag_gems, '__version__') else 'unknown'}")
        except ImportError as e:
            print(f"  [WARN] Failed to import flag_gems: {e}")

        try:
            import flaggems_vllm
            self._flaggems_vllm = flaggems_vllm
            print(f"  Loaded flaggems_vllm")
        except ImportError as e:
            print(f"  [WARN] Failed to import flaggems_vllm: {e}")

        try:
            import flag_attn as flagattention
            self._flagattention = flagattention
            print(f"  Loaded flagattention (flag_attn): {flagattention.__version__ if hasattr(flagattention, '__version__') else 'unknown'}")
        except ImportError as e:
            print(f"  [WARN] Failed to import flagattention: {e}")

    def get_impl(
        self,
        op_name: str,
        operator: BaseOperator
    ) -> Tuple[Optional[Callable], Dict[str, str]]:
        """根据算子library属性加载对应实现"""
        lib = operator.library
        # 支持算子自定义impl函数名（与注册名不同时使用）
        fn_name = getattr(operator, "impl_name", op_name)

        # Special handling for operators with parameter name differences
        if op_name == "cp_gather_indexer_k_quant_cache":
            return self._load_cp_gather_indexer_k_quant_cache()

        if lib in ("flaggems", "flag_gems") and self._flaggems is not None:
            if hasattr(self._flaggems, fn_name):
                fn = getattr(self._flaggems, fn_name)
                return fn, {"source": f"flag_gems.{fn_name}", "type": "triton"}

        elif lib == "flaggems_vllm" and self._flaggems_vllm is not None:
            if hasattr(self._flaggems_vllm, fn_name):
                fn = getattr(self._flaggems_vllm, fn_name)
                return fn, {"source": f"flaggems_vllm.{fn_name}", "type": "triton"}

        elif lib == "flagattention" and self._flagattention is not None:
            if hasattr(self._flagattention, fn_name):
                fn = getattr(self._flagattention, fn_name)
                return fn, {"source": f"flagattention.{fn_name}", "type": "triton"}

        # 没有找到实现
        return None, {"error": f"No impl for {fn_name} in {lib}"}

    def _load_cp_gather_indexer_k_quant_cache(self):
        """Wrapper for cp_gather_indexer_k_quant_cache - maps vLLM parameter names to FlagOS names

        vLLM signature: (kv_cache, dst_k, dst_scale, block_table, cu_seq_lens)
        FlagOS signature: (k_cache, k_fp8, k_fp8_scale, block_table, cu_seqlen)
        """
        if self._flaggems_vllm is None or not hasattr(self._flaggems_vllm, "cp_gather_indexer_k_quant_cache"):
            return None, {"error": "cp_gather_indexer_k_quant_cache not found in flaggems_vllm"}

        flagos_fn = self._flaggems_vllm.cp_gather_indexer_k_quant_cache

        def wrapper(kv_cache, dst_k, dst_scale, block_table, cu_seq_lens, **kwargs):
            # Map parameter names from vLLM to FlagOS
            return flagos_fn(
                k_cache=kv_cache,
                k_fp8=dst_k,
                k_fp8_scale=dst_scale,
                block_table=block_table,
                cu_seqlen=cu_seq_lens
            )

        return wrapper, {"source": "flaggems_vllm.cp_gather_indexer_k_quant_cache (wrapped)", "type": "triton"}
