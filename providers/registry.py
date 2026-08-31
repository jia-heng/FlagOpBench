"""Provider注册与工厂

提供Provider注册机制和按平台/实现名查找Provider的能力。
"""
from typing import Dict, Optional, Type

from providers.base_provider import BaseProvider


# 全局注册表: {(platform, impl_name): ProviderClass}
_PROVIDER_REGISTRY: Dict[tuple, Type[BaseProvider]] = {}

# 平台默认Provider映射: {platform: impl_name}
_PLATFORM_DEFAULTS: Dict[str, str] = {}


def register_provider(impl_name: str, platform: str, is_default: bool = False):
    """Provider注册装饰器

    Args:
        impl_name: 实现名称，如 "nvidia", "ascend", "flagos"
        platform: 所属平台，如 "nvidia", "ascend", "all"(跨平台)
        is_default: 是否为该平台的默认Provider

    Usage:
        @register_provider("nvidia", platform="nvidia", is_default=True)
        class NvidiaProvider(BaseProvider):
            ...
    """
    def decorator(cls: Type[BaseProvider]):
        _PROVIDER_REGISTRY[(platform, impl_name)] = cls
        if is_default:
            _PLATFORM_DEFAULTS[platform] = impl_name
        return cls
    return decorator


def get_provider(platform: str, impl: Optional[str] = None) -> BaseProvider:
    """获取Provider实例

    Args:
        platform: 目标平台 nvidia / ascend / metax / mthreads / iluvatar
        impl: 指定实现名（可选）。
              None = 使用平台默认Provider
              "flagos" = 使用FlagOS Provider

    Returns:
        Provider实例（已调用setup）

    Raises:
        ValueError: 找不到对应的Provider
    """
    if impl is None:
        # 使用平台默认
        impl = _PLATFORM_DEFAULTS.get(platform)
        if impl is None:
            raise ValueError(
                f"平台 '{platform}' 没有注册默认Provider。"
                f"已注册的平台: {list(_PLATFORM_DEFAULTS.keys())}"
            )

    # 查找: 先精确匹配 (platform, impl)，再尝试 ("all", impl) 跨平台
    key = (platform, impl)
    cls = _PROVIDER_REGISTRY.get(key)

    if cls is None:
        # 尝试跨平台Provider（如 flagos）
        key = ("all", impl)
        cls = _PROVIDER_REGISTRY.get(key)

    if cls is None:
        available = [
            f"{p}/{n}" for (p, n) in _PROVIDER_REGISTRY.keys()
        ]
        raise ValueError(
            f"找不到Provider: platform='{platform}', impl='{impl}'。"
            f"已注册: {available}"
        )

    # 检查可用性
    provider = cls()
    if not provider.is_available():
        raise RuntimeError(
            f"Provider '{impl}' (platform={platform}) 不可用。"
            f"请检查依赖是否安装、设备是否就绪。"
        )

    provider.setup()
    return provider


def list_providers() -> Dict[str, list]:
    """列出所有已注册的Provider

    Returns:
        {platform: [impl_name, ...]}
    """
    result: Dict[str, list] = {}
    for (platform, impl_name) in _PROVIDER_REGISTRY.keys():
        result.setdefault(platform, []).append(impl_name)
    return result
