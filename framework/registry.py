"""算子注册表"""
from typing import Dict, Type
import importlib
import pkgutil
from pathlib import Path

from .base_operator import BaseOperator

# 全局注册表
_OPERATOR_REGISTRY: Dict[str, Type[BaseOperator]] = {}


def register_operator(name: str):
    """算子注册装饰器

    Usage:
        @register_operator("swiglu")
        class SwiGLUOperator(BaseOperator):
            ...
    """
    def decorator(cls):
        if not issubclass(cls, BaseOperator):
            raise TypeError(f"{cls.__name__} must inherit from BaseOperator")
        _OPERATOR_REGISTRY[name] = cls
        return cls
    return decorator


def get_operator(name: str) -> BaseOperator:
    """从注册表获取算子实例"""
    if name not in _OPERATOR_REGISTRY:
        raise KeyError(
            f"Operator '{name}' not registered. "
            f"Available: {list(_OPERATOR_REGISTRY.keys())}"
        )
    return _OPERATOR_REGISTRY[name]()


def list_operators() -> list:
    """列出所有已注册的算子"""
    return sorted(_OPERATOR_REGISTRY.keys())


def import_all_operators():
    """导入所有算子模块以触发注册"""
    operators_dir = Path(__file__).parent.parent / "operators"
    if not operators_dir.exists():
        return

    for item in operators_dir.iterdir():
        if item.is_dir() and (item / "__init__.py").exists():
            module_name = f"operators.{item.name}"
            try:
                importlib.import_module(module_name)
            except ImportError:
                # 在某些环境下可能缺少依赖，跳过
                pass
