"""Import all operators"""
# 自动导入所有算子以触发注册
from pathlib import Path
import importlib

_current_dir = Path(__file__).parent

for item in _current_dir.iterdir():
    if item.is_dir() and (item / "__init__.py").exists() and item.name != "__pycache__":
        try:
            importlib.import_module(f"operators.{item.name}")
        except Exception as e:
            print(f"Warning: Failed to import operators.{item.name}: {e}")
