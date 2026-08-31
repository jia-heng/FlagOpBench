"""Provider实现

导入此包时自动注册所有Provider到registry。
"""
# 导入所有Provider模块以触发 @register_provider 装饰器注册
from . import nvidia_provider
from . import flagos_provider
from . import ascend_provider
from . import metax_provider
from . import mthreads_provider
from . import iluvatar_provider
