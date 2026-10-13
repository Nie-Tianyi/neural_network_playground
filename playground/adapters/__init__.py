"""适配器：把一个 nn.Module 翻译成绘制指令。

新增一种结构（KAN、多项式层、RBF……）只需要在这里加一个 adapter，
不需要碰引擎、协议或前端渲染。
"""

from .base import (
    EdgePayload,
    FunctionCurve,
    LayerAdapter,
    Opaque,
    ProductInteraction,
    Registry,
    ScalarWeight,
    default_adapter_registry,
)

__all__ = [
    "EdgePayload",
    "FunctionCurve",
    "LayerAdapter",
    "Opaque",
    "ProductInteraction",
    "Registry",
    "ScalarWeight",
    "default_adapter_registry",
]
