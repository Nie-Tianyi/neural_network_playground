"""Edge payload 类型与 adapter 协议 —— 整个可视化抽象的地基。

## 为什么边不能是标量

一个神经网络图里，"一条边"到底是什么，取决于结构：

    nn.Linear          边是一个数 w        → 粗细 + 颜色
    KANLinear          边是一个函数 φ(x)   → 画成一条曲线
    FullQuadraticLayer 边是一个单项式 x₁x₂ → 画成一个 × 节点
    未知模块           不知道             → 画成灰虚线，但节点热力图仍然可用

如果把边写死成标量，KAN 这类结构就没法接进来；如果做成 `type` 字段
让前端 switch，那每加一种结构就要改前端。所以做成一个可扩展的
**联合类型**：adapter 负责生产 payload，前端只认 payload 的类型。

## 三个角色的划分

每个模块在图中扮演三种角色之一：

    WEIGHT      产生新的一列节点 + 连接上一列的边（Linear、KANLinear……）
    ELEMENTWISE 就地变换当前列的节点值，不产生新列（ReLU、Dropout、BN……）
    IGNORE      不参与可视化（容器模块等）

这条划分是整个 introspection 的骨架：列由 WEIGHT 模块决定，
节点上显示的数值 = 该列经过后续 ELEMENTWISE 模块变换后的结果
（因为 Playground 里"神经元的输出"指的是激活之后的值）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import torch.nn as nn


class Role(str, Enum):
    WEIGHT = "weight"
    ELEMENTWISE = "elementwise"
    IGNORE = "ignore"


# --- Edge payload 联合类型 ---------------------------------------------------
# 每个 payload 都有 `kind` 字段，它是前端 dispatch 的唯一依据。


@dataclass
class ScalarWeight:
    """一条边就是一个数。`nn.Linear` 以及一切"权重矩阵"型的层用它。

    渲染：线宽 clamp(|w|, 0, 5) → [1, 10]px，颜色 = 符号（橙负蓝正）。
    """

    value: float
    dead: bool = False  # 被 L1 正则推到过零并永久断开的连接（原版 isDead 规则）
    kind: str = field(default="scalar", init=False)


@dataclass
class FunctionCurve:
    """一条边是一个可学习的一元函数 φ(x)。KAN 用它。

    `ys` 是 φ 在 `xs` 上的取值。同时给出 base / spline 两个分支各自的贡献，
    这样 UI 可以拆开显示"φ 的两部分各学到了什么"——
    这是 KAN 区别于 MLP 最直观的地方。

    `importance` 是论文 Eq.2.17 的 |φ|₁ = mean|φ(x)|，
    渲染时映射成**透明度**（不是线宽——参考实现里线宽是固定的）。
    """

    xs: list[float]
    ys: list[float]
    importance: float
    base: list[float] | None = None  # w_b · silu(x) 分支
    spline: list[float] | None = None  # w_s · Σ c_i B_i(x) 分支
    kind: str = field(default="curve", init=False)


@dataclass
class ProductInteraction:
    """一条边是一个乘法项（如 x₁·x₂），由一个显式的「×」节点承载。

    `terms` 是参与连乘的源节点在本列中的下标。渲染时前端先画这些源节点
    汇入一个 × 虚拟节点，再从 × 节点伸一条带权边到目标神经元。
    """

    terms: list[int]
    weight: float
    kind: str = field(default="product", init=False)


@dataclass
class Opaque:
    """认不出来的模块。**这不是错误路径，而是"灵活"的兜底。**

    用户写了一个从没见过的新结构时，我们仍然能画出它的节点热力图
    （那是 forward hook 的功劳，与认不认识这个层无关），
    只是画不出它内部的连接细节。
    """

    label: str
    kind: str = field(default="opaque", init=False)


EdgePayload = ScalarWeight | FunctionCurve | ProductInteraction | Opaque


# --- Adapter 协议 ------------------------------------------------------------


@dataclass
class LayerView:
    """一个模块的可视化描述 —— adapter 的输出。"""

    role: Role
    kind: str  # "linear" | "kan" | "poly" | "activation" | "opaque" ...
    label: str = ""  # 图上显示的层名

    # 仅当 role == WEIGHT 时有意义
    width: int | None = None  # 这一列有几个节点（模块输出维度）
    edges: list[EdgePayload] | None = None  # 长度 width*prev_width，行优先 (out, in)

    # adapter 想额外告诉前端的东西（KAN 的 grid 范围、门控值……）
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LayerAdapter(Protocol):
    """把一个 nn.Module 翻译成 LayerView。

    `priority` 大的先被尝试 —— 这样"更具体的 adapter"可以覆盖通用的那个，
    比如 KANLinear 应该优先于任何把 nn.Module 一网打尽的兜底 adapter。
    """

    name: str
    priority: int

    # 兜底的 adapter 要置 True：它"匹配一切"，所以不能拿它来判断
    # "这个模块我认识"。见 introspection 里挑选 hook 目标的逻辑。
    is_fallback: bool = False

    def matches(self, module: nn.Module) -> bool: ...

    def describe(self, module: nn.Module, ctx: "DescribeContext") -> LayerView: ...

    def role(self, module: nn.Module) -> Role:
        """只回答"这个模块扮演什么角色"，不构造绘制内容。

        别小看这条捷径：Tracer 每跑一次前向就要对每个叶子模块问一次角色，
        如果一律走 `describe()`，KAN 那种要算曲线（每层上百条 × 48 个采样点）
        的 adapter 会在每帧白算两遍。角色一般只看类型就能定，代价是常数级。
        """
        ...


@dataclass
class DescribeContext:
    """describe() 需要的外部信息。

    `input_range` 对 KAN 至关重要：φ(x) 的绘图定义域必须是这一层输入的
    **实测范围**，否则曲线大部分是平的（见 layers.py 里关于 grid_range 的说明）。

    `dead_masks` 是 L1 正则永久冻结掉的连接，按**参数名**索引，形状与参数一致。
    只有引擎知道这些，adapter 拿不到，所以必须从外面传进来。
    """

    input_range: tuple[float, float] | None = None
    output_width: int | None = None
    input_width: int | None = None
    dead_masks: dict[str, Any] = field(default_factory=dict)


class Registry:
    """adapter 注册表。按 priority 降序尝试，第一个 matches 的胜出。"""

    def __init__(self) -> None:
        self._adapters: list[LayerAdapter] = []

    def register(self, adapter: LayerAdapter) -> LayerAdapter:
        self._adapters.append(adapter)
        self._adapters.sort(key=lambda a: a.priority, reverse=True)
        return adapter

    def resolve(self, module: nn.Module) -> LayerAdapter | None:
        for adapter in self._adapters:
            if adapter.matches(module):
                return adapter
        return None

    def recognizes(self, module: nn.Module) -> bool:
        """这个模块有没有**专用** adapter（而不是被兜底规则认领）。

        用来决定 hook 该挂在哪一层，见 introspection._select_hook_targets。
        """
        adapter = self.resolve(module)
        return adapter is not None and not getattr(adapter, "is_fallback", False)

    def role_of(self, module: nn.Module) -> Role:
        """廉价地判断模块角色（见 LayerAdapter.role 的说明）。"""
        adapter = self.resolve(module)
        if adapter is None:
            return Role.IGNORE
        try:
            return adapter.role(module)
        except Exception:
            return Role.IGNORE

    def describe(self, module: nn.Module, ctx: DescribeContext) -> LayerView:
        """解析并描述一个模块。认不出来就返回 Opaque 降级，不抛异常。"""
        adapter = self.resolve(module)
        if adapter is None:
            return LayerView(
                role=Role.IGNORE,
                kind="opaque",
                label=type(module).__name__,
            )
        try:
            return adapter.describe(module, ctx)
        except Exception as exc:  # adapter 有 bug 也不该让整页白屏
            return LayerView(
                role=Role.IGNORE,
                kind="opaque",
                label=f"{type(module).__name__} (adapter 出错: {exc})",
            )


def default_adapter_registry() -> Registry:
    """构建内置 adapter 的注册表。

    名字里带 `adapter` 是刻意的：`playground.registry` 里也有一个
    `default_registry()`，返回的是**模型**注册表。两个同名函数返回完全不同的东西，
    已经误导过一次了（测试里 import 错、报了个看不懂的 AttributeError）。

    注册顺序不重要 —— `Registry.register` 按 priority 降序排。
    专用的 adapter（KAN）priority 更高，会自然覆盖通用规则；
    兜底的 OpaqueFallbackAdapter priority 最低，只在没人认领时生效。
    """
    from .kan import KANLinearAdapter
    from .linear import (
        ElementwiseAdapter,
        LinearAdapter,
        OpaqueFallbackAdapter,
    )

    registry = Registry()
    registry.register(KANLinearAdapter())
    registry.register(LinearAdapter())
    registry.register(ElementwiseAdapter())
    registry.register(OpaqueFallbackAdapter())
    return registry
