"""把一次前向传播变成可绘制的网络图 —— 项目的核心。

## 思路

不解析模型源码，而是**跑一遍前向传播，用 forward hook 记录每个叶子模块的
输入输出**。这一步对任意 nn.Module 都成立，不需要认识任何层类型。
然后 adapter 负责把这些张量「翻译」成绘制指令，翻译不了就降级。

这样做的直接好处：用户写一个我们完全没见过的结构，图仍然能画出来
（至少能画节点热力图），不会白屏。

## 一个容易踩的坑：节点显示的是激活后的值

Playground 里"神经元的输出"指的是 `activation(weighted_sum)`，
而 `nn.Linear` 的输出只是 `weighted_sum`。中间隔着激活层。

所以不能简单地"每遇到一个 Linear 就开一列" —— 还得知道**哪个激活层
消费了这个 Linear 的输出**。这里的做法是顺着张量同一性往下找：
`Tanh1` 的输入**就是** `Linear1` 的输出对象（nn.Sequential 直接传递引用），
所以能可靠地配对。

顺带一提，这个"沿数据流走"的做法也解释了为什么它在分支结构上会退化 ——
残差连接处会有多个消费者，我们不猜，直接停在那里（见 `_find_consumer`）。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Self

import torch
from torch import nn

from .adapters.base import (
    DescribeContext,
    EdgePayload,
    Registry,
    Role,
)

# 顺着逐元素层往下走的步数上限，纯粹为了防御环形结构。
_MAX_CHAIN = 64


# --- 追踪 --------------------------------------------------------------------


@dataclass
class TraceRecord:
    """一个叶子模块的一次调用。"""

    name: str
    module: nn.Module
    input: torch.Tensor
    output: torch.Tensor
    role: Role


class Tracer:
    """按执行顺序记录每个叶子模块的输入输出。

    只 hook **叶子模块**（没有子模块的）。如果连容器一起 hook，
    `nn.Sequential` 和它内部的 `Linear` 都会被记一次，同一条边出现两遍。

    用法::

        with Tracer(model, registry) as tracer:
            model(x)
        records = tracer.records
    """

    def __init__(self, model: nn.Module, registry: Registry) -> None:
        self.model = model
        self.registry = registry
        self.records: list[TraceRecord] = []
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> Self:
        self.records = []
        for name, module in _select_hook_targets(self.model, self.registry):
            self._handles.append(module.register_forward_hook(self._hook(name)))
        return self

    def __exit__(self, *exc_info) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _hook(self, name: str):
        def hook(module: nn.Module, inputs, output) -> None:
            # 有些模块的输入输出不是张量（RNN 的状态、dict 等），直接跳过，
            # 让它们在图上退化成"不显示"，而不是抛异常。
            if not inputs or not isinstance(inputs[0], torch.Tensor):
                return
            if not isinstance(output, torch.Tensor):
                return
            # 用 role_of 而不是 describe：只需要知道角色，不该为每个模块
            # 构造一遍绘制内容（KAN 那种要算上百条曲线的会很贵）。
            role = self.registry.role_of(module)
            self.records.append(TraceRecord(name, module, inputs[0], output, role))

        return hook


def _select_hook_targets(model: nn.Module, registry: Registry) -> list[tuple[str, nn.Module]]:
    """挑选要挂 forward hook 的模块。

    规则：**能匹配到专用 adapter 的模块就地挂钩，不再往下走**；
    认不出来的模块，有子模块就继续下潜（当容器看待），没有就自己挂钩（当叶子看待）。

    这里的关键是判据用"认不认识"而不是"有没有子模块"。曾经用后者，
    结果 `KANLinear` 因为内部有个 `self.base_activation = nn.SiLU()` 而被当成容器跳过，
    只记录到它内部的 SiLU —— 整个 KAN 层在图上消失了。

    反过来，`nn.Sequential` 这类纯容器会被"认不认识"这一条正确地区分出来：
    它匹配不到专用 adapter，于是继续下潜到里面的 Linear / 激活层。
    """
    targets: list[tuple[str, nn.Module]] = []

    def visit(module: nn.Module, name: str) -> None:
        if name and registry.recognizes(module):
            targets.append((name, module))
            return
        children = list(module.named_children())
        if not children:
            if name:
                targets.append((name, module))
            return
        for child_name, child in children:
            visit(child, f"{name}.{child_name}" if name else child_name)

    visit(model, "")
    return targets


def _find_consumer(records: list[TraceRecord], start: int, tensor: torch.Tensor) -> int | None:
    """找出消费了 `tensor` 的那个模块。

    返回**逐元素层**的下标；如果遇到权重层就返回 None
    （说明当前这一列到此为止，下一个权重层要开新列了）。

    这里用张量对象同一性判断，而不是形状或数值 —— nn.Sequential 是直接
    传引用的，所以同一性判断既快又准。
    """
    for j in range(start + 1, len(records)):
        record = records[j]
        if record.input is not tensor:
            continue
        if record.role is Role.WEIGHT:
            return None  # 被下一个权重层直接吃掉了，没有激活层
        if record.role is Role.ELEMENTWISE:
            return j
    return None


def _resolve_display_record(records: list[TraceRecord], index: int) -> int:
    """从 `records[index]` 出发，走到这一列"真正的显示值"。

    比如 Linear → Tanh → Linear，第一列显示的是 Tanh 的输出而不是 Linear 的。
    没有后继激活层时就是自己。
    """
    current = index
    for _ in range(_MAX_CHAIN):
        nxt = _find_consumer(records, current, records[current].output)
        if nxt is None:
            break
        current = nxt
    return current


# --- 图结构 ------------------------------------------------------------------


@dataclass
class Node:
    index: int
    label: str
    column: int
    # 只对输入列有意义：这个特征当前是否被喂给模型。
    # 未选中的特征**仍然画出来**（空框），这样"点最左列开关特征"这个交互
    # 才成立 —— 否则没被选中的特征根本不在图上，也就无从点起。
    active: bool = True


@dataclass
class Column:
    title: str
    kind: str  # "input" | "hidden" | "output"
    nodes: list[Node] = field(default_factory=list)

    @property
    def width(self) -> int:
        return len(self.nodes)


@dataclass
class Edge:
    src: int  # 源节点在本列中的下标
    dst: int  # 目标节点在下一列中的下标
    payload: EdgePayload


@dataclass
class DisplayGraph:
    """一次前向传播的完整可视化描述。

    `columns` 是图的骨架，`edges[k]` 连接 `columns[k]` 和 `columns[k+1]`。
    另外带上每一列的激活张量（扁平化后的网格值），供引擎做热力图。
    """

    columns: list[Column]
    edges: list[list[Edge]]
    activations: list[torch.Tensor]  # 每列一个 (N, width) 张量
    input_ranges: list[tuple[float, float]]  # 每列输入张量的实测范围

    @property
    def output_column(self) -> Column:
        return self.columns[-1]


def build_graph(
    model: nn.Module,
    x: torch.Tensor,
    input_labels: list[str],
    active_inputs: list[int] | None,
    registry: Registry,
    input_grid: torch.Tensor | None = None,
    dead_lookup: Callable[[nn.Module], dict[str, torch.Tensor]] | None = None,
) -> DisplayGraph:
    """跑一遍前向，把模型画成一张图。

    `x` 通常是决策边界的采样网格（形状 (N, 已选特征数)），这样一次前向同时
    得到网络结构和所有神经元的热力图，不用跑两遍。

    `input_labels` 是**全部**可选特征的显示名；`active_inputs` 是其中被选中的
    那些在 `input_labels` 里的下标，**顺序必须和喂给模型的列顺序一致**
    （输入列会把全部特征都画出来，未选中的标成 inactive）。

    这个顺序很重要：模型输入的第 k 列未必是显示列的第 k 个方块
    （比如只勾了 X₁X₂，它在显示列里排第 5），所以铺边时要做一次下标映射，
    否则权重会被画到错误的边上去。

    `input_grid` 是全部特征在网格上的取值 (N, len(input_labels))，用来给那些
    **没被喂进模型**的特征也画热力图（它们的值不在前向传播的路径上，拿不到）。
    """
    with torch.no_grad(), Tracer(model, registry) as tracer:
        model(x)

    records = tracer.records
    if not records:
        # 模型没有任何叶子模块（或者前向根本没被 hook 到）。
        return _degenerate_graph(x, input_labels, active_inputs, input_grid)

    _promote_width_changing(records)

    weight_indices = [i for i, r in enumerate(records) if r.role is Role.WEIGHT]
    if not weight_indices:
        return _degenerate_graph(x, input_labels, active_inputs, input_grid)

    columns: list[Column] = [_input_column(input_labels, active_inputs)]
    edges: list[list[Edge]] = []
    display_input = input_grid if input_grid is not None else x
    activations: list[torch.Tensor] = [display_input.detach()]
    input_ranges: list[tuple[float, float]] = [_range_of(display_input)]

    # 每一列**喂给模型的**宽度，和 columns[i].width（显示宽度）是两回事。
    # 输入列尤其明显：它画全部 7 个特征（这样没选中的也能点开），
    # 但模型只吃其中选中的那几个。铺边必须按模型宽度来算，
    # 否则权重矩阵会被摊到错误的边上 —— 见 _build_edges 的注释。
    model_widths: list[int] = [x.shape[-1]]

    for position, record_index in enumerate(weight_indices):
        record = records[record_index]
        view = registry.describe(
            record.module,
            DescribeContext(
                # 这一层输入的实测范围。KAN 之类需要"按定义域画曲线"的 adapter 用得上，
                # 而且正好能暴露"数据跑出了 spline 网格"这个静默失效的情况。
                input_range=_range_of(record.input),
                dead_masks=dead_lookup(record.module) if dead_lookup else {},
            ),
        )

        # 这一列显示的数值：从权重层出发，沿着逐元素层走到链尾
        # （Playground 里"神经元的输出"指的是激活之后的值，不是加权和）。
        display = records[_resolve_display_record(records, record_index)]
        activation = display.output.detach()
        if activation.dim() == 1:
            activation = activation.unsqueeze(-1)
        width = activation.shape[-1]

        is_last = position == len(weight_indices) - 1
        kind = "output" if is_last else "hidden"
        title = "OUTPUT" if is_last else f"HIDDEN {position + 1}"

        columns.append(
            Column(
                title=title,
                kind=kind,
                nodes=[Node(index=i, label=_node_label(kind, i), column=len(columns)) for i in range(width)],
            )
        )
        activations.append(activation)
        input_ranges.append(_range_of(record.input))

        # 以**实际张量宽度**为准来铺边，而不是 adapter 声称的宽度：
        # 用户写了怪结构时两者可能对不上，多出来的边直接丢弃，不画出界。
        #
        # 入边宽度取 model_widths[-1]（模型的真实输入列数），不是
        # columns[-2].width（显示列宽）。第一层两者不等：输入列画 7 个特征方块，
        # 模型只吃选中的那几个。用错的话权重矩阵会被摊到错误的边上 ——
        # 表现是"后面的神经元一条入边都没有，前面几个却从没选中的特征收到边"。
        edges.append(
            _build_edges(
                view.edges,
                out_width=width,
                in_width=model_widths[-1],
                src_map=active_inputs if position == 0 else None,
            )
        )
        model_widths.append(width)

    return DisplayGraph(
        columns=columns,
        edges=edges,
        activations=activations,
        input_ranges=input_ranges,
    )


def _promote_width_changing(records: list[TraceRecord]) -> None:
    """把"会改变宽度"的逐元素层提升成独立的一列。

    `nn.GLU` 这类层输入 6 维输出 3 维，`nn.Flatten` 也会改形状 —— 它们
    虽然"没有参数"，但**确实产生了新的一组值**，不能当成就地变换。
    如果当成逐元素层，前一层显示的宽度就会错成它的输出宽度，边也会对不上。

    这里在拿到真实张量之后做判断：形状没变才是就地变换。
    被提升的层没有可画的边（adapter 不认识它），所以只画节点、不画连线 ——
    这比硬画一批错位的连线诚实。
    """
    for record in records:
        if record.role is not Role.ELEMENTWISE:
            continue
        if record.input.dim() < 1 or record.output.dim() < 1:
            continue
        if record.input.shape[-1] != record.output.shape[-1]:
            record.role = Role.WEIGHT


def _build_edges(
    payloads: list[EdgePayload] | None,
    out_width: int,
    in_width: int,
    src_map: list[int] | None = None,
) -> list[Edge]:
    """把 adapter 给的扁平权重列表铺成边。

    权重矩阵按 (out, in) 行优先排列，所以第 o 个输出、第 i 个输入的权重在
    `payloads[o * in_width + i]`。

    `src_map` 把"模型输入列下标"翻译成"显示列里的下标"。只有第一层需要：
    输入列画的是全部特征（供点击开关），而模型只吃其中选中的那几个，
    两者下标一般对不上。
    """
    if not payloads:
        return []
    edges: list[Edge] = []
    for o in range(out_width):
        for i in range(in_width):
            idx = o * in_width + i
            if idx >= len(payloads):
                break  # 这个输出神经元的权重已经用完了
            src = src_map[i] if src_map is not None and i < len(src_map) else i
            edges.append(Edge(src=src, dst=o, payload=payloads[idx]))
    return edges


def _input_column(labels: list[str], active: list[int] | None) -> Column:
    """输入列：**全部**特征都画，未选中的标成 inactive。

    之所以不把未选中的直接藏起来，是因为原版的交互就是"点最左列的方块开关特征"——
    藏起来就没得点了。
    """
    active_set = set(active) if active is not None else None
    return Column(
        title="FEATURES",
        kind="input",
        nodes=[
            Node(index=i, label=label, column=0, active=(active_set is None or i in active_set))
            for i, label in enumerate(labels)
        ],
    )


def _node_label(kind: str, index: int) -> str:
    # 输出节点标成 y —— 30px 的方块放不下 "output"，而 y 短且数学上说得通
    # （它就是模型预测的那个量）。分类和回归都是这一个输出。
    return "y" if kind == "output" else str(index + 1)


def _range_of(tensor: torch.Tensor) -> tuple[float, float]:
    with torch.no_grad():
        return (float(tensor.min()), float(tensor.max()))


def _degenerate_graph(
    x: torch.Tensor,
    input_labels: list[str],
    active_inputs: list[int] | None = None,
    input_grid: torch.Tensor | None = None,
) -> DisplayGraph:
    """兜底图：只画输入列。用于模型没有可识别的权重层的情况。"""
    display = input_grid if input_grid is not None else x
    return DisplayGraph(
        columns=[_input_column(input_labels, active_inputs)],
        edges=[],
        activations=[display.detach()],
        input_ranges=[_range_of(display)],
    )
