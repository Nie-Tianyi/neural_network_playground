"""内置 adapter：Linear 系、常见逐元素层、以及兜底。

兜底那条路径（OpaqueFallbackAdapter）是这个项目"灵活"承诺的兑现处：
用户写了一个我们从没见过的层时，不该白屏。策略是**先猜，猜不出就降级**——
有 2D 以上的 weight 参数就当成权重矩阵画；没有参数就当逐元素变换；
都不满足就只画节点热力图、不画边。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from .base import DescribeContext, LayerView, Role, ScalarWeight

# 逐元素层：不产生新列，只就地变换当前列每个节点的值。
# 这些层前后张量形状一致，所以可以直接把它们的输出当作"神经元的输出"。
_ELEMENTWISE_TYPES: tuple[type[nn.Module], ...] = (
    nn.ReLU,
    nn.ReLU6,
    nn.LeakyReLU,
    nn.ELU,
    nn.SELU,
    nn.GELU,
    nn.SiLU,
    nn.Mish,
    nn.Tanh,
    nn.Hardtanh,
    nn.Sigmoid,
    nn.Softplus,
    nn.Softsign,
    nn.Identity,
    nn.Dropout,
    nn.Dropout1d,
    nn.Dropout2d,
    nn.AlphaDropout,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.LayerNorm,
    nn.GroupNorm,
    nn.Flatten,
)


class LinearAdapter:
    """`nn.Linear` → 每条边一个标量权重。

    边上权重的排列是行优先的 (out, in)：`edges[o * in_features + i]` 是
    第 i 个输入到第 o 个输出的那条边。前端按这个索引直接取用。
    """

    name = "linear"
    priority = 10

    def matches(self, module: nn.Module) -> bool:
        return isinstance(module, nn.Linear)

    def role(self, module: nn.Module) -> Role:
        return Role.WEIGHT

    def describe(self, module: nn.Linear, ctx: DescribeContext) -> LayerView:
        weight = module.weight.detach().to("cpu")
        dead = ctx.dead_masks.get("weight")
        edges: list[ScalarWeight] = [
            ScalarWeight(value=float(weight[o, i]), dead=bool(dead[o, i]) if dead is not None else False)
            for o in range(module.out_features)
            for i in range(module.in_features)
        ]
        bias = (
            [float(b) for b in module.bias.detach().to("cpu")]
            if module.bias is not None
            else [0.0] * module.out_features
        )
        return LayerView(
            role=Role.WEIGHT,
            kind="linear",
            label="Linear",
            width=module.out_features,
            edges=edges,
            extra={"inFeatures": module.in_features, "bias": bias},
        )


class ElementwiseAdapter:
    """激活、Dropout、归一化 —— 就地变换，不产生新列。

    它们在图上不画任何东西，但**决定了节点上显示的数值**：
    Playground 里"神经元的输出"指激活之后的值，所以链到 Linear 后面的
    激活层，才是这一列热力图的取值来源。
    """

    name = "elementwise"
    priority = 5

    def matches(self, module: nn.Module) -> bool:
        return isinstance(module, _ELEMENTWISE_TYPES)

    def role(self, module: nn.Module) -> Role:
        return Role.ELEMENTWISE

    def describe(self, module: nn.Module, ctx: DescribeContext) -> LayerView:
        return LayerView(
            role=Role.ELEMENTWISE,
            kind="activation",
            label=type(module).__name__,
        )


class OpaqueFallbackAdapter:
    """兜底：任何认不出的模块都落到这里。

    priority 是最低的，只有在其他 adapter 都不匹配时才生效。
    三种猜测，从强到弱：

      1. 有 >=2 维的 `weight` 参数 → 当成权重矩阵画标量边。
         这能覆盖绝大多数"自己写了个类似 Linear 的层"的情况。
      2. 完全没有参数 → 当成逐元素变换（很可能是某种激活/归一化）。
      3. 有参数但不是权重矩阵的形状 → 只画节点，边留空。

    猜错了也不会崩，最多是画得不精确 —— 这正是这个兜底存在的意义。
    """

    name = "opaque"
    priority = 0
    is_fallback = True

    def matches(self, module: nn.Module) -> bool:
        return True

    def role(self, module: nn.Module) -> Role:
        # 兜底判角色只需要看参数形状，不用构造边，直接复用 describe 的结论即可
        return self.describe(module, DescribeContext()).role

    def describe(self, module: nn.Module, ctx: DescribeContext) -> LayerView:
        name = type(module).__name__
        weight = _find_matrix_weight(module)

        if weight is not None:
            out_features, in_features = weight.shape[0], weight.shape[1]
            flat = weight.reshape(out_features, in_features)
            edges = [
                ScalarWeight(value=float(flat[o, i]))
                for o in range(out_features)
                for i in range(in_features)
            ]
            return LayerView(
                role=Role.WEIGHT,
                kind="opaque-weight",
                label=name,
                width=out_features,
                edges=edges,
                extra={"inFeatures": in_features, "guessed": True},
            )

        has_params = any(True for _ in module.parameters(recurse=False))
        if not has_params:
            return LayerView(role=Role.ELEMENTWISE, kind="activation", label=name)

        # 有参数但形状不像权重矩阵：保留节点热力图，不画边。
        return LayerView(role=Role.WEIGHT, kind="opaque", label=name, width=None)


def _find_matrix_weight(module: nn.Module) -> torch.Tensor | None:
    """找出这个模块里最像"权重矩阵"的参数。

    优先看名为 `weight` 的，其次看第一个 >=2 维的参数。
    只取模块自己的直接参数（recurse=False）—— 递归进去会拿到子模块的权重，
    而子模块自己会被单独 hook，重复计数。
    """
    named = dict(module.named_parameters(recurse=False))
    weight = named.get("weight")
    if weight is not None and weight.dim() >= 2:
        return weight.detach().to("cpu")

    for param in named.values():
        if param.dim() >= 2:
            return param.detach().to("cpu")
    return None
