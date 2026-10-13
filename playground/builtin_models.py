"""内置模型 —— 复刻原版行为的那个网络放在这里。

单独成文件是有意的：你在 `models.py` 里写模型时哪怕保存到一半（有语法错误），
这个文件里的模型仍然可用，页面不会空掉。

想加自己的模型请改 `models.py`，不用动这里。
"""

from __future__ import annotations

import torch.nn as nn

from .config import make_activation
from .layers import KANLinear
from .registry import register


def _init_like_playground(module: nn.Module) -> None:
    """按原版的初始化方式初始化权重。

    原版对**所有层用同一个分布** `U(-0.5, 0.5)`，没有 fan-in 缩放、
    没有 Xavier/He，bias 统一 0.1（不是 0）。

    这里照抄不是为了"更好"——恰恰相反，fan-in 缩放是有道理的，
    而原版的均匀初始化在小网络上够用。照抄是为了让复刻版的实验结论
    和原版可比。它本身也是个教学点：初始化方式会随网络规模变化而失效。
    """
    if isinstance(module, nn.Linear):
        nn.init.uniform_(module.weight, -0.5, 0.5)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.1)


@register(
    "mlp",
    title="MLP",
    description="全连接网络，形状由前端的加/减层按钮控制。复刻原版行为。",
    configurable_shape=True,
)
class MLP(nn.Module):
    """标准多层感知机。

    形状来自 `cfg.network_shape`（每个隐藏层的神经元数），
    所以前端的加层/加神经元按钮对它有效。空元组 = 没有隐藏层（纯线性模型）。
    """

    def __init__(self, in_dim: int, out_dim: int, cfg) -> None:
        super().__init__()
        shape = list(cfg.network_shape)
        dims = [in_dim, *shape, out_dim]

        layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            if i < len(dims) - 2:
                layers.append(make_activation(cfg.activation))
            else:
                # 输出层：回归用线性，分类用 tanh。
                # 分类用 tanh 是因为标签是 ±1，tanh 的值域 (-1,1) 正好对齐，
                # 于是输出值可以直接当热力图的颜色用。
                layers.append(make_activation(cfg.output_activation))

        self.net = nn.Sequential(*layers)
        self.apply(_init_like_playground)

    def forward(self, x):
        return self.net(x)


@register(
    "kan",
    title="KAN",
    description=(
        "Kolmogorov-Arnold 网络：非线性放在**边**上而不是节点上，"
        "每条边是一个可学习的一元函数，图上会画成曲线。"
        "训练前会自动把 spline 网格对齐到数据分布。"
    ),
    configurable_shape=True,
    # KAN 比 MLP 对学习率敏感得多（最优点在 1e-4 量级），
    # 而且纯 SGD 在这个尺度上很难推进，所以选中时自动切到 Adam + 小学习率。
    defaults={"learningRate": 0.001, "optimizer": "adam"},
)
class KAN(nn.Module):
    """KAN，形状同样由前端的加/减层按钮控制。

    和 MLP 的唯一结构差异就是 `KANLinear` 取代了 `Linear + 激活`。
    输出层仍然接一个激活（分类是 tanh），和 MLP 保持一致 ——
    否则两者的输出值域不同，决策边界和 loss 就没法直接对比了。
    """

    def __init__(self, in_dim: int, out_dim: int, cfg) -> None:
        super().__init__()
        shape = list(cfg.network_shape)
        dims = [in_dim, *shape, out_dim]

        # grid_range 只是初始值：引擎在训练开始前会调用 update_grid，
        # 按真实数据的分位数重建网格。之所以给个宽一点的默认值，
        # 是因为 Playground 的原始特征在 [-6,6] 上 —— 用 KAN 论文里
        # 那个 [-1,1] 的默认值，第一个前向就会有一半数据落在网格外，
        # spline 分支直接失效。
        self.layers = nn.ModuleList(
            KANLinear(dims[i], dims[i + 1], grid_size=5, spline_order=3, grid_range=(-6.0, 6.0))
            for i in range(len(dims) - 1)
        )
        self.output_activation = make_activation(cfg.output_activation)

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i == len(self.layers) - 1:
                x = self.output_activation(x)
        return x
