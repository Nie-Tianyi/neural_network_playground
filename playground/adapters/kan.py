"""KAN 层的 adapter —— 把「边是一个函数」这件事画出来。

## 为什么 KAN 值得单独做可视化

MLP 的一条边只有"多粗、什么颜色"两个信息量。KAN 的一条边是一个完整的
一元函数：你可以直接看到它长什么样、在定义域的哪一段陡、哪一段平。

整个网络 [2, h, 1] 也就 3h 条曲线，全部能同时画出来 —— 这在 MLP 上是做不到的。

## 编码方式（照参考实现，不要自创）

    **透明度** ∝ 归一化后的重要性 |φ|₁，**线宽固定**

参考实现的绘图也是这么做的（`alpha = tanh(3·score)`，线宽是常量）。
不要改成"用线宽表示重要性"—— 线宽要留给曲线本身的视觉重量，
两个信息挤在一个通道里会都看不清。
"""

from __future__ import annotations

from torch import nn

from ..layers import KANLinear
from .base import DescribeContext, FunctionCurve, LayerView, Role


class KANLinearAdapter:
    """`KANLinear` → 每条边一个 `FunctionCurve`。

    priority 取得比 Linear 高：KANLinear 也是 nn.Module、也有 weight 参数，
    兜底 adapter 会把它当成权重矩阵 —— 那样画出来的是完全错误的信息
    （KAN 的 spline_weight 是 (out, in, G+k) 的三维张量，不是权重矩阵）。
    """

    name = "kan"
    priority = 100

    # 每条曲线的采样点数。48 个点足够让三次样条看起来平滑，
    # 而一条边序列化后只占几百字节。
    SAMPLES = 48

    def matches(self, module: nn.Module) -> bool:
        return isinstance(module, KANLinear)

    def role(self, module: nn.Module) -> Role:
        return Role.WEIGHT

    def describe(self, module: KANLinear, ctx: DescribeContext) -> LayerView:
        explanation = module.explain(samples=self.SAMPLES)

        xs = explanation.xs.detach().to("cpu")
        phi = explanation.phi.detach().to("cpu")
        base = explanation.base.detach().to("cpu")
        spline = explanation.spline.detach().to("cpu")
        importance = explanation.importance.detach().to("cpu")

        edges: list[FunctionCurve] = []
        for out_index in range(module.out_features):
            for in_index in range(module.in_features):
                edges.append(
                    FunctionCurve(
                        xs=xs[in_index].tolist(),
                        ys=phi[out_index, in_index].tolist(),
                        # 拆开给出两个分支：UI 可以把 φ = w_b·silu + w_s·spline 画成
                        # 两条淡线 + 一条实线，直观看清各自贡献了什么。
                        base=base[out_index, in_index].tolist(),
                        spline=spline[out_index, in_index].tolist(),
                        importance=float(importance[out_index, in_index]),
                    )
                )

        grid_lo = float(module.grid.min())
        grid_hi = float(module.grid.max())
        return LayerView(
            role=Role.WEIGHT,
            kind="kan",
            label="KAN",
            width=module.out_features,
            edges=edges,
            extra={
                "inFeatures": module.in_features,
                # 网格范围 vs 输入范围。UI 用它们提示"数据跑出网格了"——
                # 那时 spline 分支恒为 0，φ 会静默退化成 silu，
                # 只看曲线是看不出来的（它会变得特别平滑）。
                "gridRange": [grid_lo, grid_hi],
                "inputRange": list(ctx.input_range) if ctx.input_range else None,
                "gridSize": module.grid_size,
                "splineOrder": module.spline_order,
            },
        )
