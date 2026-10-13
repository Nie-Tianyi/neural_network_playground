"""KAN 层与它的 adapter。

这个文件里最重要的两组断言，对应两个"静默失效"的坑 —— 它们都不会报错，
只会让你看到一个行为完全不对、但看起来很正常的网络：

1. **spline 网格外基函数恒为 0**：数据跑出网格时 φ 退化成 silu，
   你看到的是个"假 KAN"，怎么训练曲线都不变形。
2. **update_grid 会衰减 spline 分支**：上游参考实现的 bug，
   每调用一次就把 spline 输出乘上一个 spline_scaler。几十次之后 spline 分支
   就没了，同样表现为"退化成 silu"。
"""

from __future__ import annotations

import pytest
import torch

from playground.adapters.base import (
    DescribeContext,
    FunctionCurve,
    Role,
    default_adapter_registry,
)
from playground.adapters.kan import KANLinearAdapter
from playground.config import Config
from playground.engine import Engine
from playground.layers import KANLinear
from playground.registry import default_registry


# --- 层本身 ------------------------------------------------------------------


def test_forward_shapes() -> None:
    layer = KANLinear(3, 5, grid_size=5, spline_order=3)
    assert tuple(layer(torch.randn(7, 3)).shape) == (7, 5)
    assert tuple(layer.grid.shape) == (3, 5 + 2 * 3 + 1)  # (in, G+2k+1)
    assert tuple(layer.spline_weight.shape) == (5, 3, 5 + 3)  # (out, in, G+k)


def test_spline_branch_is_dead_outside_the_grid() -> None:
    """⚠️ 网格外所有 B-spline 基函数恒为 0。

    这正是"假 KAN"的成因：数据落在网格外时 spline 分支整个失效，
    φ 只剩 base 分支（silu），而图像上完全看不出来。
    所以引擎必须在训练前调用 update_grid 把网格对齐到数据。
    """
    layer = KANLinear(1, 1, grid_size=5, spline_order=3, grid_range=(-1.0, 1.0))
    outside = torch.tensor([[10.0], [-7.0], [3.5]])
    assert float(layer.b_splines(outside).abs().max()) == 0.0


def test_initial_phi_is_close_to_silu() -> None:
    """初始化时 spline 系数很小，所以 φ ≈ w_b·silu(x)。

    这不是巧合而是设计：于是"训练 = 曲线从 silu 逐渐变形出结构"，
    本身就是一段很好的教学动画。
    """
    torch.manual_seed(0)
    layer = KANLinear(2, 3, grid_size=5, spline_order=3, grid_range=(-3.0, 3.0))
    explanation = layer.explain(samples=16)
    assert float(explanation.spline.abs().max()) < 0.05
    assert float(explanation.phi.abs().max()) > 0.0


def test_update_grid_preserves_the_function() -> None:
    """`update_grid` 的全部意义是"换个网格表示、函数不变"。

    上游参考实现里这条不变量是**不成立**的：它用乘过 spline_scaler 的曲线
    去拟合没乘 scaler 的系数，而前向又会再乘一次 scaler，于是每调用一次
    spline 分支就被整体乘上一个 |scaler| < 1。实测 40 次之后 spline_weight
    从 1e-8 掉到 1e-24，spline 分支彻底死掉。
    """
    torch.manual_seed(0)
    layer = KANLinear(2, 3, grid_size=5, spline_order=3, grid_range=(-3.0, 3.0))
    x = torch.rand(200, 2) * 6 - 3

    with torch.no_grad():
        before = layer(x).clone()
    for _ in range(50):
        layer.update_grid(x)
    with torch.no_grad():
        after = layer(x)

    assert torch.allclose(before, after, atol=1e-3), "update_grid 改变了函数本身"


def test_update_grid_does_not_decay_spline_weight() -> None:
    """上面那条不变量的直接可观测后果：系数不能被反复更新给吃掉。"""
    torch.manual_seed(0)
    layer = KANLinear(1, 1, grid_size=5, spline_order=3, grid_range=(-3.0, 3.0))
    x = torch.linspace(-3, 3, 200).unsqueeze(-1)

    magnitude = float(layer.spline_weight.detach().abs().mean())
    for _ in range(30):
        layer.update_grid(x)
    after = float(layer.spline_weight.detach().abs().mean())

    # 系数会随重拟合小幅变化，但不该按数量级衰减
    assert after > magnitude * 0.1, f"spline_weight 衰减了：{magnitude:.2e} → {after:.2e}"


def test_update_grid_moves_the_grid_to_the_data() -> None:
    """网格要跟着数据走 —— 这是它存在的理由。"""
    layer = KANLinear(1, 1, grid_size=8, spline_order=3, grid_range=(-1.0, 1.0))
    x = torch.linspace(20, 40, 500).unsqueeze(-1)
    layer.update_grid(x)

    lo, hi = float(layer.grid.min()), float(layer.grid.max())
    assert lo < 20 and hi > 40, f"网格没有覆盖数据：{lo:.1f}..{hi:.1f}"


def test_kan_can_fit_a_function() -> None:
    """能学会拟合才有资格被叫做一个网络。"""
    torch.manual_seed(0)
    layer = KANLinear(1, 1, grid_size=8, spline_order=3, grid_range=(-3.0, 3.0))
    x = torch.linspace(-3, 3, 400).unsqueeze(-1)
    target = torch.sin(x)

    optimizer = torch.optim.Adam(layer.parameters(), lr=1e-2)
    for step in range(1500):
        optimizer.zero_grad()
        loss = ((layer(x) - target) ** 2).mean()
        loss.backward()
        optimizer.step()
        if step % 20 == 0:
            layer.update_grid(x)

    assert float(loss.detach()) < 0.01, f"拟合 sin 失败，MSE={float(loss.detach()):.4f}"


def test_explain_returns_one_curve_per_edge() -> None:
    layer = KANLinear(3, 4, grid_size=5, spline_order=3, grid_range=(-2.0, 2.0))
    explanation = layer.explain(samples=24)

    assert tuple(explanation.phi.shape) == (4, 3, 24)  # (out, in, samples)
    assert tuple(explanation.xs.shape) == (3, 24)  # 每个输入特征一条定义域
    assert tuple(explanation.importance.shape) == (4, 3)


def test_explain_domain_is_per_input_feature() -> None:
    """每个输入特征的定义域是独立的 —— 它们的数据范围本来就不一样。"""
    layer = KANLinear(2, 1, grid_size=5, spline_order=3, grid_range=(-1.0, 1.0))
    # 两个输入的量级差 100 倍
    x = torch.stack([torch.rand(300) * 2 - 1, torch.rand(300) * 200 - 100], dim=-1)
    layer.update_grid(x)

    xs = layer.explain(samples=8).xs
    span0 = float(xs[0].max() - xs[0].min())
    span1 = float(xs[1].max() - xs[1].min())
    assert span1 > span0 * 10, "两个输入特征的定义域没有分开"


# --- adapter -----------------------------------------------------------------


def test_adapter_produces_curve_payloads() -> None:
    adapter = KANLinearAdapter()
    layer = KANLinear(2, 3, grid_size=5, spline_order=3, grid_range=(-2.0, 2.0))

    assert adapter.matches(layer)
    assert adapter.role(layer) is Role.WEIGHT

    view = adapter.describe(layer, DescribeContext(input_range=(-2.0, 2.0)))
    assert view.kind == "kan"
    assert view.width == 3
    assert len(view.edges) == 2 * 3  # 行优先 (out, in)

    edge = view.edges[0]
    assert isinstance(edge, FunctionCurve)
    assert len(edge.xs) == len(edge.ys) == adapter.SAMPLES
    assert edge.base is not None and edge.spline is not None
    # 两个分支加起来等于总量：φ = w_b·silu + w_s·spline
    assert edge.ys[0] == pytest.approx(edge.base[0] + edge.spline[0], abs=1e-5)


def test_adapter_takes_priority_over_the_fallback() -> None:
    """KANLinear 也有 weight 参数，兜底 adapter 会把它当权重矩阵画 —— 那是完全
    错误的信息（spline_weight 是三维的，不是权重矩阵）。所以它必须优先级更高。"""
    adapters = default_adapter_registry()
    layer = KANLinear(2, 2, grid_size=5, spline_order=3)

    assert adapters.resolve(layer) is not None
    assert adapters.resolve(layer).name == "kan"
    assert adapters.recognizes(layer)


def test_kan_layer_is_hooked_even_though_it_has_children() -> None:
    """KANLinear 内部有子模块（base_activation），但它自己才是有意义的层。

    挑选 hook 目标的规则如果写成"只挂叶子模块"，KANLinear 会被当成容器跳过，
    只记录到它内部的 SiLU —— 整个 KAN 层在图上消失。判据必须是"认不认识"。
    """
    from playground.introspection import _select_hook_targets

    model = torch.nn.Sequential(
        KANLinear(2, 3, grid_size=5, spline_order=3),
        KANLinear(3, 1, grid_size=5, spline_order=3),
    )
    targets = [name for name, _ in _select_hook_targets(model, default_adapter_registry())]
    assert targets == ["0", "1"]  # 两个 KAN 层自己，而不是它们内部的激活


# --- 接进引擎 ---------------------------------------------------------------


@pytest.fixture(scope="module")
def registry():
    return default_registry()


def test_builtin_kan_is_registered(registry) -> None:
    spec = registry.get("kan")
    assert spec is not None
    # 选中 KAN 时应该自动切到适合它的优化器和学习率
    assert spec.defaults["optimizer"] == "adam"
    assert spec.defaults["learningRate"] < Config().learning_rate


def test_engine_aligns_the_spline_grid_before_training(registry) -> None:
    """这是"假 KAN"的解药：训练还没开始，网格就得对上数据。

    circle 数据集在 [-5,5] 上，而 KANLinear 的默认网格是 [-1,1] ——
    不对齐的话第一次前向就有一半数据在网格外，spline 分支直接失效。
    """
    engine = Engine(registry)
    engine.configure(Config(model="kan", dataset="circle", features=("x", "y"), network_shape=(3,)))

    layer = engine.model.layers[0]
    assert float(layer.grid.min()) < -4.0
    assert float(layer.grid.max()) > 4.0


def test_engine_keeps_the_grid_aligned_during_training(registry) -> None:
    """激活会随训练漂移，网格得跟着走。"""
    engine = Engine(registry)
    engine.configure(
        Config(
            model="kan",
            dataset="spiral",
            features=("x", "y"),
            network_shape=(2,),
            grid_update_every=10,
        )
    )
    # 模拟"激活漂移出网格"：把网格整体挪到远离数据的地方
    engine.model.layers[0].grid.add_(500.0)
    for _ in range(10):
        engine.step()

    # 跑满一个 grid_update_every 之后必须被拉回数据范围附近
    assert float(engine.model.layers[0].grid.abs().max()) < 100


def test_grid_update_can_be_disabled(registry) -> None:
    engine = Engine(registry)
    engine.configure(
        Config(
            model="kan",
            dataset="circle",
            features=("x", "y"),
            network_shape=(2,),
            grid_update_every=0,
        )
    )
    engine.model.layers[0].grid.add_(500.0)
    for _ in range(30):
        engine.step()

    assert float(engine.model.layers[0].grid.abs().max()) > 400  # 没被动过


def test_kan_graph_has_curve_edges(registry) -> None:
    """端到端：引擎产出的图里，KAN 的边必须是曲线而不是标量。"""
    engine = Engine(registry)
    engine.configure(
        Config(model="kan", dataset="circle", features=("x", "y"), network_shape=(4,))
    )
    for _ in range(30):
        engine.step()
    graph, _ = engine.snapshot()

    assert [(c.kind, c.width) for c in graph.columns] == [
        ("input", 7),
        ("hidden", 4),
        ("output", 1),
    ]
    assert [len(layer) for layer in graph.edges] == [2 * 4, 4 * 1]

    first = graph.edges[0]
    assert all(edge.payload.kind == "curve" for edge in first)
    # 第一层的边必须从**已选中的**特征出发（那条老坑在这里也要成立）
    assert {edge.src for edge in first} == {0, 1}
    assert {edge.dst for edge in first} == {0, 1, 2, 3}


def test_kan_trains_on_every_dataset(registry) -> None:
    """KAN 在每个数据集上都要能学动。"""
    budgets = {"circle": 400, "xor": 400, "plane": 400, "spiral": 800}
    for dataset_id, epochs in budgets.items():
        engine = Engine(registry)
        engine.configure(
            Config(
                model="kan",
                dataset=dataset_id,
                features=("x", "y"),
                network_shape=(4,),
                learning_rate=0.003,
                optimizer="adam",
                seed=3.0,
            )
        )
        start = engine.losses.train
        for _ in range(epochs):
            engine.step()
        assert engine.losses.train < start * 0.5, (
            f"KAN 在 {dataset_id} 上没学动：{start:.4f} → {engine.losses.train:.4f}"
        )
