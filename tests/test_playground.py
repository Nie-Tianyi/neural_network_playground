"""无头测试 —— 不依赖浏览器就能验证引擎的行为。

前端渲染得对不对，光看后端"算出来的数好像没问题"是看不出来的
（这个项目里真正容易出错的恰恰是"画什么"），所以这些测试只覆盖
引擎侧的不变量；渲染是另外用无头浏览器截图核对的。

跑：uv run pytest
"""

from __future__ import annotations

import math

import pytest
import torch
from torch import nn

from playground.config import Config
from playground.datasets import DATASETS, FEATURES, generate
from playground.engine import Engine
from playground.introspection import build_graph
from playground.registry import ModelRegistry, default_registry


@pytest.fixture(scope="module")
def registry() -> ModelRegistry:
    return default_registry()


# --- 数据集 ------------------------------------------------------------------


@pytest.mark.parametrize("dataset_id", list(DATASETS))
def test_dataset_structural_invariants(dataset_id: str) -> None:
    """每个数据集的点数、值域、标签域都要落在原版规格内。

    这里查的是**结构性质**而不是逐点比对：原版用的是 JS 的 Math.random +
    seedrandom，和 Python 的 random 不可能产生同一个随机流，
    所以"同种子逐点相等"这个目标本身就不成立。能验证的是采样范围、
    各类点数量、标签取值这些由代码逻辑（而非随机数）决定的性质。
    """
    spec = DATASETS[dataset_id]
    split = generate(dataset_id, noise=0.0, perc_train=50.0, seed=42)
    points = split.train + split.test

    assert len(points) == spec.num_samples

    if spec.is_1d:
        # 一维拟合：y 恒为 0（不参与任何计算），标签范围由目标函数决定，
        # 所以这里只查"落在定义域内"和"确实是那个函数的值"。
        for p in points:
            assert spec.domain[0] <= p.x <= spec.domain[1]
            assert p.y == 0.0
            assert p.label == pytest.approx(spec.truth(p.x), abs=1e-9)
        return

    if spec.problem == "classification":
        labels = {p.label for p in points}
        assert labels <= {1.0, -1.0}
        # 两类都要有，否则数据生成器写反了
        assert labels == {1.0, -1.0}
        assert len(split.train) == len(split.test)  # 50/50 且总数是偶数
    else:
        # 回归标签理论上在 [-1,1]，但 plane 是 `(x+y)/10` 而 x,y ∈ [-6,6]，
        # 所以极值能到 ±1.2 —— 原版也没做 clamp，这里照旧。
        for p in points:
            assert -1.25 <= p.label <= 1.25

    # 分类数据在 [-5,5] 上采样（XOR 加了 padding 后最多到 5.3），
    # 回归数据在 [-6,6] 上
    limit = 5.4 if dataset_id == "xor" else 6.0
    for p in points:
        assert -limit <= p.x <= limit
        assert -limit <= p.y <= limit


def test_xor_labels_match_quadrants() -> None:
    """XOR 的标签必须等于 sign(x*y)（padding 只影响坐标，不影响符号）。"""
    split = generate("xor", noise=0.0, perc_train=100.0, seed=7)
    for p in split.train:
        assert p.label == (1.0 if p.x * p.y >= 0 else -1.0)


def test_circle_inner_is_positive() -> None:
    """圆环数据集里，内圈的点是正类。"""
    split = generate("circle", noise=0.0, perc_train=100.0, seed=1)
    for p in split.train:
        inside = math.hypot(p.x, p.y) < 2.5
        assert (p.label == 1.0) == inside


def test_train_test_split_is_disjoint() -> None:
    split = generate("spiral", noise=0.0, perc_train=40.0, seed=3)
    assert len(split.train) == 200
    assert len(split.test) == 300
    combined = {(p.x, p.y, p.label) for p in split.train + split.test}
    assert len(combined) == 500  # 没有点被漏掉或重复


def test_grid_is_row_major_top_down() -> None:
    """网格点顺序决定热力图的方向 —— 反了的话整张决策边界就是上下颠倒的。"""
    from playground.datasets import DENSITY, grid_points

    grid = grid_points(DENSITY)
    assert grid[0] == (-6.0, 6.0)  # 第一个点在左上
    assert grid[DENSITY - 1] == (6.0, 6.0)  # 第一行最后一个在右上
    assert grid[-1] == (6.0, -6.0)  # 最后一个点在右下


# --- 特征 --------------------------------------------------------------------


@pytest.mark.parametrize("feature_id", list(FEATURES))
def test_feature_scalar_and_tensor_agree(feature_id: str) -> None:
    """标量实现和向量实现必须给出同样的值。

    这两份实现是刻意分开的（一份用来逐点理解，一份用来批量算网格），
    分开写就有写岔的风险 —— 这个测试就是防这个的。
    """
    from playground.datasets import FEATURE_TORCH

    feature = FEATURES[feature_id]
    xs = torch.tensor([-2.5, 0.0, 1.7, 4.2])
    ys = torch.tensor([3.1, -1.4, 0.0, 2.2])
    expected = torch.tensor([feature.fn(float(x), float(y)) for x, y in zip(xs, ys, strict=True)])
    assert torch.allclose(FEATURE_TORCH[feature_id](xs, ys), expected, atol=1e-6)


# --- 训练 --------------------------------------------------------------------


# 每个数据集需要多少 epoch 才收敛差别很大。spiral 尤其典型：前 500 个 epoch
# 几乎纹丝不动（0.535 → 0.453），然后突然跌到 0.03 —— 纯 SGD 的长时间平台期。
# 这不是 bug，给它足够的时间就行；用小 epoch 数去断言反而会得出错误结论。
_CONVERGENCE_BUDGET = {
    "circle": 150,
    "xor": 150,
    "gauss": 150,
    "plane": 150,
    "multi-gauss": 400,
    "spiral": 1500,
    # 一维拟合里**振荡函数才是最难的**，实测（8×8 tanh，默认 SGD）：
    #     x³/9      2 个 epoch 就到起始 loss 的 20%
    #     |x|      16 个
    #     sin(3x)  ~410 个
    #     exp(sin3x) ~440 个
    # 直觉上会觉得 |x| 在 0 处不可导、应该最难，但它只是两段单调曲线拼起来的；
    # 而 sin(3x) 要求在定义域上拟合出多个完整振荡。给两倍的余量。
    "fit-sin": 900,
    "fit-exp-sin": 900,
    "fit-cubic": 200,
    "fit-abs": 200,
}


@pytest.mark.parametrize("dataset_id", list(DATASETS))
def test_loss_decreases(dataset_id: str, registry: ModelRegistry) -> None:
    """每个数据集上，训练都要让 loss 显著下降。"""
    engine = Engine(registry)
    engine.configure(
        Config(dataset=dataset_id, network_shape=(8, 8), seed=5.0)
    )
    start = engine.losses.train
    for _ in range(_CONVERGENCE_BUDGET[dataset_id]):
        engine.step()

    assert engine.iter == _CONVERGENCE_BUDGET[dataset_id]
    # 阈值放宽到 20%：这里要抓的是"完全学不动"，不是"收敛得好不好"
    assert engine.losses.train < start * 0.2, (
        f"{dataset_id} 训练 {_CONVERGENCE_BUDGET[dataset_id]} 个 epoch 后几乎没进展"
        f"（{start:.4f} → {engine.losses.train:.4f}）"
    )


def test_regression_uses_linear_output(registry: ModelRegistry) -> None:
    """回归任务的输出不能被 tanh 夹到 (-1,1) 之外进不去 —— 它是线性的。

    plane 数据集的标签约在 ±1.2，如果输出层误用了 tanh，模型永远拟合不到位。
    """
    assert Config(dataset="plane").output_activation == "linear"
    assert Config(dataset="circle").output_activation == "tanh"


def test_training_is_reproducible_with_same_seed(registry: ModelRegistry) -> None:
    """同样的配置跑两次要得到同样的曲线 —— 否则做不了对照实验。

    光给数据播种是不够的：权重初始化也要播种。这一点比原版多做了一步
    （原版权重用未播种的 Math.random()，所以同配置重跑曲线不一样）。
    """

    def run() -> float:
        engine = Engine(registry)
        engine.configure(Config(dataset="spiral", features=("x", "y"), seed=11.0))
        for _ in range(30):
            engine.step()
        return engine.losses.train

    assert run() == pytest.approx(run(), abs=1e-9)


def test_different_seed_gives_different_run(registry: ModelRegistry) -> None:
    """换种子必须换结果 —— 否则"可复现"其实是"根本没在随机"。"""

    def run(seed: float) -> float:
        engine = Engine(registry)
        engine.configure(Config(dataset="spiral", features=("x", "y"), seed=seed))
        for _ in range(30):
            engine.step()
        return engine.losses.train

    assert run(11.0) != pytest.approx(run(12.0))


def test_reset_gives_a_fresh_initialization(registry: ModelRegistry) -> None:
    """Reset 按钮必须换一套新的随机权重，否则这个按钮没有意义。"""
    engine = Engine(registry)
    engine.configure(Config(features=("x", "y")))
    before = [p.detach().clone() for p in engine.model.parameters()]

    engine.reset()
    after = [p.detach().clone() for p in engine.model.parameters()]

    assert any(not torch.equal(a, b) for a, b in zip(before, after, strict=True))
    assert engine.iter == 0


def test_batch_size_does_not_break_tail(registry: ModelRegistry) -> None:
    """batch 不能整除样本数时尾部要被安全丢弃，而不是崩掉或漏算。"""
    for batch_size in (1, 7, 30, 250):
        engine = Engine(registry)
        engine.configure(Config(dataset="circle", features=("x", "y"), batch_size=batch_size))
        engine.step()
        assert engine.iter == 1


# --- L1 剪枝 -----------------------------------------------------------------


def test_l1_prunes_more_aggressively_as_rate_grows(registry: ModelRegistry) -> None:
    """L1 正则率越大，被永久冻结的连接越多。

    这是原版"连接一条条消失"那种观感的来源，也是这个规则值得照抄的原因。
    """

    def dead_count(rate: float) -> int:
        engine = Engine(registry)
        engine.configure(
            Config(
                dataset="spiral",
                features=("x", "y"),
                network_shape=(8,),
                regularization="l1",
                regularization_rate=rate,
            )
        )
        for _ in range(120):
            engine.step()
        graph, _ = engine.snapshot()
        return sum(1 for layer in graph.edges for edge in layer if edge.payload.dead)

    assert dead_count(0.0) == 0
    assert dead_count(0.003) > 0
    assert dead_count(0.3) > dead_count(0.003)


def test_l2_does_not_kill_links(registry: ModelRegistry) -> None:
    """L1 才有"过零即杀死"规则，L2 只做收缩 —— 拉到极端也不能把网络剪光。"""
    engine = Engine(registry)
    engine.configure(
        Config(
            dataset="circle",
            features=("x", "y"),
            network_shape=(8,),
            regularization="l2",
            regularization_rate=3.0,
        )
    )
    for _ in range(60):
        engine.step()
    graph, _ = engine.snapshot()
    assert not any(edge.payload.dead for layer in graph.edges for edge in layer)


# --- 内省：图结构 ------------------------------------------------------------


def test_graph_structure_matches_architecture(registry: ModelRegistry) -> None:
    """列宽、边数必须精确对应网络的形状。

    边数对不上是最隐蔽的一类 bug：图还是画得出来，只是连错了。
    """
    engine = Engine(registry)
    engine.configure(Config(features=("x", "y"), network_shape=(5, 3)))
    graph, payload = engine.snapshot()

    assert [(c.kind, c.width) for c in graph.columns] == [
        ("input", 7),  # 输入列画**全部** 7 个特征，未选中的也在（可点击开关）
        ("hidden", 5),
        ("hidden", 3),
        ("output", 1),
    ]
    # 边数按**实际喂给模型的**特征数算（2 个），不是输入列的宽度
    assert [len(layer) for layer in graph.edges] == [2 * 5, 5 * 3, 3 * 1]

    # 每个隐层节点的热力图是 10×10，输出层是全分辨率的 100×100
    assert [tuple(h.shape) for h in payload["heatmaps"]] == [
        (7, 10, 10),
        (5, 10, 10),
        (3, 10, 10),
        (1, 100, 100),
    ]


def test_every_hidden_neuron_receives_edges_from_all_selected_features(
    registry: ModelRegistry,
) -> None:
    """每个隐藏神经元都要从**每一个**已选特征各收到一条边。

    只数边数是不够的 —— 曾经有个 bug 让总边数恰好还是对的，但边全部落错：
    前两个神经元各收了 7 条和 3 条，后三个一条都没有，源点还指向了没选中的特征。
    所以这里查的是**边的着落点**，不是数量。
    """
    engine = Engine(registry)
    engine.configure(Config(features=("x", "sinY"), network_shape=(5,)))
    graph, _ = engine.snapshot()

    first = graph.edges[0]
    expected = {(src, dst) for dst in range(5) for src in (0, 6)}  # X_1 和 sin(X_2)
    assert {(e.src, e.dst) for e in first} == expected


def test_first_layer_edges_point_at_the_active_nodes(registry: ModelRegistry) -> None:
    """只选 X₁X₂ 时，权重必须落在显示列的第 5 个方块上（下标 4），
    而不是"模型输入的第 0 列 = 显示列的第 0 个"。"""
    engine = Engine(registry)
    engine.configure(Config(features=("xTimesY",), network_shape=(5,)))
    graph, _ = engine.snapshot()

    first = graph.edges[0]
    assert {e.src for e in first} == {4}
    assert sorted(e.dst for e in first) == [0, 1, 2, 3, 4]


def test_edges_follow_feature_order_not_display_order(registry: ModelRegistry) -> None:
    """特征的**选择顺序**决定模型输入列的顺序，未必和显示列的顺序一致。

    在页面上关掉 X₁ 再打开，它会被追加到列表末尾，于是模型的第一列变成 X₂。
    如果铺边时假设"模型第 k 列 = 显示列第 k 个方块"，
    每一条边的权重都会挂到错误的特征上 —— 而且边数和连线位置看起来都正常，
    只有权重值是错的，肉眼极难发现。
    """
    engine = Engine(registry)
    # 模型输入顺序是 (X₂, X₁)，而显示列里 X₁ 在前
    engine.configure(Config(features=("y", "x"), network_shape=(3,)))
    graph, _ = engine.snapshot()

    first = graph.edges[0]
    # 显示下标：X_1 → 0，X_2 → 1。模型第 0 列是 y，所以要映射到显示下标 1。
    assert sorted((e.src, e.dst) for e in first) == sorted(
        (src, dst) for dst in range(3) for src in (0, 1)
    )
    assert len(first) == 6

    # 真正要盯的是"权重有没有跟着特征走"：模型第一列是 y，
    # 那么 dst=0 上 src=显示下标1（也就是 X_2）的那条边，
    # 应该拿到 linear.weight[0][0]
    head = engine.model.net[0]
    weight = head.weight.detach()
    for edge in first:
        expected = float(weight[edge.dst][0 if edge.src == 1 else 1])
        assert edge.payload.value == pytest.approx(expected, abs=1e-6)


def test_hidden_column_edges_are_fully_connected(registry: ModelRegistry) -> None:
    """从第二层起，模型列宽和显示列宽一致，必须全连接。

    如果第一层的下标映射写错，错误会顺着 model_widths 传下去，这里就会露馅。
    """
    engine = Engine(registry)
    engine.configure(Config(features=("x", "y", "xSquared"), network_shape=(4, 3, 2)))
    graph, _ = engine.snapshot()

    for layer_index, (in_width, out_width) in enumerate([(3, 4), (4, 3), (3, 2)]):
        edges = graph.edges[layer_index]
        assert len(edges) == in_width * out_width, f"第 {layer_index} 层边数不对"
        assert {(e.src, e.dst) for e in edges} == {
            (src, dst) for dst in range(out_width) for src in range(in_width)
        }


def test_input_column_marks_active_features(registry: ModelRegistry) -> None:
    """输入列要能分辨哪些特征真的参与了训练 —— 这是"点方块开关特征"的基础。"""
    engine = Engine(registry)
    engine.configure(Config(features=("x", "xTimesY")))
    graph, _ = engine.snapshot()

    column = graph.columns[0]
    assert column.kind == "input"
    assert [n.label for n in column.nodes] == [
        "X_1", "X_2", "X_1^2", "X_2^2", "X_1X_2", "sin(X_1)", "sin(X_2)",
    ]
    # 只有 X_1 和 X_1X_2 是打开的
    assert [n.active for n in column.nodes] == [True, False, False, False, True, False, False]


def test_node_activation_comes_after_the_activation_function(registry: ModelRegistry) -> None:
    """节点上显示的必须是**激活之后**的值，不是加权和。

    做法：构造一个输出层权重全是 0、bias 是 0 的网络。此时加权和恒为 0，
    而 tanh(0)=0、relu(0)=0 —— 两者分不开。所以换成一个 bias 很大的网络：
    加权和约为大正数，tanh 会把它压到接近 1。
    """
    engine = Engine(registry)
    engine.configure(Config(features=("x", "y"), activation="tanh", network_shape=(2,)))
    with torch.no_grad():
        head = engine.model.net[0]
        head.bias.fill_(5.0)
        head.weight.zero_()

    graph, _ = engine.snapshot()
    hidden = graph.activations[1]
    # 加权和是 5.0；如果显示的是加权和，这里会看到 5
    assert float(hidden.max()) < 1.0
    assert float(hidden.min()) > 0.99  # tanh(5) ≈ 0.9999


def test_unknown_module_degrades_instead_of_crashing() -> None:
    """用户写了一个我们从没见过的层时，图要退化而不是白屏。"""

    class Mystery(nn.Module):
        def __init__(self, in_dim, out_dim, cfg):
            super().__init__()
            self.pre = nn.Linear(in_dim, 4)
            self.weird = nn.GLU()  # 不认识的层，而且会把 4 → 2
            self.post = nn.Linear(2, out_dim)

        def forward(self, x):
            return self.post(self.weird(self.pre(x)))

    registry = default_registry()
    engine = Engine(registry)
    engine.configure(Config(features=("x", "y")))
    engine.model = Mystery(2, 1, engine.cfg)

    graph, _ = engine.snapshot()
    # 降维层必须自成一列，否则它前面那一列的宽度会算错
    assert [c.width for c in graph.columns] == [7, 4, 2, 1]
    # 认不出的层没有可画的边 —— 宁可不画，也不画一批错位的
    assert [len(layer) for layer in graph.edges] == [8, 0, 2]


def test_branching_model_does_not_crash() -> None:
    """有分支（残差）的模型不追求画得精确，但不能崩。"""

    class Residual(nn.Module):
        def __init__(self, in_dim, out_dim, cfg):
            super().__init__()
            self.stem = nn.Linear(in_dim, 4)
            self.block = nn.Linear(4, 4)
            self.act = nn.Tanh()
            self.head = nn.Linear(4, out_dim)

        def forward(self, x):
            h = self.act(self.stem(x))
            return self.head(h + self.act(self.block(h)))

    engine = Engine(default_registry())
    engine.configure(Config(features=("x", "y")))
    engine.model = Residual(2, 1, engine.cfg)

    graph, payload = engine.snapshot()
    assert graph.columns[-1].width == 1
    assert len(payload["heatmaps"]) == len(graph.columns)


def test_model_without_weight_layers_degrades() -> None:
    """完全没有权重层的模型（比如恒等映射）不该让引擎崩。"""

    class Identity(nn.Module):
        def forward(self, x):
            return x

    engine = Engine(default_registry())
    engine.configure(Config(features=("x", "y")))
    engine.model = Identity()

    graph, _ = engine.snapshot()
    assert len(graph.columns) == 1  # 只剩输入列
    assert graph.columns[0].width == 7  # 输入列仍然画全部特征
    assert sum(n.active for n in graph.columns[0].nodes) == 2  # x, y 是打开的


# --- 配置变更语义 ------------------------------------------------------------


def test_learning_rate_change_keeps_progress(registry: ModelRegistry) -> None:
    """调学习率不该把你训到一半的结果清掉。"""
    engine = Engine(registry)
    engine.configure(Config(features=("x", "y"), learning_rate=0.03))
    for _ in range(20):
        engine.step()
    assert engine.iter == 20

    engine.configure(engine.cfg.replace(learning_rate=0.1))
    assert engine.iter == 20  # 迭代数保留


def test_dataset_change_regenerates(registry: ModelRegistry) -> None:
    engine = Engine(registry)
    engine.configure(Config(dataset="circle", features=("x", "y")))
    for _ in range(20):
        engine.step()

    engine.configure(engine.cfg.replace(dataset="spiral"))
    assert engine.iter == 0  # 换数据集必须重来


def test_feature_toggle_changes_input_width(registry: ModelRegistry) -> None:
    """开关特征改变的是**喂给模型的输入宽度**，输入列本身永远是 7 个方块。"""
    engine = Engine(registry)
    engine.configure(Config(features=("x", "y"), network_shape=(4,)))
    assert engine.snapshot()[0].columns[0].width == 7

    engine.configure(engine.cfg.replace(features=("x", "y", "xTimesY", "sinX")))
    graph, _ = engine.snapshot()
    assert graph.columns[0].width == 7  # 列宽不变
    assert sum(n.active for n in graph.columns[0].nodes) == 4  # 4 个被打开
    assert len(graph.edges[0]) == 4 * 4  # 边数按打开的算


# --- 序列化 ------------------------------------------------------------------


def test_serialization_roundtrip(registry: ModelRegistry) -> None:
    """编码出来的东西必须是能直接塞进 JSON 的 —— 一个 numpy 标量就够炸掉整帧。"""
    import json

    from playground.serialize import encode_graph, encode_heatmap

    engine = Engine(registry)
    engine.configure(Config(features=("x", "y", "xTimesY"), network_shape=(4, 3)))
    for _ in range(5):
        engine.step()
    graph, payload = engine.snapshot()

    message = {
        "graph": encode_graph(graph),
        "heatmaps": [encode_heatmap(h) for h in payload["heatmaps"]],
    }
    text = json.dumps(message)  # 不抛异常就说明类型都是干净的
    assert len(text) < 200_000  # 单帧别太大，否则没法维持 60fps


def test_heatmap_quantization_saturates(registry: ModelRegistry) -> None:
    """超出 [-1,1] 的值要饱和到两端，而不是回绕。"""
    from playground.serialize import encode_heatmap

    encoded = encode_heatmap(torch.tensor([[[-3.0, -1.0], [0.0, 5.0]]]))
    import base64

    raw = base64.b64decode(encoded["data"])
    assert raw[0] == 0  # -3.0 饱和到最负
    assert raw[3] == 255  # +5.0 饱和到最正
    assert raw[2] == 128  # 0.0 落在中间附近


def test_snapshot_is_safe_while_training_in_another_thread(registry: ModelRegistry) -> None:
    """训练和取快照并发时必须安全。

    服务端把训练放在线程里跑（`asyncio.to_thread`，为了不阻塞事件循环），
    而消息处理留在事件循环里 —— 两者都会碰同一个 Engine。

    没有串行化的话，两条前向传播会交错，`Tracer` 的 forward hook 互相串味：
    网格快照里会混进 **batch_size 大小的训练批次**，`_as_grid` 就报
    "网格点数 10 与 density=100 不符"。这个 bug 真实发生过 ——
    用户那边的表现是"点一下数据集就断连"，而 10 正好是默认 batch size。

    修复前实测 60 次并发里失败 50 次，所以这个测试是真能抓到回归的。
    """
    import threading
    import time

    engine = Engine(registry)
    engine.configure(Config(dataset="spiral", features=("x", "y"), network_shape=(6, 4)))

    failures: list[str] = []
    stop = threading.Event()

    def train_forever() -> None:
        try:
            while not stop.is_set():
                engine.step()
        except Exception as exc:  # noqa: BLE001 —— 线程里的异常要带回主线程断言
            failures.append(f"训练线程: {type(exc).__name__}: {exc}")

    trainer = threading.Thread(target=train_forever, daemon=True)
    trainer.start()
    try:
        time.sleep(0.05)  # 让训练线程先跑起来
        for _ in range(40):
            try:
                engine.snapshot()
            except Exception as exc:  # noqa: BLE001 —— 同上
                failures.append(f"快照: {type(exc).__name__}: {exc}")
    finally:
        stop.set()
        trainer.join(timeout=5)

    assert not failures, "训练与快照并发时出错：\n" + "\n".join(failures[:3])
    assert engine.iter > 0  # 训练线程确实在推进


def test_build_graph_is_deterministic(registry: ModelRegistry) -> None:
    """同样的模型跑两次网格前向，结构必须完全一致
    —— 否则前端会因为"结构签名变了"每帧都重建一次 DOM。"""
    engine = Engine(registry)
    engine.configure(Config(features=("x", "y"), network_shape=(4, 4)))

    model = engine.model
    grid = engine._grid_input()
    labels = ["X_1", "X_2", "X_1^2", "X_2^2", "X_1X_2", "sin(X_1)", "sin(X_2)"]
    # 必须是 list 不是 set：顺序有语义（对应模型输入列的顺序）
    active = [0, 1]
    first = build_graph(model, grid, labels, active, engine.adapters)
    second = build_graph(model, grid, labels, active, engine.adapters)

    assert [c.width for c in first.columns] == [c.width for c in second.columns]
    assert [len(layer) for layer in first.edges] == [len(layer) for layer in second.edges]
