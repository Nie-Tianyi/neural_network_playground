"""一维函数拟合模式。

这一组数据集是原版没有的。理由：**KAN 的招牌能力是 1D 函数拟合，
而 circle/spiral 那类 2D 小任务上它和小 MLP 打得有来有回** ——
只看那些任务会得出"KAN 好像也没什么"的结论。

一维模式在实现上和二维共用同一条流水线（前向 → hook → DisplayGraph → 编码），
差别只在三处：采样网格是一条线而不是一个平面、节点画的是曲线而不是热力图、
输出面板画的是拟合图而不是决策边界。所以这里重点测这三处的边界。
"""

from __future__ import annotations

import math

import pytest
import torch

from playground.config import Config
from playground.datasets import (
    DATASETS,
    DOMAIN_1D,
    GRID_POINTS_1D,
    grid_line,
    grid_points,
)
from playground.engine import Engine
from playground.registry import default_registry

FIT_DATASETS = [d for d, spec in DATASETS.items() if spec.is_1d]


@pytest.fixture(scope="module")
def registry():
    return default_registry()


# --- 数据集 ------------------------------------------------------------------


def test_there_are_one_dimensional_datasets() -> None:
    assert len(FIT_DATASETS) >= 4


@pytest.mark.parametrize("dataset_id", FIT_DATASETS)
def test_1d_points_follow_the_target_function(dataset_id: str) -> None:
    """没有噪声时，每个点的标签必须精确等于目标函数在那个 x 上的取值。"""
    from playground.datasets import generate

    spec = DATASETS[dataset_id]
    split = generate(dataset_id, noise=0.0, perc_train=50.0, seed=1.0)

    for point in split.train + split.test:
        assert spec.domain[0] <= point.x <= spec.domain[1]
        assert point.y == 0.0  # 一维数据没有第二个坐标
        assert point.label == pytest.approx(spec.truth(point.x), abs=1e-9)


def test_1d_noise_only_jitters_the_label() -> None:
    """函数拟合的标准设定是"输入是准的、观测有噪声"。

    而且**换噪声值不能让 x 的位置变**：滑动噪声滑条时你想看的是同一批点
    被加上抖动，不是换了一批点。噪声走的是独立随机流，所以这里能直接断言
    x 序列逐点相等。
    """
    from playground.datasets import generate

    clean = generate("fit-sin", noise=0.0, perc_train=50.0, seed=7.0)
    noisy = generate("fit-sin", noise=0.3, perc_train=50.0, seed=7.0)

    assert [p.x for p in clean.train] == [p.x for p in noisy.train]
    # 加了噪声之后标签要真的变了
    assert any(abs(a.label - b.label) > 1e-3 for a, b in zip(clean.train, noisy.train))


def test_1d_noise_shrinks_toward_the_clean_function() -> None:
    """噪声越小，标签离真实函数越近 —— 滑条的方向不能反。"""
    from playground.datasets import generate

    def mean_deviation(noise: float) -> float:
        spec = DATASETS["fit-sin"]
        split = generate("fit-sin", noise=noise, perc_train=50.0, seed=3.0)
        points = split.train + split.test
        return sum(abs(p.label - spec.truth(p.x)) for p in points) / len(points)

    assert mean_deviation(0.0) < 1e-9
    assert mean_deviation(0.05) < mean_deviation(0.3)


def test_1d_datasets_only_allow_the_x_feature() -> None:
    """一维数据集只暴露一个输入特征 —— 让用户去勾 X₁² 之类的是没有意义的。"""
    for dataset_id in FIT_DATASETS:
        assert DATASETS[dataset_id].features == ("x",)
        assert DATASETS[dataset_id].problem == "regression"  # 拟合就是回归


def test_grid_line_spans_the_domain() -> None:
    line = grid_line(DOMAIN_1D)
    assert len(line) == GRID_POINTS_1D
    assert float(line[0]) == pytest.approx(DOMAIN_1D[0])
    assert float(line[-1]) == pytest.approx(DOMAIN_1D[1])


# --- 配置 --------------------------------------------------------------------


def test_switching_to_1d_clamps_the_features(registry) -> None:
    """从二维切到一维时，原来选中的 (x, y, x²) 里只有 x 还有意义。

    不夹住的话模型会拿到 in_dim=3，而输入列只画 1 个方块，索引立刻对不上
    —— 那正是之前在二维模式里踩过的"边挂到错误的特征上"那一类 bug。
    """
    engine = Engine(registry)
    engine.configure(Config(dataset="circle", features=("x", "y", "xSquared")))
    assert engine.cfg.features == ("x", "y", "xSquared")

    engine.configure(engine.cfg.replace(dataset="fit-cubic"))
    assert engine.cfg.features == ("x",)


def test_switching_back_to_2d_keeps_working(registry) -> None:
    """一维切回二维之后要能正常训练，不能留下坏的 in_dim。"""
    engine = Engine(registry)
    engine.configure(Config(model="mlp", dataset="fit-sin"))
    engine.step()

    engine.configure(engine.cfg.replace(dataset="circle"))
    for _ in range(20):
        engine.step()
    assert engine.iter == 20
    assert engine.losses.train < 1.0


def test_output_activation_is_linear_for_1d_fits(registry) -> None:
    """拟合任务必须用线性输出。

    分类用的 tanh 会把输出夹在 (-1,1)，而 `x³/9` 的标签能到 ±3 ——
    夹住就永远拟合不到位。
    """
    for dataset_id in FIT_DATASETS:
        assert Config(dataset=dataset_id).output_activation == "linear"


# --- 图与可视化 --------------------------------------------------------------


def test_1d_input_column_has_a_single_node(registry) -> None:
    engine = Engine(registry)
    engine.configure(Config(model="kan", dataset="fit-sin", network_shape=(3,)))
    graph, payload = engine.snapshot()

    assert [(c.kind, c.width) for c in graph.columns] == [
        ("input", 1),
        ("hidden", 3),
        ("output", 1),
    ]
    assert [len(layer) for layer in graph.edges] == [1 * 3, 3 * 1]


def test_1d_snapshot_returns_curves_not_heatmaps(registry) -> None:
    """一维模式送的是曲线、不是热力图 —— 两者的形状和渲染方式都不同，
    混了的话前端会拿热力图的画法去解读曲线数组。"""
    engine = Engine(registry)
    engine.configure(Config(model="kan", dataset="fit-sin", network_shape=(3,)))
    graph, payload = engine.snapshot()

    assert "curves" in payload
    assert "heatmaps" not in payload
    assert [tuple(c.shape) for c in payload["curves"]] == [
        (GRID_POINTS_1D, 1),
        (GRID_POINTS_1D, 3),
        (GRID_POINTS_1D, 1),
    ]


def test_2d_snapshot_still_returns_heatmaps(registry) -> None:
    engine = Engine(registry)
    engine.configure(Config(model="mlp", dataset="circle", network_shape=(3,)))
    _, payload = engine.snapshot()

    assert "heatmaps" in payload
    assert "curves" not in payload


def test_1d_model_input_is_a_single_column(registry) -> None:
    """一维模式喂给模型的输入必须是 (N, 1)。

    实现上是用 (x, 0) 补齐成二维坐标再走同一套特征函数 ——
    这里查的就是那个补位没有漏出来变成两个输入。
    """
    engine = Engine(registry)
    engine.configure(Config(model="mlp", dataset="fit-cubic"))
    assert tuple(engine._grid_input().shape) == (GRID_POINTS_1D, 1)
    assert tuple(engine._grid_all_features().shape) == (GRID_POINTS_1D, 1)


def test_1d_grid_input_equals_the_line(registry) -> None:
    engine = Engine(registry)
    engine.configure(Config(model="mlp", dataset="fit-sin"))
    x = engine._grid_input()[:, 0]
    assert torch.allclose(x, grid_line(DOMAIN_1D))


def test_2d_grid_is_unchanged(registry) -> None:
    """二维那边的网格不能被动过。"""
    engine = Engine(registry)
    engine.configure(Config(model="mlp", dataset="circle", features=("x", "y")))
    assert tuple(engine._grid_input().shape) == (100 * 100, 2)
    assert engine._grid_input()[0].tolist() == list(grid_points(100)[0])


# --- 训练 --------------------------------------------------------------------


@pytest.mark.parametrize("model", ["mlp", "kan"])
@pytest.mark.parametrize("dataset_id", FIT_DATASETS)
def test_both_models_fit_every_1d_function(
    model: str, dataset_id: str, registry
) -> None:
    """MLP 和 KAN 都要能把每个一维函数拟合下来。

    这组对照才是这个模式的意义所在：同样的数据、同样的预算，两条曲线叠在一起看。
    """
    engine = Engine(registry)
    engine.configure(
        Config(
            model=model,
            dataset=dataset_id,
            network_shape=(8,) if model == "kan" else (16, 16),
            optimizer="adam",
            learning_rate=0.01,
            seed=2.0,
        )
    )
    start = engine.losses.train
    for _ in range(1200):
        engine.step()

    assert engine.losses.train < start * 0.2, (
        f"{model} 在 {dataset_id} 上没拟合出来：{start:.4f} → {engine.losses.train:.4f}"
    )


def test_1d_training_is_reproducible(registry) -> None:
    def run() -> float:
        engine = Engine(registry)
        engine.configure(Config(model="kan", dataset="fit-sin", seed=4.0))
        for _ in range(50):
            engine.step()
        return engine.losses.train

    assert run() == pytest.approx(run(), abs=1e-9)


def test_target_function_is_reachable(registry) -> None:
    """拟合到位的标志：预测曲线和真实函数逐点接近。"""
    engine = Engine(registry)
    engine.configure(
        Config(model="kan", dataset="fit-cubic", network_shape=(8,), optimizer="adam",
               learning_rate=0.01, seed=2.0)
    )
    for _ in range(1200):
        engine.step()

    graph, payload = engine.snapshot()
    prediction = payload["curves"][-1][:, 0]
    x_axis = engine._grid_input()[:, 0]
    truth = torch.tensor([math.pow(float(v), 3) / 9 for v in x_axis])

    # x³/9 在两端到 ±3，边界处样条要多花点力气，0.2 是个合理的容差
    assert torch.allclose(prediction, truth, atol=0.2)
    # 整体误差很小才是真的拟合上了（不是"每点都差一点点"）
    assert float((prediction - truth).abs().mean()) < 0.05
