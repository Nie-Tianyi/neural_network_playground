"""服务端协议层。

这里最重要的是一件事：**前端发过来的每个字段都要能落到 Config 上**。

曾经踩过的坑：_config_from_payload 把前端的 camelCase 字段名
（learningRate、networkShape、percTrainData…）直接透传给 snake_case 的
Config.replace()，于是前端一动下拉框就是 TypeError、整条 WebSocket 断掉。
后端单测全用 snake_case 输入，所以完全没发现 —— 是浏览器截图里
那个"连接断开"的提示暴露的。

所以这组测试对每个字段都单独断言，漏掉哪个都会被立刻发现。
"""

from __future__ import annotations

import pytest

from playground.config import Config
from playground.datasets import FEATURES
from playground.server import _config_from_payload


@pytest.mark.parametrize(
    ("payload", "field", "expected"),
    [
        ({"dataset": "spiral"}, "dataset", "spiral"),
        ({"model": "mlp"}, "model", "mlp"),
        ({"activation": "relu"}, "activation", "relu"),
        ({"regularization": "l1"}, "regularization", "l1"),
        ({"optimizer": "adam"}, "optimizer", "adam"),
        ({"learningRate": 0.1}, "learning_rate", 0.1),
        ({"regularizationRate": 0.03}, "regularization_rate", 0.03),
        ({"batchSize": 7}, "batch_size", 7),
        ({"percTrainData": 80}, "perc_train", 80.0),
        ({"seed": 0.5}, "seed", 0.5),
        ({"networkShape": [8, 4]}, "network_shape", (8, 4)),
        ({"features": ["x", "sinY"]}, "features", ("x", "sinY")),
    ],
)
def test_every_wire_field_reaches_config(payload: dict, field: str, expected: object) -> None:
    """逐个字段验证：前端用的名字 → Config 上的字段。"""
    cfg = _config_from_payload(payload, Config())
    assert getattr(cfg, field) == expected


def test_noise_is_in_normalized_units_on_the_wire() -> None:
    """协议里的 noise 一律是 0–0.5（界面上那个 0–50 的滑条由前端自己换算）。

    之前这里是 clamp(0, 0.5)，而前端发的是 0–50，于是噪声滑条实际上只有
    "关"和"拉满"两档 —— 中间 9 个刻度全是同一个值。所以单位约定必须写死并被测到。
    """
    assert _config_from_payload({"noise": 0.3}, Config()).noise == pytest.approx(0.3)
    assert _config_from_payload({"noise": 0.5}, Config()).noise == pytest.approx(0.5)
    # 超出范围要夹住，不能让前端传个 999 把数据生成搞崩
    assert _config_from_payload({"noise": 999}, Config()).noise == pytest.approx(0.5)
    assert _config_from_payload({"noise": -5}, Config()).noise == pytest.approx(0.0)


def test_client_slider_units_are_not_accepted() -> None:
    """传滑条单位（0–50）会被当成归一化值夹住 —— 这是刻意的。

    与其猜"这个数到底是 0–50 还是 0–0.5"，不如把单位固定成一种：
    前端负责换算，服务端只管夹范围。歧义比错误更危险。
    """
    assert _config_from_payload({"noise": 30}, Config()).noise == pytest.approx(0.5)


def test_network_shape_is_clamped_to_playground_limits() -> None:
    """原版约束：最多 6 个隐藏层，每层 1–8 个神经元。"""
    cfg = _config_from_payload({"networkShape": [99, 0, 4, 4, 4, 4, 4, 4, 4]}, Config())
    assert cfg.network_shape == (8, 1, 4, 4, 4, 4)


def test_unknown_features_are_dropped() -> None:
    cfg = _config_from_payload({"features": ["x", "不存在", "sinY"]}, Config())
    assert cfg.features == ("x", "sinY")


def test_empty_feature_selection_is_ignored() -> None:
    """不能把特征全关掉 —— 模型就没有输入了。"""
    cfg = _config_from_payload({"features": []}, Config())
    assert cfg.features == ("x", "y")  # 保持原样


def test_unknown_and_mistyped_fields_are_ignored() -> None:
    """前端多发字段、类型发错，都不该让连接断掉。"""
    cfg = _config_from_payload(
        {
            "不懂的字段": 1,
            "learningRate": "不是数字",  # 类型错 → 忽略
            "batchSize": True,  # bool 是 int 的子类，必须挡掉
            "dataset": 123,  # 类型错 → 忽略
        },
        Config(),
    )
    assert cfg.learning_rate == Config().learning_rate
    assert cfg.batch_size == Config().batch_size
    assert cfg.dataset == Config().dataset


def test_partial_payload_keeps_other_fields() -> None:
    """只改一个字段时，其余配置必须原样保留。"""
    original = Config(dataset="spiral", learning_rate=0.3, network_shape=(8, 8))
    cfg = _config_from_payload({"batchSize": 4}, original)
    assert cfg.dataset == "spiral"
    assert cfg.learning_rate == 0.3
    assert cfg.network_shape == (8, 8)
    assert cfg.batch_size == 4


def test_none_current_config_uses_defaults() -> None:
    cfg = _config_from_payload({"dataset": "xor"}, None)
    assert cfg.dataset == "xor"
    assert cfg.activation == Config().activation


def test_switching_model_applies_its_suggested_hyperparameters() -> None:
    """换模型要带上它建议的超参 —— KAN 用 MLP 的默认 SGD/0.03 基本学不动。

    这条在服务端做而不是只靠前端，是因为 `?model=kan` 这种 URL 直接进来的
    路径也要生效，否则分享出去的链接会给别人一套错误的超参。
    """
    from playground.registry import default_registry

    registry = default_registry()
    cfg = _config_from_payload({"model": "kan"}, Config(), registry)

    spec = registry.get("kan")
    assert spec is not None, "内置模型表里应该有 kan"
    assert cfg.optimizer == spec.defaults["optimizer"]
    assert cfg.learning_rate == spec.defaults["learningRate"]


def test_explicit_fields_win_over_model_defaults() -> None:
    """URL 里显式写了学习率，就不该被模型建议值覆盖。"""
    from playground.registry import default_registry

    cfg = _config_from_payload(
        {"model": "kan", "learningRate": 0.3}, Config(), default_registry()
    )
    assert cfg.learning_rate == 0.3


def test_model_defaults_do_not_apply_when_model_is_unchanged() -> None:
    """模型没变就不该动用户的超参 —— 否则每次刷新页面都会把调好的参数重置。"""
    from playground.registry import default_registry

    current = Config(model="kan", learning_rate=0.05, optimizer="momentum")
    cfg = _config_from_payload({"model": "kan"}, current, default_registry())
    assert cfg.learning_rate == 0.05
    assert cfg.optimizer == "momentum"


def test_problem_and_output_activation_derived_from_dataset() -> None:
    """问题类型和输出层激活是**推导**出来的，不是独立可设的字段。"""
    assert _config_from_payload({"dataset": "circle"}, Config()).problem == "classification"
    assert _config_from_payload({"dataset": "plane"}, Config()).problem == "regression"
    assert _config_from_payload({"dataset": "plane"}, Config()).output_activation == "linear"
    assert _config_from_payload({"dataset": "circle"}, Config()).output_activation == "tanh"


def test_static_message_is_json_serializable() -> None:
    """static 消息要能直接塞进 JSON —— 里面有个 numpy 标量就够炸掉首屏。"""
    import json

    from playground.registry import default_registry
    from playground.server import _static_message

    message = _static_message(default_registry(), Config())
    text = json.dumps(message)
    assert len(text) > 0

    assert {f["id"] for f in message["features"]} == set(FEATURES)
    assert len(message["datasets"]) == 10  # 6 个二维 + 4 个一维拟合
    assert all(len(d["thumbnail"]) == 200 for d in message["datasets"])
    assert any(m["key"] == "mlp" for m in message["models"])
    # 一维数据集必须带 mode/domain/features，前端靠它们切换输出面板
    for dataset in message["datasets"]:
        assert dataset["mode"] in ("1d", "2d")
        assert len(dataset["domain"]) == 2
        assert len(dataset["features"]) >= 1


def test_default_config_round_trips_through_the_wire() -> None:
    """服务端发出去的 config 再发回来，应该原样不变（前端是照抄着传的）。"""
    original = Config()
    wire = original.to_dict()
    assert _config_from_payload(wire, original) == original
