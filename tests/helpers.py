"""测试共用的取值辅助。

`Engine.model` 和 `Engine.cfg` 的类型是 `X | None` —— 在 `configure()` 之前
它们确实是 `None`，生产代码里每次用之前都有 `assert ... is not None`。

但测试里到处都要把它们取出来用（`engine.model.parameters()`、
`engine.cfg.replace(...)`、`engine.model.layers[0]`），把断言散在每个用例里
既啰嗦又容易漏掉一处。所以收成这几个 helper：**一次断言，之后拿到的是具体类型**。

这不是"为了过类型检查"才加的 —— "此处不该是 None" 这个前提本来就成立，
写出来只是把它显式化，顺带让 PyLance / pyright 能正确地往下推断。
"""

from __future__ import annotations

from typing import cast

from torch import nn

from playground.builtin_models import KAN, MLP
from playground.config import Config
from playground.engine import Engine
from playground.layers import KANLinear


def cfg_of(engine: Engine) -> Config:
    """取出引擎的配置（`configure()` 之后必然存在）。"""
    assert engine.cfg is not None
    return engine.cfg


def model_of[ModelT: nn.Module](engine: Engine, kind: type[ModelT]) -> ModelT:
    """取出引擎里的模型，并断言它的具体类型。

    `kind` 有**两个不同的用途**，这点不太直观：

    - 给**类型检查器**看：让它知道返回的是 `KAN` / `MLP` 而不是裸的 `nn.Module`，
      这样 `model.net[0]`、`model.layers[0]` 才有类型。
    - 运行期的类校验用的是**注册表里那个类** `spec.cls`，不是 `kind`。

    为什么不能直接 `isinstance(engine.model, kind)`：`KAN` 定义在
    `builtin_models.py` 里，而注册表是用 `_exec_fresh` **全新执行那个文件**
    加载的（热重载机制，见 registry.py）—— 每次都是新的模块对象、新的类。
    所以 `from playground.builtin_models import KAN` 拿到的和注册表里的
    **不是同一个类对象**，isinstance 会失败。
    """
    spec = engine.registry.get(cfg_of(engine).model)
    assert spec is not None, f"注册表里没有 {cfg_of(engine).model}"
    model = engine.model
    assert model is not None
    assert isinstance(model, spec.cls), (
        f"期望 {spec.cls.__name__}，实际是 {type(model).__name__}"
    )
    return cast(ModelT, model)


def kan_layer_of(engine: Engine, index: int = 0) -> KANLinear:
    """取出 KAN 模型的第 index 个 KANLinear（测试里反复要拿它看网格）。"""
    layer = model_of(engine, KAN).layers[index]
    assert isinstance(layer, KANLinear)
    return layer


def mlp_head_of(engine: Engine) -> nn.Linear:
    """内置 MLP 的第一个 Linear。

    有几个测试要直接改它的权重/偏置来构造特定情形（比如"偏置很大、
    权重全零"，用来验证节点上显示的是激活后的值而不是加权和）。
    """
    head = model_of(engine, MLP).net[0]
    assert isinstance(head, nn.Linear)
    return head
