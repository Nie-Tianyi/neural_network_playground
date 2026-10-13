"""超参与选项 —— 取值和默认值对齐原版（见原版 index.html 的 select 元素与 state.ts）。

`OPTIONS` 是前端渲染下拉框的唯一数据源：前端不硬编码任何选项，
所以加一个新的激活函数只需要改这一个文件。
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import torch
import torch.nn as nn

# --- 激活函数 ----------------------------------------------------------------
# 前三个对应原版 nn.ts 的 Activations。linear 在原版里也有，
# 但只被用作回归任务的输出层激活。
#
# softplus / gelu / silu 是本项目额外加的 —— 原版只有四种，
# 而 silu 是 KAN 的 base 分支默认使用的，放在一起对照更直观。

ACTIVATIONS: dict[str, type[nn.Module]] = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "sigmoid": nn.Sigmoid,
    "linear": nn.Identity,
    "softplus": nn.Softplus,
    "gelu": nn.GELU,
    "silu": nn.SiLU,
}


def make_activation(name: str) -> nn.Module:
    return ACTIVATIONS[name]()


# --- 选项清单（前端下拉框的数据源）-------------------------------------------

OPTIONS = {
    "learningRate": [1e-5, 1e-4, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
    "activation": list(ACTIVATIONS),
    "regularization": ["none", "l1", "l2"],
    "regularizationRate": [0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
    "optimizer": ["sgd", "momentum", "adam"],
}

DEFAULTS = {
    "learningRate": 0.03,
    "activation": "tanh",
    "regularization": "none",
    "regularizationRate": 0.0,
    "optimizer": "sgd",
    "batchSize": 10,
    "noise": 0.0,  # 归一化后；UI 滑块是 0–50
    "percTrainData": 50.0,
    "dataset": "circle",
    "features": ["x", "y"],
}


@dataclass
class Config:
    """一次实验的全部可调项。

    这是引擎和前端之间的唯一契约：前端只能改这里有的字段，
    引擎也只读这里有的字段。
    """

    # 数据
    dataset: str = "circle"
    noise: float = 0.0  # [0, 0.5]，对应 UI 滑块的 0–50
    perc_train: float = 50.0  # [10, 90]
    seed: float = 0.0
    features: tuple[str, ...] = ("x", "y")

    # 结构
    model: str = "mlp"  # 模型注册表里的 key
    # 内置 MLP 的形状（每层的神经元数）。只有声明了 configurable_shape 的模型
    # 会读它 —— 用户自己写的模型完全自由，忽略这个字段。
    # 原版默认是"1 个隐藏层 × 4 个神经元"。
    network_shape: tuple[int, ...] = (4,)

    # 优化
    learning_rate: float = 0.03
    activation: str = "tanh"
    regularization: str = "none"  # none | l1 | l2
    regularization_rate: float = 0.0
    optimizer: str = "sgd"  # sgd | momentum | adam
    batch_size: int = 10

    # 运行
    device: str = "cpu"

    def replace(self, **kwargs) -> "Config":
        return replace(self, **kwargs)

    @property
    def problem(self) -> str:
        """分类还是回归 —— 由数据集决定，不是独立选项。"""
        from .datasets import DATASETS

        return DATASETS[self.dataset].problem

    @property
    def is_regression(self) -> bool:
        return self.problem == "regression"

    @property
    def output_activation(self) -> str:
        """原版 reset() 里的规则：回归强制线性，分类用 tanh。

        分类用 tanh 是因为要输出的值域 (-1,1) 和热力图的 ±1 值域对齐，
        这样"输出值"能直接当颜色用。
        """
        return "linear" if self.is_regression else "tanh"

    def to_dict(self) -> dict:
        return {
            "dataset": self.dataset,
            "problem": self.problem,
            "noise": self.noise,
            "percTrainData": self.perc_train,
            "seed": self.seed,
            "features": list(self.features),
            "model": self.model,
            "learningRate": self.learning_rate,
            "activation": self.activation,
            "regularization": self.regularization,
            "regularizationRate": self.regularization_rate,
            "optimizer": self.optimizer,
            "batchSize": self.batch_size,
            "networkShape": list(self.network_shape),
        }
