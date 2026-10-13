"""数据集生成 —— 逐条复刻 TensorFlow Playground 的 src/dataset.ts。

复刻这件事上，"看起来差不多"是没有意义的：真正决定实验结论的是那些
具体的采样半径、噪声施加位置、标签阈值。所以这里的每个魔数都来自原版源码，
注释里标了出处，改动前请先确认是不是有意为之。

原版有一个很容易看漏、但会实质影响结果的设计：**噪声的施加方式不统一**。
    - Circle / XOR / Gaussian / Plane / Multi-Gaussian：噪声只参与"标签怎么定"，
      push 进数据集的坐标是干净的。也就是说噪声制造的是"贴着边界的点被标错"，
      而不是"点被撒开了"。
    - Spiral：噪声直接加在坐标上。
这两者有本质区别，本项目照抄。
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Callable

import torch
from torch import Tensor

# --- 原版常量（playground.ts）------------------------------------------------

NUM_SAMPLES_CLASSIFY = 500
NUM_SAMPLES_REGRESS = 1200

# 主热力图与决策边界的坐标域固定为 [-6, 6]（原版 xDomain）。
X_DOMAIN = (-6.0, 6.0)

# 决策边界网格边长。原版是 100×100，逐点做前向传播（这是它最大的性能瓶颈）。
DENSITY = 100


@dataclass(frozen=True)
class Point:
    """一个数据点。x/y 是画在坐标域里的位置，label 是分类的 ±1 或回归的连续值。"""

    x: float
    y: float
    label: float


@dataclass(frozen=True)
class DataSplit:
    train: list[Point]
    test: list[Point]


# --- 基础随机数 --------------------------------------------------------------
# 原版用 seedrandom 库全局替换 Math.random 来实现"可复现"。
# Python 这边用显式传入的 Random 实例，等价且更干净。
# normalRandom 用拒绝采样，是为了和原版逐行对应（标准库的 gauss 是另一种算法）。


def _rand_uniform(rng: random.Random, a: float, b: float) -> float:
    return rng.random() * (b - a) + a


def _normal_random(rng: random.Random, mean: float = 0.0, variance: float = 1.0) -> float:
    """正态分布，拒绝采样实现（对应原版 dataset.ts 的 normalRandom）。"""
    while True:
        v1 = 2 * rng.random() - 1
        v2 = 2 * rng.random() - 1
        s = v1 * v1 + v2 * v2
        if s <= 1:
            break
    result = math.sqrt(-2 * math.log(s) / s) * v1
    return mean + math.sqrt(variance) * result


def _shuffle(rng: random.Random, array: list) -> None:
    """Fisher–Yates，原地打乱。逐行对应原版的 shuffle。"""
    counter = len(array)
    while counter > 0:
        index = int(rng.random() * counter)
        counter -= 1
        array[counter], array[index] = array[index], array[counter]


# --- 分类数据集 --------------------------------------------------------------


def classify_circle(rng: random.Random, num_samples: int, noise: float) -> list[Point]:
    """圆环套圆点。内圈正类、外圈负类，判定阈值是半径 2.5。"""
    radius = 5.0
    points: list[Point] = []

    for r_lo, r_hi in ((0.0, radius * 0.5), (radius * 0.7, radius)):
        for _ in range(num_samples // 2):
            r = _rand_uniform(rng, r_lo, r_hi)
            angle = _rand_uniform(rng, 0, 2 * math.pi)
            x = r * math.sin(angle)
            y = r * math.cos(angle)

            # 噪声只用于判定标签 —— 上面的 x, y 保持干净。
            noise_x = _rand_uniform(rng, -radius, radius) * noise
            noise_y = _rand_uniform(rng, -radius, radius) * noise
            dist = math.hypot(x + noise_x, y + noise_y)
            label = 1.0 if dist < radius * 0.5 else -1.0

            points.append(Point(x, y, label))

    return points


def classify_xor(rng: random.Random, num_samples: int, noise: float) -> list[Point]:
    """异或。四个象限交替，`padding=0.3` 把点推离坐标轴，避免落在 x*y=0 的判定边界上。"""
    padding = 0.3
    points: list[Point] = []

    for _ in range(num_samples):
        x = _rand_uniform(rng, -5, 5)
        x += padding if x > 0 else -padding
        y = _rand_uniform(rng, -5, 5)
        y += padding if y > 0 else -padding

        noise_x = _rand_uniform(rng, -5, 5) * noise
        noise_y = _rand_uniform(rng, -5, 5) * noise
        label = 1.0 if (x + noise_x) * (y + noise_y) >= 0 else -1.0

        points.append(Point(x, y, label))

    return points


def classify_gaussian(rng: random.Random, num_samples: int, noise: float) -> list[Point]:
    """两个高斯团。noise 直接控制方差：0 → 0.5，0.5 → 4（标准差 0.71 → 2）。

    注意这里噪声是"把点撒开"而不是"改标签"—— 和 circle/xor 相反，
    所以调大 noise 时这两类的直观感受完全不同。
    """
    variance = 0.5 + (noise / 0.5) * (4.0 - 0.5)
    points: list[Point] = []

    for cx, cy, label in ((2.0, 2.0, 1.0), (-2.0, -2.0, -1.0)):
        for _ in range(num_samples // 2):
            x = _normal_random(rng, cx, variance)
            y = _normal_random(rng, cy, variance)
            points.append(Point(x, y, label))

    return points


def classify_spiral(rng: random.Random, num_samples: int, noise: float) -> list[Point]:
    """双螺旋。两条臂角度差 π，各自绕 1.75 圈，半径从 0 长到 5。

    和其他数据集不同，这里的噪声**直接加在坐标上**。
    """
    n = num_samples // 2
    points: list[Point] = []

    for delta_t, label in ((0.0, 1.0), (math.pi, -1.0)):
        for i in range(n):
            r = i / n * 5
            t = 1.75 * i / n * 2 * math.pi + delta_t
            x = r * math.sin(t) + _rand_uniform(rng, -1, 1) * noise
            y = r * math.cos(t) + _rand_uniform(rng, -1, 1) * noise
            points.append(Point(x, y, label))

    return points


# --- 回归数据集 --------------------------------------------------------------


def regress_plane(rng: random.Random, num_samples: int, noise: float) -> list[Point]:
    """平面 `label = (x+y)/10`，值域约 [-1, 1]。"""
    radius = 6.0
    points: list[Point] = []

    for _ in range(num_samples):
        x = _rand_uniform(rng, -radius, radius)
        y = _rand_uniform(rng, -radius, radius)
        noise_x = _rand_uniform(rng, -radius, radius) * noise
        noise_y = _rand_uniform(rng, -radius, radius) * noise
        points.append(Point(x, y, (x + noise_x + y + noise_y) / 10))

    return points


# 六个高斯鼓包的位置与符号。距离 0 处 |label|=1，距离 ≥2 处为 0，
# 取 |label| 最大的那个鼓包胜出 —— 于是平面被切成六块"台地"。
_GAUSSIANS = [
    (-4.0, 2.5, 1.0),
    (0.0, 2.5, -1.0),
    (4.0, 2.5, 1.0),
    (-4.0, -2.5, -1.0),
    (0.0, -2.5, 1.0),
    (4.0, -2.5, -1.0),
]


def regress_gaussian(rng: random.Random, num_samples: int, noise: float) -> list[Point]:
    """六个交替符号的高斯台地。回归里最难的那个，适合演示"欠拟合 vs 容量"。"""
    radius = 6.0
    points: list[Point] = []

    for _ in range(num_samples):
        x = _rand_uniform(rng, -radius, radius)
        y = _rand_uniform(rng, -radius, radius)
        noise_x = _rand_uniform(rng, -radius, radius) * noise
        noise_y = _rand_uniform(rng, -radius, radius) * noise
        points.append(Point(x, y, _multi_gaussian_label(x + noise_x, y + noise_y)))

    return points


def _multi_gaussian_label(x: float, y: float) -> float:
    label = 0.0
    for cx, cy, sign in _GAUSSIANS:
        d = math.hypot(x - cx, y - cy)
        # labelScale: domain [0,2] → range [1,0]，clamp(true)
        scaled = min(max(1.0 - d / 2.0, 0.0), 1.0)
        candidate = sign * scaled
        if abs(candidate) > abs(label):
            label = candidate
    return label


# --- 数据集注册表 ------------------------------------------------------------


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    title: str
    problem: str  # "classification" | "regression"
    num_samples: int
    generator: Callable[[random.Random, int, float], list[Point]]


DATASETS: dict[str, DatasetSpec] = {
    spec.id: spec
    for spec in (
        DatasetSpec("circle", "Circle", "classification", NUM_SAMPLES_CLASSIFY, classify_circle),
        DatasetSpec("xor", "Exclusive or", "classification", NUM_SAMPLES_CLASSIFY, classify_xor),
        DatasetSpec("gauss", "Gaussian", "classification", NUM_SAMPLES_CLASSIFY, classify_gaussian),
        DatasetSpec("spiral", "Spiral", "classification", NUM_SAMPLES_CLASSIFY, classify_spiral),
        DatasetSpec("plane", "Plane", "regression", NUM_SAMPLES_REGRESS, regress_plane),
        DatasetSpec("multi-gauss", "Multi gaussian", "regression", NUM_SAMPLES_REGRESS, regress_gaussian),
    )
}


def generate(
    dataset_id: str,
    noise: float,
    perc_train: float,
    seed: float,
) -> DataSplit:
    """生成数据并按原版方式划分 train/test。

    noise 传进来就是 [0, 0.5]（UI 上的 0–50 已在这一层之外除过 100）。
    perc_train 是百分比（10–90）。

    划分方式是**先整体打乱，再切前 N 个作训练集**，不是随机抽样 ——
    所以训练集和测试集的分布是同质的，两边 loss 的差距反映的是过拟合而非分布偏移。
    """
    spec = DATASETS[dataset_id]
    rng = random.Random(seed)

    points = spec.generator(rng, spec.num_samples, noise)
    _shuffle(rng, points)

    split_index = int(len(points) * perc_train / 100)
    return DataSplit(train=points[:split_index], test=points[split_index:])


# --- 输入特征 ----------------------------------------------------------------
# 原版的 7 个特征。注意没有 AND/OR 之类的逻辑门特征 —— X_1X_2 只是一个乘法特征。
# 这 7 个里挑哪几个，直接决定了模型看到的世界，是 Playground 最重要的教学点之一。


@dataclass(frozen=True)
class Feature:
    id: str
    label: str  # 显示名，下划线后一位会渲染成下标、^ 后一位渲染成上标
    fn: Callable[[float, float], float]
    is_input: bool  # True 表示画在网络图最左列（即"原始输入"）


FEATURES: dict[str, Feature] = {
    f.id: f
    for f in (
        Feature("x", "X_1", lambda x, y: x, True),
        Feature("y", "X_2", lambda x, y: y, True),
        Feature("xSquared", "X_1^2", lambda x, y: x * x, False),
        Feature("ySquared", "X_2^2", lambda x, y: y * y, False),
        Feature("xTimesY", "X_1X_2", lambda x, y: x * y, False),
        Feature("sinX", "sin(X_1)", lambda x, y: math.sin(x), False),
        Feature("sinY", "sin(X_2)", lambda x, y: math.sin(y), False),
    )
}

# 默认选中的特征（原版初始状态只开 X_1、X_2）。
DEFAULT_FEATURES = ("x", "y")


# 特征的分量级（torch）实现。
# 上面 Feature.fn 是标量版，便于逐点理解和调试；这里是向量版，
# 用来一次性算出上万个网格点的特征 —— 每帧都要跑，不能用 Python 循环。
FEATURE_TORCH: dict[str, Callable[[Tensor, Tensor], Tensor]] = {
    "x": lambda x, y: x,
    "y": lambda x, y: y,
    "xSquared": lambda x, y: x * x,
    "ySquared": lambda x, y: y * y,
    "xTimesY": lambda x, y: x * y,
    "sinX": lambda x, y: torch.sin(x),
    "sinY": lambda x, y: torch.sin(y),
}


def feature_matrix(xs: Tensor, ys: Tensor, feature_ids: tuple[str, ...] | list[str]) -> Tensor:
    """把 (x, y) 坐标张量按选中的特征拼成输入矩阵 (N, F)。"""
    return torch.stack([FEATURE_TORCH[fid](xs, ys) for fid in feature_ids], dim=-1)


def points_to_xy(points: list[Point]) -> tuple[Tensor, Tensor]:
    xs = torch.tensor([p.x for p in points], dtype=torch.float32)
    ys = torch.tensor([p.y for p in points], dtype=torch.float32)
    labels = torch.tensor([p.label for p in points], dtype=torch.float32).unsqueeze(-1)
    return xs, ys, labels


def grid_points(density: int = DENSITY) -> list[tuple[float, float]]:
    """决策边界的采样网格，按行优先返回 (x, y)。

    对齐原版的坐标映射：
        x = xDomain[0] + i/(density-1) * (xDomain[1]-xDomain[0])
        y = yDomain[1] - j/(density-1) * (yDomain[1]-yDomain[0])   # j=0 在最上方
    返回顺序是 j 外层、i 内层，所以直接 reshape(density, density) 就是一个
    行 0 在顶部、可以直接当图像画出来的矩阵。
    """
    lo, hi = X_DOMAIN
    span = hi - lo
    return [
        (lo + i / (density - 1) * span, hi - j / (density - 1) * span)
        for j in range(density)
        for i in range(density)
    ]
