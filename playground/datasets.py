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


# --- 一维函数拟合 ------------------------------------------------------------
#
# 原版 Playground 只有二维任务（2D 分类 + 2D 回归）。这一组是额外加的，
# 理由是：**KAN 的招牌能力是 1D 函数拟合，而 circle/spiral 这类 2D 小任务上
# 它和小 MLP 打得有来有回**，只看那些任务会得出"KAN 好像也没什么"的结论。
#
# 一维、且目标是光滑函数时，KAN 每条边的 φ(x) 拼起来就是目标函数本身 ——
# 这是它区别于 MLP 最直观的地方，也是这一组数据集存在的意义。

# 定义域比二维任务窄：[-3, 3] 足够放下几个周期，又不会让曲线挤成一团。
DOMAIN_1D = (-3.0, 3.0)
NUM_SAMPLES_1D = 300
# 一维网格的点数。二维要 100×100 才填满一个平面，一维 200 个点已经很细了。
GRID_POINTS_1D = 200


def _f_sin(x: float) -> float:
    return math.sin(3 * x)


def _f_cubic(x: float) -> float:
    return x**3 / 9


def _f_exp_sin(x: float) -> float:
    """复合函数 —— KAN 论文里的招牌例子就是这个形状。"""
    return math.exp(math.sin(3 * x)) - 1.5


def _f_abs(x: float) -> float:
    """分段线性、在 0 处不可导。用来看基函数的局部性够不够。"""
    return abs(x) - 1.5


def fit_1d(rng: random.Random, num_samples: int, noise: float, fn=None) -> list[Point]:
    """在定义域上均匀采样 x，标签是 f(x)（可选加高斯噪声）。

    `noise` 在这里被当成标签噪声的**标准差**（二维任务里它各有各的含义，
    见各生成器的注释）。x 本身不加噪 —— 函数拟合的标准设定就是"输入是准的，
    观测有噪声"。

    **噪声单独走一条随机流**（只用主随机流生成一次它的种子）。否则噪声的
    拒绝采样会消耗掉主随机流的 Draw，于是滑动噪声滑条时连 x 的位置都会变 ——
    而这时候你最想看到的恰恰是"同一批点被加上了抖动"。
    """
    lo, hi = DOMAIN_1D
    noise_rng = random.Random(rng.randrange(2**63))
    points: list[Point] = []
    for _ in range(num_samples):
        x = _rand_uniform(rng, lo, hi)
        value = fn(x) if fn is not None else 0.0
        if noise > 0:
            value += _normal_random(noise_rng, 0.0, noise * noise)
        # y 恒为 0：一维数据没有第二个坐标轴，Point.y 在这里不参与任何计算。
        points.append(Point(x, 0.0, value))
    return points


def _make_1d_generator(fn: Callable[[float], float]):
    def generator(rng: random.Random, num_samples: int, noise: float) -> list[Point]:
        return fit_1d(rng, num_samples, noise, fn)

    return generator


# --- 数据集注册表 ------------------------------------------------------------


# 二维任务可用的输入特征。**顺序必须和下面 FEATURES 的插入顺序一致** ——
# 输入列画的是全部特征、模型只吃选中的那几个，两边下标要对得上。
FEATURE_IDS_2D = ("x", "y", "xSquared", "ySquared", "xTimesY", "sinX", "sinY")
# 一维拟合只有一个输入：标量 x。
FEATURE_IDS_1D = ("x",)


@dataclass(frozen=True)
class DatasetSpec:
    id: str
    title: str
    problem: str  # "classification" | "regression"
    num_samples: int
    generator: Callable[[random.Random, int, float], list[Point]]

    # "2d" 画决策边界热力图；"1d" 画函数拟合曲线。两者的采样网格、
    # 节点渲染方式、输出面板都不一样。
    mode: str = "2d"
    domain: tuple[float, float] = X_DOMAIN
    # 这个数据集允许使用哪些输入特征（决定输入列画几个方块）
    features: tuple[str, ...] = FEATURE_IDS_2D
    # 一维模式下用来画"真实函数"参考线的那条曲线
    truth: Callable[[float], float] | None = None

    @property
    def is_1d(self) -> bool:
        return self.mode == "1d"


DATASETS: dict[str, DatasetSpec] = {
    spec.id: spec
    for spec in (
        DatasetSpec("circle", "Circle", "classification", NUM_SAMPLES_CLASSIFY, classify_circle),
        DatasetSpec("xor", "Exclusive or", "classification", NUM_SAMPLES_CLASSIFY, classify_xor),
        DatasetSpec("gauss", "Gaussian", "classification", NUM_SAMPLES_CLASSIFY, classify_gaussian),
        DatasetSpec("spiral", "Spiral", "classification", NUM_SAMPLES_CLASSIFY, classify_spiral),
        DatasetSpec("plane", "Plane", "regression", NUM_SAMPLES_REGRESS, regress_plane),
        DatasetSpec("multi-gauss", "Multi gaussian", "regression", NUM_SAMPLES_REGRESS, regress_gaussian),
        # --- 一维函数拟合 ---
        DatasetSpec(
            "fit-sin", "sin(3x)", "regression", NUM_SAMPLES_1D,
            _make_1d_generator(_f_sin),
            mode="1d", domain=DOMAIN_1D, features=FEATURE_IDS_1D, truth=_f_sin,
        ),
        DatasetSpec(
            "fit-cubic", "x³/9", "regression", NUM_SAMPLES_1D,
            _make_1d_generator(_f_cubic),
            mode="1d", domain=DOMAIN_1D, features=FEATURE_IDS_1D, truth=_f_cubic,
        ),
        DatasetSpec(
            "fit-exp-sin", "exp(sin 3x)", "regression", NUM_SAMPLES_1D,
            _make_1d_generator(_f_exp_sin),
            mode="1d", domain=DOMAIN_1D, features=FEATURE_IDS_1D, truth=_f_exp_sin,
        ),
        DatasetSpec(
            "fit-abs", "|x|", "regression", NUM_SAMPLES_1D,
            _make_1d_generator(_f_abs),
            mode="1d", domain=DOMAIN_1D, features=FEATURE_IDS_1D, truth=_f_abs,
        ),
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


def grid_line(domain: tuple[float, float], count: int = GRID_POINTS_1D) -> Tensor:
    """一维模式的采样网格：定义域上均匀的 x。"""
    return torch.linspace(domain[0], domain[1], count, dtype=torch.float32)


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
