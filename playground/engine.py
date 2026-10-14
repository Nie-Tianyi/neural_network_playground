"""训练引擎 —— 不依赖 Web 的纯 Python 类。

刻意做成不碰 fastapi/websocket 的样子，好处有三个：
  * 「多实验对比」只是同时持有多个 Engine 并同步推进一步，不需要特殊机制
  * 能在 Jupyter / 脚本里直接用（构造 Engine + 反复 step + 读 snapshot）
  * 能脱离浏览器做单元测试

## 几个照抄原版的地方（不是随手写的）

1. **1 epoch = 完整遍历训练集一遍**，每 batch_size 个点更新一次权重。
2. **数据顺序在 epoch 之间是固定的**。原版只在生成数据时打乱一次，
   之后每个 epoch 都按同样的顺序切 batch。这和"每轮重新打乱"是实质区别，
   会影响你观察到的训练动态，所以照抄。
3. **loss = mean(0.5 * (输出 - 目标)²)**，即 MSE 的一半，数值和原版一致。
4. **正则化不污染数据梯度**：原版是先 `w -= lr*grad`，再单独
   `w -= lr*regRate*∂reg/∂w`。数学上等价于把 reg 项加进 loss，
   这里就加进 loss 了 —— 对 Adam 这类自适应优化器来说这才是正确处理方式。
5. **L1 的"过零即杀死"规则**：权重被推过零点时置 0 并**永久冻结**。
   这是原版稀疏化的来源，照抄。
"""

from __future__ import annotations

import functools
import threading
import zlib
from dataclasses import dataclass

import torch
from torch import Tensor

from .adapters.base import default_adapter_registry
from .config import Config
from .datasets import (
    DATASETS,
    DENSITY,
    FEATURES,
    Point,
    feature_matrix,
    generate,
    grid_line,
    grid_points,
    points_to_xy,
)
from .introspection import DisplayGraph, build_graph
from .registry import ModelRegistry

# 神经元小热力图的分辨率。原版就是把 100×100 按 10×10 块平均降采样到 10×10。
# 神经元在界面上只占 30px，传全分辨率是纯浪费：每个神经元 100 字节就够了。
NEURON_DENSITY = DENSITY // 10


def _serialized(method):
    """把 Engine 的公开方法串行化 —— 理由见 `Engine.__init__` 里那把锁。"""

    @functools.wraps(method)
    def wrapper(self: Engine, *args, **kwargs):
        with self._lock:
            return method(self, *args, **kwargs)

    return wrapper


@dataclass
class Losses:
    train: float
    test: float


@dataclass
class Batch:
    """一份数据，既保留了画点用的原始坐标，也备好了喂给模型的特征矩阵。"""

    points: list[Point]
    features: Tensor  # (N, F)
    labels: Tensor  # (N, 1)

    def __len__(self) -> int:
        return self.features.shape[0]


class Engine:
    """一步步训练一个网络，并把每一步的状态整理成可绘制的形式。"""

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.adapters = default_adapter_registry()

        self.cfg: Config | None = None
        self.model: torch.nn.Module | None = None
        self.optimizer: torch.optim.Optimizer | None = None

        self.iter = 0
        self.losses = Losses(0.0, 0.0)

        self.train: Batch | None = None
        self.test: Batch | None = None

        # L1 冻结掉的连接：参数 id → 布尔掩码
        self._dead: dict[int, Tensor] = {}
        # 一次权重更新前的权重快照，用来判断哪些连接跨过了零点
        self._before_update: list[tuple[Tensor, Tensor]] = []

        self._grid_xy_cache: Tensor | None = None
        # 这个引擎实例被 reset 过几次。用来给权重初始化派生种子 —— 见 reset()。
        self._reset_count = 0

        # 串行化所有公开操作。
        #
        # 服务端把训练放在线程里跑（`asyncio.to_thread`，为了不阻塞事件循环），
        # 而消息处理留在事件循环里 —— 两者都会碰同一个 Engine。没有这把锁时，
        # 两条前向传播会交错，`Tracer` 的 forward hook 互相串味：网格快照里
        # 会混进 batch_size 大小的**训练批次**，于是 `_as_grid` 报
        # "网格点数 10 与 density=100 不符"（10 正好是默认 batch size）。
        #
        # 实测：训练线程 + 主线程并发取 60 次快照，50 次失败。加锁后 0 次。
        #
        # 用可重入锁是因为公开方法之间会互相调用（step → _refresh_grids、
        # snapshot → _dead_masks_for）。单线程使用时几乎没有开销。
        self._lock = threading.RLock()

    # --- 配置 ---------------------------------------------------------------

    @_serialized
    def configure(self, cfg: Config) -> None:
        """应用新配置。

        数据只在**影响数据的字段**变化时重新生成 —— 调学习率不该把你训到一半的
        结果清掉，原版也是这个行为。
        """
        cfg = self._clamp_features(cfg)

        previous = self.cfg
        self.cfg = cfg

        data_changed = previous is None or self._data_fingerprint(
            previous
        ) != self._data_fingerprint(cfg)
        model_changed = previous is None or self._model_fingerprint(
            previous
        ) != self._model_fingerprint(cfg)

        if data_changed:
            self._regenerate_data()
        if data_changed or model_changed:
            self.reset()

    @staticmethod
    def _clamp_features(cfg: Config) -> Config:
        """把输入特征限制在当前数据集允许的范围内。

        从二维数据集切到一维拟合时，原来选中的 (x, y, x²) 里只有 x 还有意义
        —— 剩下那些的输入列上根本不存在对应的方块。不夹住的话模型会拿到
        in_dim=3 而输入列只画 1 个方块，索引立刻对不上。
        """
        available = DATASETS[cfg.dataset].features
        kept = tuple(f for f in cfg.features if f in available)
        if kept:
            return cfg if kept == cfg.features else cfg.replace(features=kept)

        # 一个都不剩就回落到默认：一维就是它唯一的那个，二维是前两个
        spec = DATASETS[cfg.dataset]
        return cfg.replace(features=available if spec.is_1d else available[:2])

    @staticmethod
    def _data_fingerprint(cfg: Config) -> tuple:
        return (cfg.dataset, cfg.noise, cfg.perc_train, cfg.seed, cfg.features)

    @staticmethod
    def _model_fingerprint(cfg: Config) -> tuple:
        return (cfg.model, cfg.features, cfg.activation, cfg.network_shape)

    def _regenerate_data(self) -> None:
        assert self.cfg is not None
        split = generate(
            self.cfg.dataset,
            noise=self.cfg.noise,
            perc_train=self.cfg.perc_train,
            seed=self.cfg.seed,
        )
        self.train = self._make_batch(split.train)
        self.test = self._make_batch(split.test)
        # 换了数据就重置初始化计数：这样"新数据集 + 同样配置"仍然可复现，
        # 而不是接着上一个数据集的计数往下走。
        self._reset_count = 0

    def _make_batch(self, points: list[Point]) -> Batch:
        assert self.cfg is not None
        xs, ys, labels = points_to_xy(points)
        return Batch(
            points=points,
            features=feature_matrix(xs, ys, self.cfg.features),
            labels=labels,
        )

    # --- 生命周期 -----------------------------------------------------------

    @_serialized
    def reset(self) -> None:
        """重新随机初始化权重、清空优化器状态。

        **不重新生成数据** —— 原版的 reset 也是这个行为。想换一批数据请改
        noise / 数据集 / 训练集比例 / 种子。

        权重初始化由 `(seed, 结构, 第几次 reset)` 派生 —— 这一点比原版多做了一步。
        原版的权重直接用未播种的 Math.random()，于是"同样的配置跑两次"会得到
        两条不同的曲线，没法做"同数据同初始化、只改学习率"的对照实验。
        这里让**新建引擎的第一次 reset 可复现**，而反复按 Reset 仍然能得到
        不同的初始化（那正是 Reset 按钮的意义）。
        """
        assert self.cfg is not None
        spec = self.registry.get(self.cfg.model)
        if spec is None:
            raise KeyError(f"模型 {self.cfg.model!r} 不在注册表里")

        torch.manual_seed(self._init_seed())

        # 分类和回归都只输出 1 维：分类的标签是 ±1，用 tanh 输出到 (-1,1)。
        self.model = spec.build(len(self.cfg.features), 1, self.cfg)
        self.model.to(self.cfg.device)

        self.optimizer = self._make_optimizer()
        self._dead = {id(p): torch.zeros_like(p, dtype=torch.bool) for p, _ in self._regularized()}
        self.iter = 0
        self._reset_count += 1

        # 训练开始前先把「定义域会漂移的层」的网格对齐到真实数据。
        # 对 KAN 是**必须**的：默认网格是 [-1,1] 而 Playground 数据在 [-6,6]，
        # 不先对齐的话 spline 分支全程恒为 0，你看到的是个"假 KAN"。
        if self.train is not None:
            self._refresh_grids(self.train.features)

        self.losses = self.evaluate()

    # --- 定义域对齐（KAN 等）-------------------------------------------------

    def _refresh_grids(self, x: Tensor) -> None:
        """把实现了 `update_grid` 的层的网格重新对齐到当前数据分布。

        钩子名 `update_grid` 是约定好的（见 layers.py 的模块文档）。
        用 forward pre-hook 来投递"该层自己的输入"是刻意的 ——
        逐层手工传下去的话，第二层的输入必须用第一层**更新后**的网格算出来，
        否则对齐是错的。pre-hook 天然满足这个顺序：
        L1 更新网格 → L1 前向（用新网格）→ L2 更新网格 → L2 前向 → …
        和参考实现的 `for layer: layer.update_grid(x); x = layer(x)` 一致。
        """
        if self.model is None:
            return
        targets = [m for m in self.model.modules() if callable(getattr(m, "update_grid", None))]
        if not targets:
            return

        handles = [m.register_forward_pre_hook(_update_grid_hook) for m in targets]
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                self.model(x)
        finally:
            for handle in handles:
                handle.remove()
            self.model.train(was_training)

    def _init_seed(self) -> int:
        """从配置派生一个稳定的整数种子。

        用 crc32 而不是内置的 hash()：str 的 hash 受 PYTHONHASHSEED 影响，
        换个进程就变了，那样"可复现"就只是本进程内可复现。
        """
        assert self.cfg is not None
        key = repr((self._model_fingerprint(self.cfg), round(self.cfg.seed, 6), self._reset_count))
        return zlib.crc32(key.encode("utf-8"))

    def _make_optimizer(self) -> torch.optim.Optimizer:
        assert self.model is not None and self.cfg is not None
        params = [p for p in self.model.parameters() if p.requires_grad]
        lr = self.cfg.learning_rate
        if self.cfg.optimizer == "adam":
            return torch.optim.Adam(params, lr=lr)
        if self.cfg.optimizer == "momentum":
            return torch.optim.SGD(params, lr=lr, momentum=0.9)
        return torch.optim.SGD(params, lr=lr)

    # --- 训练 ---------------------------------------------------------------

    @_serialized
    def step(self) -> None:
        """跑一个 epoch（完整遍历一遍训练集）。"""
        assert self.model is not None and self.optimizer is not None and self.cfg is not None
        assert self.train is not None

        x, y = self.train.features, self.train.labels
        n = x.shape[0]
        batch = max(1, self.cfg.batch_size)
        use_l1 = self._regularization_active() and self.cfg.regularization == "l1"

        self.model.train()
        # 尾部不足一个 batch 的点直接丢弃。原版会把它们的梯度累积到下一个 epoch，
        # 那是实现的副作用而非设计意图，这里不复刻。
        for start in range(0, n - batch + 1, batch):
            self.optimizer.zero_grad(set_to_none=True)
            out = self.model(x[start : start + batch])
            loss = 0.5 * (out - y[start : start + batch]).pow(2).mean()
            loss.backward()
            self._add_regularization_grad()

            if use_l1:
                self._before_update = [
                    (p, p.detach().clone()) for p, _ in self._regularized()
                ]
            self.optimizer.step()
            if use_l1:
                self._kill_crossed_links()

        self.iter += 1

        # 激活值会随训练漂移，定期把网格重新对齐 —— 否则 spline 分支
        # 会慢慢失效，而图上完全看不出来（φ 只是悄悄变回 silu）。
        every = self.cfg.grid_update_every
        if every > 0 and self.iter % every == 0:
            self._refresh_grids(x)

        self.losses = self.evaluate()

    def _regularized(self) -> list[tuple[Tensor, str]]:
        """参与正则化的参数。

        只挑权重矩阵（ndim >= 2），不碰 bias —— 原版也只对连接（link）做正则，
        不对神经元偏置做。
        """
        assert self.model is not None
        return [
            (p, name)
            for name, p in self.model.named_parameters()
            if p.requires_grad and p.dim() >= 2
        ]

    def _dead_masks_for(self, module: torch.nn.Module) -> dict[str, Tensor]:
        """给 introspection 用的查询：这个模块有哪些参数被 L1 冻死了。

        只回传"确实有死连接"的参数，避免每帧给每个模块都建一堆空掩码。
        """
        masks: dict[str, Tensor] = {}
        for name, param in module.named_parameters(recurse=False):
            mask = self._dead.get(id(param))
            if mask is not None and bool(mask.any()):
                masks[name] = mask
        return masks

    def _regularization_active(self) -> bool:
        assert self.cfg is not None
        return self.cfg.regularization != "none" and self.cfg.regularization_rate != 0

    def _add_regularization_grad(self) -> None:
        """把正则项加进 loss。

        加进 loss 而不是手动改权重，是为了让 Adam 这类自适应优化器也能正确处理它。
        对纯 SGD 而言两者数学等价：`w -= lr*grad + lr*regRate*∂reg/∂w`。
        """
        assert self.cfg is not None
        if not self._regularization_active():
            return

        total: Tensor | None = None
        for param, _ in self._regularized():
            term = param.abs().sum() if self.cfg.regularization == "l1" else 0.5 * param.pow(2).sum()
            total = term if total is None else total + term

        if total is not None:
            (total * self.cfg.regularization_rate).backward()

    def _kill_crossed_links(self) -> None:
        """L1 的"过零即杀死"规则 —— 原版稀疏化的来源。

        原版判断的是 `oldWeight * newWeight < 0`，即"这一步更新把权重推到了符号的
        另一侧"。但这个式子漏掉了最常见的情况：**权重正好落到 0**。
        一旦落到 0，L1 在 0 处的次梯度是 0，权重就永远卡在 0 了 ——
        图上看它还在（细线、灰色），实际上已经死了。

        所以这里补上"从非零落到零"也算死亡，两种情形一起判。

        实现上是"置零 + 冻结"，而不是真的从计算图里删掉连接：
        梯度仍然会流过权重为 0 的连接，但权重不再变化，效果等价。
        """
        for param, before in self._before_update:
            mask = self._dead.setdefault(id(param), torch.zeros_like(param, dtype=torch.bool))
            with torch.no_grad():
                crossed = (before * param < 0) | ((before != 0) & (param == 0))
                mask |= crossed
                param[mask] = 0.0
        self._before_update = []

    # --- 评估 ---------------------------------------------------------------

    @_serialized
    def evaluate(self) -> Losses:
        assert self.model is not None
        self.model.eval()
        with torch.no_grad():
            return Losses(
                train=self._loss_on(self.train),
                test=self._loss_on(self.test),
            )

    def _loss_on(self, batch: Batch | None) -> float:
        if batch is None or len(batch) == 0:
            return 0.0
        out = self.model(batch.features)
        return float(0.5 * (out - batch.labels).pow(2).mean())

    # --- 可视化 -------------------------------------------------------------

    def _grid_base(self) -> Tensor:
        """采样网格的 (x, y) 坐标，形状 (N, 2)。

        - 二维：100×100 的方格（原版的决策边界分辨率）
        - 一维：定义域上的一行点，**y 填 0**

        一维这里把 y 补成 0 是为了让下游完全不用分支：所有特征函数都是
        `f(x, y)` 的形式，x 这一行的特征函数只用第一个参数，y 取什么无所谓。
        于是 `feature_matrix`、`build_graph`、模型全都原样复用。
        """
        assert self.cfg is not None
        spec = DATASETS[self.cfg.dataset]
        if spec.is_1d:
            x = grid_line(spec.domain)
            return torch.stack([x, torch.zeros_like(x)], dim=-1)

        if self._grid_xy_cache is None:
            self._grid_xy_cache = torch.tensor(grid_points(DENSITY), dtype=torch.float32)
        return self._grid_xy_cache

    def _grid_input(self) -> Tensor:
        """喂给模型的那份网格输入，只含**当前选中的**特征。"""
        assert self.cfg is not None
        base = self._grid_base()
        return feature_matrix(base[:, 0], base[:, 1], self.cfg.features)

    def _grid_all_features(self) -> Tensor:
        """数据集允许的**全部**特征在网格上的取值。

        输入列要把没选中的特征也画出来（它们是可点击的开关），
        而那些特征的取值不在前向传播的路径上，只能在这儿单独算一份。
        一维数据集只有 "x" 一个可用特征，所以这一列只有一个方块。
        """
        base = self._grid_base()
        return feature_matrix(base[:, 0], base[:, 1], self._available_features())

    def _available_features(self) -> tuple[str, ...]:
        assert self.cfg is not None
        return DATASETS[self.cfg.dataset].features

    @_serialized
    def snapshot(self) -> tuple[DisplayGraph, dict]:
        """跑一遍网格前向，产出这一帧的图和热力图。

        一次前向同时做两件事：hook 记录结构 → 画网络图；拿到每层激活值 → 画神经元
        热力图。所以每帧模型只跑一次。

        用 eval 模式跑：这样 Dropout 不随机、BatchNorm 用滑动统计量，
        可视化结果是确定的（否则每隔一帧热力图就会抖一下）。
        """
        assert self.model is not None and self.cfg is not None
        self.model.eval()

        # 输入列画数据集允许的全部特征，但模型只吃选中的那些。这里算的是
        # "模型的第 k 个输入 → 显示列里的第几个方块" 的映射，
        # 顺序必须和 cfg.features 一致（那是模型的列顺序）。
        available = self._available_features()
        display_index = {fid: i for i, fid in enumerate(available)}
        graph = build_graph(
            self.model,
            self._grid_input(),
            [FEATURES[fid].label for fid in available],
            [display_index[fid] for fid in self.cfg.features],
            self.adapters,
            input_grid=self._grid_all_features(),
            dead_lookup=self._dead_masks_for,
        )

        payload = {
            "iter": self.iter,
            "lossTrain": self.losses.train,
            "lossTest": self.losses.test,
        }
        if self.is_1d:
            # 一维：每列的激活就是一条 (N,) 的曲线，不需要降采样
            payload["curves"] = [a.detach() for a in graph.activations]
        else:
            heatmaps = [_downsample(a, DENSITY, NEURON_DENSITY) for a in graph.activations]
            # 输出列保留全分辨率 —— 它就是主决策边界图。
            heatmaps[-1] = _as_grid(graph.activations[-1], DENSITY)
            payload["heatmaps"] = heatmaps
        return graph, payload

    @property
    def is_1d(self) -> bool:
        assert self.cfg is not None
        return DATASETS[self.cfg.dataset].is_1d


def _update_grid_hook(module: torch.nn.Module, args: tuple) -> None:
    """给实现了 `update_grid` 的层投递它自己的输入。

    输入可能带着前导维度（用户模型可能先做了 reshape），统一压成
    (batch, features) 再交给层 —— update_grid 只认这个形状。
    """
    if not args or not isinstance(args[0], Tensor):
        return
    x = args[0].detach()
    if x.dim() < 2:
        return
    module.update_grid(x.reshape(-1, x.shape[-1]))


# --- 张量 → 网格 -------------------------------------------------------------


def _as_grid(activations: Tensor, density: int) -> Tensor:
    """(N, width) → (width, density, density)。

    网格点的生成顺序是行优先（j 外层、i 内层），所以直接 reshape 就是一个
    行 0 在最上方的矩阵，前端可以当图像直接用。
    """
    total, width = activations.shape
    assert total == density * density, f"网格点数 {total} 与 density={density} 不符"
    return activations.reshape(density, density, width).permute(2, 0, 1).contiguous()


def _downsample(activations: Tensor, density: int, target: int) -> Tensor:
    """把 (N, width) 的激活值降采样成 (width, target, target)。

    对应原版的 reduceMatrix：按 (density/target)² 的块取平均。
    """
    grid = _as_grid(activations, density)
    factor = density // target
    width = grid.shape[0]
    return grid.reshape(width, target, factor, target, factor).mean(dim=(2, 4)).contiguous()
