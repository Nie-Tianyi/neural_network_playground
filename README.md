# PyTorch Neural Network Playground

在浏览器里**实时看神经网络训练过程**的教学工具，模仿 [TensorFlow Playground](https://playground.tensorflow.org)，
但网络结构用 PyTorch 定义 —— 想写什么结构就写什么结构。

```
┌─ DATA ──────────┐ ┌─ FEATURES ──────────────────────┐ ┌─ OUTPUT ────────┐
│ Circle  XOR     │ │  X₁ ─┐                          │ │ Test loss 0.012 │
│ Gaussian Spiral │ │  X₂ ─┤ ╲  ╱╲  ╱╲              │ │ Train loss 0.008│
│ Plane  Multi-G  │ │  X₁² ┤  ╳  ╳  ╳  → y           │ │ ╱╲___ 曲线      │
│                 │ │  X₁X₂┤ ╱  ╲╱  ╲╱               │ │ [决策边界热力图] │
│ 训练比例 / 噪声  │ │      │  每个方块 = 该神经元的激活 │ │ [橙←→蓝 色标]   │
│ Batch size      │ │      │  每条线   = 一个权重      │ │                 │
└─────────────────┘ └─────────────────────────────────┘ └─────────────────┘
```

---

## 快速开始

```bash
uv run main.py
```

浏览器会自动打开 <http://127.0.0.1:8000>。`--no-browser` 不自动开，`--port 8080` 换端口。

---

## 写你自己的网络

打开 [playground/models.py](playground/models.py)，写一个普通的 `nn.Module`：

```python
from .registry import register
import torch.nn as nn
from .config import make_activation

@register("my-net", title="我的网络")
class MyNet(nn.Module):
    def __init__(self, in_dim, out_dim, cfg):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 8)
        self.act = make_activation(cfg.activation)
        self.fc2 = nn.Linear(8, out_dim)

    def forward(self, x):
        return self.fc2(self.act(self.fc1(x)))
```

**保存文件就行，不用重启、不用刷新页面。** 哪怕正在训练，下一个动画帧新模型就会出现在下拉框里。

构造函数固定收三个参数：`in_dim`（当前勾选的特征数）、`out_dim`（输出维度，分类回归都是 1）、
`cfg`（超参对象，读 `cfg.activation` / `cfg.learning_rate` / `cfg.regularization` 等）。

### 可视化是自动的

你不需要为可视化做任何事。引擎在前向传播时挂 hook，自己判断每一层怎么画：

| 你的层 | 画成什么 |
|---|---|
| `nn.Linear` | 边 = 粗细/颜色（权重矩阵） |
| `KANLinear` | 边 = **一条函数曲线**（见下） |
| 激活 / Dropout / BatchNorm | 就地变换，不单独占一列 |
| 认不出来的层 | **节点热力图照画，边退化成灰虚线** |

所以最差情况也只是画得不精细，不会白屏 —— 你在 `__init__` 里写什么怪结构都行。

如果想让自定义层画出更丰富的东西，有两个可选约定：

```python
class MyLayer(nn.Module):
    def explain(self, samples=48) -> MyExplanation:
        """返回 hooks 拿不到的信息（边的函数、门控值……），
        再配一个 adapter 就能画出任意形状的边。见 playground/layers.py。"""

    def update_grid(self, x):
        """定义域会随训练漂移的层实现它：引擎会在训练开始前、以及每
        grid_update_every 个 epoch 调用一次，参数是**该层自己的输入**。"""
```

---

## KAN：边是一个函数，不是一条线

MLP 的一条边只有"多粗、什么颜色"两个信息量。KAN（Kolmogorov-Arnold Network）
把非线性从**节点**挪到了**边**上：

```
nn.Linear:   y_j = act( Σ_i w_ji · x_i )      ← act 在节点上
KANLinear:   y_j = Σ_i φ_ji( x_i )            ← φ 在边上，节点只做求和
```

每条边 `φ` 本身是一个可学习的一元函数（`φ(x) = w_b·silu(x) + w_s·ΣcᵢBᵢ(x)`），
所以图上直接把它画成曲线 —— 你可以看到每条边长什么样、在哪一段陡、哪一段平。
整个 `[2, h, 1]` 网络也就 `3h` 条曲线，能全部同时画出来。

**颜色编码**：曲线的**透明度** ∝ 重要性 `|φ|₁`（论文 Eq.2.17），
**线宽固定** —— 这是参考实现的做法，别改成用线宽表示重要性，两个信息挤在一个通道里会都看不清。
整层共用一个纵向标度，所以"又平又淡"就准确表达了"这条边没在干活"。

### 两个会让 KAN 静默失效的坑

**1. 数据跑出 spline 网格**　B-spline 基函数在网格范围外**恒为 0**，
此时 `φ(x)` 退化成 `w_b·silu(x)`，spline 分支彻底死掉。
而 KANLinear 的默认网格是 `[-1,1]`，Playground 的数据却在 `[-6,6]` ——
不处理的话你看到的是个"假 KAN"，所有边都长得像 silu，怎么训练都不变形。

引擎在训练开始前会自动调用 `update_grid()` 把网格对齐到数据分位数，
每 `grid_update_every` 个 epoch 再对齐一次（激活会随训练漂移）。

**2. `update_grid` 会衰减 spline 分支**　上游参考实现的一个 bug：它用乘过
`spline_scaler` 的曲线去拟合没乘 scaler 的系数，而前向又会再乘一次 scaler，
于是**每调用一次 spline 分支就被整体乘上一个 |scaler|**。实测调用 40 次后
`spline_weight` 从 1e-8 掉到 1e-24 —— 症状和坑 1 一模一样（退化成 silu），
但成因完全不同。参考实现里没暴露，是因为他们一辈子只调十几次网格、且那时
`scaler ≈ 1`；我们每 20 步调一次，是指数累积的。`tests/test_kan.py` 里有专门钉这条的测试。

---

## 页面上有什么

**数据集**　Circle / XOR / Gaussian / Spiral（分类）+ Plane / Multi-Gaussian（回归）。
噪声滑条的施加方式按原版：Circle/XOR/Gaussian 只让**标签**出错，Spiral 则直接**把点撒开**。

**特征**　点网络图最左列的方块开关。7 个特征全画出来，没选中的显示为虚线淡框
（能预览它长什么样，再决定要不要打开）。这 7 个里的选择直接决定模型看到的世界。

**网络**　每个方块是一个神经元，上面的小图是它在整个输入平面上的激活值。
连线粗细/颜色是权重。**点一个神经元的方块**可以把主热力图切过去看它。

**输出**　决策边界 + 训练/测试 loss 曲线。

**超参**　模型、学习率、激活函数、优化器、正则化类型与强度。

---

## 引擎能脱离浏览器用

`Engine` 不依赖 Web，可以直接在脚本或 Jupyter 里跑：

```python
from playground.registry import default_registry
from playground.engine import Engine
from playground.config import Config

engine = Engine(default_registry())
engine.configure(Config(dataset="spiral", features=("x", "y"), network_shape=(8, 8)))
for _ in range(500):
    engine.step()
print(engine.losses)          # Losses(train=0.03, test=0.04)

graph, payload = engine.snapshot()   # 可绘制的图结构 + 热力图张量
```

---

## 相对原版的取舍

### 照抄的部分

原版的行为细节决定了实验结论，所以这些不是"差不多就行"：

- 数据集的采样半径、噪声施加位置、标签阈值、train/test 划分方式
- 配色（`#f59322` 橙 / `#e8eaeb` 灰 / `#0877bd` 蓝，30 档量化，热力图 alpha 160）
- 权重连线：线宽 `clamp(|w|,0,5) → [1,10]px`，线性不是 log
- 1 epoch = 完整遍历训练集一遍；loss = `mean(0.5·(ŷ-y)²)`；**数据顺序在 epoch 之间固定**
- 权重初始化 `U(-0.5, 0.5)` 所有层同一分布，bias 恒 0.1
- **L1 的"过零即杀死"规则**：权重被推过零点就置 0 并永久冻结（原版稀疏化的来源）
- 控制项取值与默认值（学习率 0.03、tanh、batch 10、50% 划分……）
- 分类输出层用 tanh、回归用 linear；最多 6 隐藏层、每层 1–8 个神经元

### 有意偏离的部分

| 偏离 | 原因 |
|---|---|
| 网格前向用批量 PyTorch | 原版每帧对 100×100=10000 个格点**逐个**做前向（纯 JS），是它最大的性能瓶颈。批量化后约 1ms |
| 神经元缩略图只传降采样后的 10×10 | 原版算了全分辨率再降采样。只传需要的部分，一帧约 15KB |
| 提供了 Adam / Momentum | 原版只有纯 SGD。默认仍是 SGD 以保持原版行为 |
| **权重初始化也播种了** | 原版用未播种的 `Math.random()`，同配置重跑曲线不一样，没法做对照实验。这里让新建引擎的首次初始化可复现，而反复按 Reset 仍会换一套权重 |
| **模型文件热重载** | 这是这个项目存在的理由 |
| **特征列始终画满 7 个** | 未选中的显示为虚线淡框。原版也是全画，藏起来就"没得点"了 |
| 缩略图 y 轴方向与主图一致 | 原版缩略图的 y 是反的，螺旋在主图和缩略图里是镜像的——那是无心之失，没复刻 |

---

## 项目结构

```
playground/
├── datasets.py        6 个数据集（逐条对照原版 dataset.ts）
├── config.py          超参 dataclass + 选项清单（前端下拉框的数据源）
├── registry.py        模型注册表 + 热重载
├── introspection.py   ★ 核心：一次前向 → 可绘制的图结构
├── adapters/          ★ 层 → 绘制指令的翻译器（加新结构只改这里）
├── engine.py          训练引擎（纯 Python，不依赖 Web）
├── serialize.py       状态 → JSON（热力图量化成 uint8 + base64）
├── server.py          FastAPI + WebSocket
├── builtin_models.py  内置 MLP
└── models.py          ★ 你写模型的地方
web/
├── app.js             WebSocket + 控件绑定
└── render/            network（网络图）/ heatmap / losschart / palette
```

### 协议

客户端发 `init` / `config` / `play` / `pause` / `step` / `reset` / `regenerate` / `reload_models`；
服务端回 `static`（选项、模型列表）/ `data`（数据点）/ `frame`（每步状态）/ `error`。

URL 参数可以直接指定初始状态，方便存书签：
`?dataset=spiral&shape=8,8&activation=relu&lr=0.1&play=1`

---

## 测试

```bash
uv run pytest
```

103 个测试，覆盖数据集的采样不变量、L1 剪枝强度、图结构与边数是否精确对应网络形状、
每条边的**着落点**（不只数边数）、未知模块的降级、协议字段映射、模型热重载
（含语法错误时的降级）、以及 KAN 那两个静默失效的坑。

前端渲染另有验证方式：无头浏览器截图 + 直接看画面。后端的数字"看起来对"并不代表图能画出来
—— 事实上"改一下学习率就断连"这种 bug 就是靠截图才发现的。

---

## 已知约束

- 分支结构（残差连接）不追求画得精确：连线按执行顺序走，遇到多个消费者会停在那里，不猜。
- 服务器是单会话的（一个 Engine 实例），适合本地单人用。
- 网格前向用 eval 模式：Dropout 不随机、BatchNorm 用滑动统计量，这样可视化结果是确定的
  （否则热力图每隔一帧就会抖一下）。
