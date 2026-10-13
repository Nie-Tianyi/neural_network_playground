# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目是什么

PyTorch 版的 TensorFlow Playground：在浏览器里实时可视化神经网络的训练过程（每个神经元的激活热力图、权重连线、决策边界、loss 曲线）。与原版的核心差异是**网络结构由用户自己在 `playground/models.py` 里用 PyTorch 定义**，而不是从固定几种结构里挑 —— 所以整个可视化层必须能处理任意 `nn.Module`。

## 常用命令

```bash
uv run main.py                 # 启动，自动开浏览器（--no-browser / --port 8080）
uv run pytest                  # 全部测试（约 20 秒）
uv run pytest -k "l1 or prune" # 按关键字筛选
uv run pytest tests/test_playground.py::test_first_layer_edges_point_at_the_active_nodes  # 单个测试
```

没有配置 linter / formatter。

### 改完前端必须截图看一眼

**后端测试全绿不代表页面是对的。** 这个项目里最容易出错的恰恰是"画成什么样"，而 pytest 完全覆盖不到。已发生过的实例：改一下学习率就断连、噪声滑条实际是二值的、神经网络图整体变形、权重挂到了错误的边上 —— 这些全是靠截图或 `--dump-dom` 发现的。

```bash
# 截图（用系统已装的 Chrome，不需要 Playwright）
"/c/Program Files/Google/Chrome/Application/chrome.exe" --headless --disable-gpu \
  --no-sandbox --hide-scrollbars --window-size=1000,760 \
  --virtual-time-budget=20000 --screenshot=/tmp/shot.png \
  "http://127.0.0.1:8000/?dataset=spiral&shape=8,8&features=x,xSquared,sinY&play=1"

# 抓渲染后的真实 DOM —— 排查 SVG 几何问题时比截图更好用
... --dump-dom "http://127.0.0.1:8000/" > dom.html
```

URL 参数：`dataset` / `model` / `shape`（如 `8,8`）/ `features`（如 `x,xTimesY`）/ `activation` / `lr` / `play`。

**已知的工具坑**：`--virtual-time-budget` 下 WebSocket 偶尔连不上，截图会是空白页（只有顶部框架）。重试一两次即可，不是应用的问题。判据是文件大小：正常 150–350KB，空白约 35KB。

## 架构

### 数据流

```
用户写的 nn.Module
  ↓  engine.snapshot() 跑一次网格前向；Tracer 用 forward hook 记录每个叶子模块的输入输出
DisplayGraph（列 + 边 + 激活张量）        ← introspection.build_graph()
  ↓  serialize.py：热力图量化成 uint8 + base64
  ↓  server.py：WebSocket "frame" 消息
canvas 绘制                              ← web/app.js + web/render/*
```

**一次网格前向同时产出网络结构和神经元热力图**，不要为了两件事跑两遍模型。

### introspection.py 是支点

把一次前向传播翻译成图结构。四个关键机制：

1. **角色三分**：每个叶子模块是 `WEIGHT`（产生新的一列节点 + 连接上一列的边）/ `ELEMENTWISE`（就地变换当前列的值）/ `IGNORE`。**列只由 WEIGHT 模块产生。**

2. **列的显示值来自"激活之后"**：Playground 里"神经元的输出"指 `activation(weighted_sum)`，而 `nn.Linear` 的输出只是加权和。所以 `_resolve_display_record` 会从权重层出发，沿**消费了上一个输出的逐元素层**往下走，直到链尾。追踪靠**张量对象同一性**（`nn.Sequential` 直接传引用），遇到另一个权重层就停 —— 分支结构（残差）因此会退化，这是有意为之，不猜。

3. **`_promote_width_changing`**：形状会变的"逐元素层"（`nn.GLU` 把 6 维变 3 维、`nn.Flatten`）其实**产生了新的一组值**，必须提升成独立一列，否则它前面那一列的宽度会算成它的输出宽度，边也全对不上。判断依据是真实张量形状，不是层类型。

4. **认不出来就降级，绝不抛异常**。`OpaqueFallbackAdapter` 按"有 2 维以上 weight 参数 → 当权重矩阵画"、"没有参数 → 当逐元素层"、"都不满足 → 只画节点不画边"三级猜测。这是"用户随便写什么怪结构都不白屏"这个承诺的兑现处。

### ⚠️ 最大的坑：显示宽度 ≠ 模型宽度

这是本项目最容易出错、且**错误极难肉眼发现**的地方，已经因此产生过两个 bug。

**输入列画的是全部 7 个特征**（未选中的也画，虚线淡显，因为原版的交互就是"点最左列方块开关特征"，藏起来就没得点）。但模型只吃其中选中的那几个。于是：

- **铺边的入边宽度**必须用模型的实际输入列数（`model_widths[-1]`），不能用 `columns[-2].width`（显示列宽 = 7）。
  用错的表现：总边数恰好还是对的（7+3 = 10 条），但前几个神经元各收到 7 条和 3 条、后面的**一条入边都没有**，源点还指向了没选中的特征。
- **模型输入下标 ≠ 显示下标**。特征的选择顺序决定模型输入列的顺序：在页面上关掉 X₁ 再打开，它会追加到末尾，于是模型第 0 列变成 X₂，而显示列里 X₂ 排第 2。所以第一层的 `src` 必须过一次 `src_map`（`engine.snapshot()` 里算的 `display_index` 映射）。
  这个更隐蔽：**边数和连线位置看起来完全正常，只有权重值是错的**。

`model_widths` 和 `columns[i].width` 是刻意分开跟踪的两个量，不要合并。

**写这类测试时要查"边的着落点"而不是"边数"** —— 数量对得上完全不能说明索引是对的。参考 `test_every_hidden_neuron_receives_edges_from_all_selected_features` 和 `test_edges_follow_feature_order_not_display_order`。

### adapters/ 是唯一的扩展点

加一种新结构（KAN、多项式层、RBF……）只需要写一个 adapter，不用碰引擎、协议或前端渲染。前端只认 `EdgePayload.kind`：

| payload | 用于 | 渲染 |
|---|---|---|
| `ScalarWeight` | `nn.Linear` | 线宽 = `clamp(\|w\|,0,5)→[1,10]px`，颜色 = 符号 |
| `FunctionCurve` | KAN 的边（一个可学习的一元函数） | 连线中点的迷你曲线；**透明度**编码重要性，线宽不变 |
| `ProductInteraction` | 多项式项（x₁·x₂） | 显式「×」节点 |
| `Opaque` | 认不出的层 | 灰虚线 |

`Registry` 按 `priority` 降序匹配，所以更具体的 adapter（KANLinear）自然覆盖通用的那个。`describe()` 抛异常会被 `Registry.describe` 兜住并降级成 opaque。

`DescribeContext` 里有三样 adapter 自己拿不到、必须从外面传的东西：`input_range`（KAN 画曲线要用该层输入的**实测范围**）、`dead_masks`（只有引擎知道哪些连接被 L1 冻死了）。

### engine.py 不依赖 Web

`Engine` 是纯 Python 类（`reset()` / `step()` / `snapshot()`），故意不碰 fastapi。好处：「多实验对比」= 同时持有多个 Engine 同步推进；能在 Jupyter 里直接用；能脱离浏览器做单测。

几个**照抄原版、不要"顺手优化"**的行为：
- 1 epoch = 完整遍历训练集一遍；loss = `mean(0.5·(ŷ-y)²)`
- **数据顺序在 epoch 之间固定**（原版只在生成数据时打乱一次，不每轮重打乱）
- **L1 的"过零即杀死"规则**：权重被推过零点就置 0 并永久冻结，这是原版稀疏化的来源。注意原版的判断式 `old*new < 0` 漏掉了"正好落到 0"的情况（落到 0 后 L1 次梯度是 0，权重永远卡住），这里补上了
- 权重初始化 `U(-0.5,0.5)` 所有层同一分布、bias 恒 0.1
- **但初始化做了播种**（原版用未播种的 `Math.random()`，导致同配置重跑曲线不一样，没法做对照实验）。种子由 `(seed, 结构, 第几次 reset)` 派生：新建引擎首次可复现，反复按 Reset 仍换一套权重

网格前向用 `eval()` 模式（Dropout 不随机、BN 用滑动统计量），否则热力图每隔一帧就会抖一下。

### registry.py 热重载

**不能用 `importlib.reload`** —— 它就地重置已有模块，旧类还挂在别处引用上，会出现"改了代码但注册表还是老模型"甚至同一个 key 注册出两份。这里每次都用 `module_from_spec` + `exec_module` 建**全新的模块对象**，从一个干净的命名空间里重新收集。

注册方式是给类挂 `__playground_model__` 标记（不是注册到全局单例），所以没有全局可变状态。

模块名必须是 `playground` 的**直接子模块**（`playground._loaded_xxx`），不能再多一层，否则用户文件里的 `from .config import ...` 会解析到不存在的包上。

`builtin_models.py` 和 `models.py` 分成两个文件是有意的：用户文件写到一半保存（语法错误）时，内置模型仍然可用，页面不会空掉。

### 协议层

`server.py` 的 `_config_from_payload` 是前端 camelCase 和 Python snake_case 之间的**唯一**桥梁，`_FIELDS` 表驱动。曾经这里的 bug 是字段名直接透传给 `Config.replace()`，于是前端一动下拉框就是 TypeError、整条连接断掉。

**约定的单位**：`noise` 在协议里一律是归一化的 `0–0.5`，界面上那个 `0–50` 的滑条是前端的显示单位，由前端自己换算。不要把两种单位混进协议 —— 歧义比错误更危险。

服务端异常**必须同时 `log.exception`**，不能只发给浏览器：否则"改学习率就断连"这种 bug 在终端里完全看不见。

## 复刻规格的来源

`datasets.py` / 配色 / 几何参数里的魔数全部来自 TensorFlow Playground 的源码（`src/dataset.ts`、`heatmap.ts`、`playground.ts`、`nn.ts`），注释里标了出处。**这些不是随手写的，改之前先确认是不是有意为之**：

- 数据集噪声的施加方式**不统一**：Circle/XOR/Gaussian/Plane 的噪声只让标签出错（点坐标干净），Spiral 直接给坐标加噪
- 配色 `#f59322` 橙 / `#e8eaeb` 灰 / `#0877bd` 蓝，30 档量化，热力图 alpha = 160
- 主决策边界 100×100 格点；神经元缩略图 10×10（由 100×100 按 10×10 块平均降采样）
- 控制项默认值：学习率 0.03、tanh、batch 10、50% 划分
- 最多 6 隐藏层、每层 1–8 个神经元

README 的「相对原版的取舍」一节列了所有**有意偏离**及其理由，改这些行为前先看那里。

## 写代码的约定

- 注释用中文，风格是"解释为什么这么做"，尤其是那些**看起来可以简化但实际不能**的地方。不要写复述代码的注释。
- 常量尽量在注释里带上"为什么是这个值"。
- `playground/models.py` 是**给用户写的文件**，改它之前先确认不是要破坏用户内容。
- 前端不硬编码任何选项清单 —— `config.py` 的 `OPTIONS` 是下拉框的唯一数据源。
- 前端所有渲染都挂在 `web/render/` 下，`app.js` 只做 WebSocket 和控件绑定。
- `web/render/network.js` 里 SVG 的 `viewBox` 必须和元素的 CSS 尺寸**严格相等**（1:1 映射）。不要用 `svg.clientWidth` 量宽度 —— SVG 元素上这个属性在 Chrome 里恒为 0；也不要给宽度加"最小下限"，一旦超过容器实际宽度就会撑破容器导致整页横向滚动。
- **同一列等距分布**：网络图里所有列（输入 / 隐藏层 / 输出）在 `[INPUT_X, 右缘]` 上等距铺开，一套公式管到底。历史上这里分成两段算过（输入列钉常量、隐藏层铺另一个区间），结果相邻列间距完全取决于层数，两层隐藏层时首尾各挤 50px、中间空 500px。
- **超宽屏的内容居中靠 `padding-inline`，不是 `margin: auto`**：`header` / `#top-controls` / `#main-part` 共用 `padding-inline: max(--page-pad, (100% - --page-width) / 2)`。这样背景色和底部分隔线仍然通铺满宽、只有内容居中。改成 `max-width + margin: auto` 会让超宽屏上出现一条悬空的、和内容同宽的色带。
