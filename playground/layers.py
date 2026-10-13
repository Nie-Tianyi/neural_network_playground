"""非常规结构的层实现 —— KAN、高阶/多项式感知机等。

## 这些层为什么需要单独实现

MLP 里一条边就是一个标量权重；这些结构里，边的语义完全不同：

    MLP / nn.Linear       边 = 一个数 w           → 粗细 + 颜色就够了
    KAN                   边 = 一个函数 φ(x)       → 必须画成曲线
    高阶感知机            边 = 一个单项式 x₁x₂     → 必须画成乘法节点

所以可视化层不能假设"边是标量"。这里每提供一个层，就配套在 `playground/adapters/`
里提供一个 adapter 告诉引擎怎么画。

## 给自定义层做可视化的约定

写自己的层时，只要加一个 `explain()` 方法返回"hooks 拿不到的信息"
（边的函数、门控值……），就能获得完整可视化；不加也至少有激活热力图
（那是 forward hook 的功劳，与认不认识这个层无关）。

`update_grid()` 是一个另类的约定：**名字被引擎硬编码**，实现它的层会在
训练开始前和数据分布漂移时被自动调用一次，用来把自己的"定义域"对齐到真实数据。
KAN 需要它，理由见 KANLinear 的文档。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


# --- KAN ---------------------------------------------------------------------

@dataclass
class KANExplanation:
    """一层 KAN 的所有边函数，供可视化用。

    定义域是**每个输入特征各自**的 —— KAN 的 spline 网格逐输入特征独立，
    所以 X₁ 和 X₁² 的定义域不可能相同，曲线必须分开采样。

    `xs` 是唯一的定义域来源（形状 (in, samples)）：adapter 直接用它，
    不要自己重新 linspace 一遍，否则和 ys 对不上就是一条错位的曲线。
    """

    xs: Tensor  # (in, samples) 每条曲线的采样横坐标
    phi: Tensor  # (out, in, samples) φ(x) 总输出
    base: Tensor  # (out, in, samples) w_b·silu(x) 分支
    spline: Tensor  # (out, in, samples) w_s·ΣcᵢBᵢ(x) 分支
    importance: Tensor  # (out, in) 论文 Eq.2.17 的 |φ|₁

    @property
    def samples(self) -> int:
        return self.phi.shape[-1]


class KANLinear(nn.Module):
    """一层 KAN（Kolmogorov-Arnold Network）。

    和 `nn.Linear` 的区别只有一句话：**非线性放在边上，不放在节点上**。

        nn.Linear:   y_j = act( Σ_i w_ji · x_i )        ← act 在节点上
        KANLinear:   y_j = Σ_i φ_ji( x_i )              ← φ 在边上，节点只做求和

    每条边 φ 本身是一个可学习的一元函数（论文 Eq.2.10）：

        φ(x) = w_b · silu(x)  +  w_s · Σ_i c_i · B_i(x)
                └ base 分支 ┘    └── B-spline 分支 ──┘

    初始化时 `c_i ~ N(0, σ²)` 很小，所以 **φ 一开始长得像 silu**，
    训练过程就是这些曲线慢慢变形出结构 —— 这个"从 silu 长出来"的过程
    本身就是很好的教学画面。

    ## ⚠️ 必须在训练前把网格对齐到数据

    B-spline 基函数的定义是 `(x >= grid[:-1]) & (x < grid[1:])`，
    **在网格范围之外所有基函数恒为 0**，此时 φ(x) 退化成 `w_b·silu(x)`，
    spline 分支彻底死掉。

    默认网格是 [-1,1]，而 Playground 的数据在 [-6,6] —— 直接把数据喂进来，
    你看到的会是一个"假 KAN"：所有边都长得像 silu，怎么训练都不变形。

    所以引擎会在训练开始前调用 `update_grid()`（见模块文档），
    把网格重排到真实的数据分位数上。
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        grid_size: int = 5,
        spline_order: int = 3,
        scale_noise: float = 0.1,
        scale_base: float = 1.0,
        scale_spline: float = 1.0,
        enable_standalone_scale_spline: bool = True,
        base_activation: type[nn.Module] = nn.SiLU,
        grid_eps: float = 0.02,
        grid_range: tuple[float, float] = (-1.0, 1.0),
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size  # G：网格区间数
        self.spline_order = spline_order  # k：B-spline 阶数（3 = 三次）

        # 均匀节点向量，向两侧各外扩 k 个节点（B-spline 在边界处需要支撑）
        h = (grid_range[1] - grid_range[0]) / grid_size
        grid = (
            torch.arange(-spline_order, grid_size + spline_order + 1) * h + grid_range[0]
        ).expand(in_features, -1).contiguous()
        self.register_buffer("grid", grid)  # (in, G+2k+1)，不可训练

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        self.spline_weight = nn.Parameter(
            torch.empty(out_features, in_features, grid_size + spline_order)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(torch.empty(out_features, in_features))

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.reset_parameters()

    # --- 初始化 -------------------------------------------------------------

    def reset_parameters(self) -> None:
        """base 分支按 nn.Linear 的方式初始化。

        这里**必须**用 kaiming 而不是常数初始化。早期流传的一些 KAN 实现用
        常数初始化 base_weight，结果是模型基本学不动（在 MNIST 上只能到 20%），
        因为 base 分支是 φ 的主要成分，一开始就被压死了。
        """
        nn.init.kaiming_uniform_(self.base_weight, a=5**0.5 * self.scale_base)

        with torch.no_grad():
            # 在内部网格点上采小噪声，再用最小二乘反解出 spline 系数，
            # 使得初始时 spline 分支 ≈ 0（于是 φ ≈ w_b·silu(x)）。
            noise = (
                torch.rand(self.grid_size + 1, self.in_features, self.out_features) - 0.5
            ) * self.scale_noise / self.grid_size
            self.spline_weight.data.copy_(
                (1.0 if self.enable_standalone_scale_spline else self.scale_spline)
                * self.curve2coeff(
                    self.grid.T[self.spline_order : -self.spline_order],
                    noise,
                )
            )
            if self.enable_standalone_scale_spline:
                nn.init.kaiming_uniform_(self.spline_scaler, a=5**0.5 * self.scale_spline)

    # --- B-spline 基 --------------------------------------------------------

    def b_splines(self, x: Tensor) -> Tensor:
        """Cox–de Boor 递推，算出每个输入特征上的基函数取值。

        x: (batch, in_features) → (batch, in_features, G+k)
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        grid = self.grid  # (in, G+2k+1)
        x = x.unsqueeze(-1)  # (batch, in, 1)

        # 0 阶：区间指示函数
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)]) / (grid[:, k:-1] - grid[:, : -(k + 1)]) * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x) / (grid[:, k + 1 :] - grid[:, 1:(-k)]) * bases[:, :, 1:]
            )
        assert bases.size() == (x.size(0), self.in_features, self.grid_size + self.spline_order)
        return bases.contiguous()

    def curve2coeff(self, x: Tensor, y: Tensor) -> Tensor:
        """用最小二乘把"函数值 y 在 x 上的采样"反解成 spline 系数。

        y: (batch, in, out) → (out, in, G+k)
        初始化（把噪声变成系数）和 update_grid（在新网格上重拟合旧曲线）都用它。
        """
        A = self.b_splines(x).transpose(0, 1)  # (in, batch, G+k)
        B = y.transpose(0, 1)  # (in, batch, out)
        solution = torch.linalg.lstsq(A, B).solution  # (in, G+k, out)
        return solution.permute(2, 0, 1).contiguous()  # (out, in, G+k)

    @property
    def scaled_spline_weight(self) -> Tensor:
        return self.spline_weight * (
            self.spline_scaler.unsqueeze(-1) if self.enable_standalone_scale_spline else 1.0
        )

    # --- 前向 ---------------------------------------------------------------

    def forward(self, x: Tensor) -> Tensor:
        assert x.size(-1) == self.in_features
        original_shape = x.shape
        x = x.reshape(-1, self.in_features)

        # 全部重构成普通矩阵乘 —— 这是这个实现比原版 pykan 快 5-8 倍的原因。
        # 数学上和"对每条边逐条算 φ 再求和"完全等价。
        base_output = F.linear(self.base_activation(x), self.base_weight)
        spline_output = F.linear(
            self.b_splines(x).view(x.size(0), -1),
            self.scaled_spline_weight.view(self.out_features, -1),
        )
        return (base_output + spline_output).reshape(*original_shape[:-1], self.out_features)

    # --- 网格对齐 -----------------------------------------------------------

    @torch.no_grad()
    def update_grid(self, x: Tensor, margin: float = 0.01) -> None:
        """把 spline 网格重排到当前数据分布上（论文 Eq.2.16 的离散版）。

        做的事：先用**旧**网格算出旧曲线在这些样本点上的取值，再把网格换成
        「均匀网格和分位数网格的混合」，最后在新网格上最小二乘重拟合出一条
        和旧曲线尽可能接近的新曲线。

        于是**函数图像是连续的**，变的是"这条曲线现在用哪一段网格来表示"。
        不做这件事的话，激活值一旦漂移出网格，spline 分支就失效，
        φ 会静默退化成 silu —— 而且从图上完全看不出来。

        顺序不能反：`unreduced_spline_output` 必须用旧网格算，
        `curve2coeff` 必须用新网格拟合。写反了曲线会跳变。
        """
        assert x.dim() == 2 and x.size(1) == self.in_features
        batch = x.size(0)

        splines = self.b_splines(x).permute(1, 0, 2)  # (in, batch, G+k)
        # 用**未缩放**的系数算出**未缩放**的 spline 输出，两边都不带 scaler。
        #
        # 这里不能图省事用 scaled_spline_weight：那样等于"用乘过 s 的曲线去拟合
        # 没乘 s 的系数"，而前向又会再乘一次 s，于是**每次 update_grid 都把
        # spline 分支整体乘上一个 s**。|s| < 1 时它按几何级数衰减 ——
        # 实测调用 40 次后 spline_weight 从 1e-8 掉到 1e-24，spline 分支彻底死掉，
        # φ 静默退化成 silu。上游参考实现也有这个 bug，只是他们一辈子只调十几次
        # 网格、且那时 s 接近 1，所以没暴露；我们每 20 步调一次，是致命的。
        orig_coeff = self.spline_weight.permute(1, 2, 0)  # (in, G+k, out)
        unreduced_spline_output = torch.bmm(splines, orig_coeff).permute(1, 0, 2)  # (batch, in, out)

        x_sorted = torch.sort(x, dim=0)[0]

        # 某个输入特征在当前批次里完全没有变化时，均匀网格的步长会是 0，
        # 接着 B-spline 递推里会出现 0/0，一路 NaN 下去最后在 lstsq 里报一个
        # 完全看不懂的 MKL 错误。这时保留原网格更安全 —— 一个常数输入本来
        # 也没什么可对齐的。
        spread = x_sorted[-1] - x_sorted[0]
        if bool((spread < 1e-9).any()):
            return

        grid_adaptive = x_sorted[
            torch.linspace(0, batch - 1, self.grid_size + 1, dtype=torch.int64, device=x.device)
        ]
        uniform_step = (x_sorted[-1] - x_sorted[0] + 2 * margin) / self.grid_size
        grid_uniform = (
            torch.arange(self.grid_size + 1, dtype=torch.float32, device=x.device).unsqueeze(1)
            * uniform_step
            + x_sorted[0]
            - margin
        )
        # 分位数网格适合数据分布不均的情况，均匀网格在数据稀疏处更稳，混合折中
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive
        grid = torch.concatenate(
            [
                grid[:1]
                - uniform_step * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(1),
                grid,
                grid[-1:]
                + uniform_step * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(1),
            ],
            dim=0,
        )
        self.grid.copy_(grid.T)
        self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))

    # --- 可视化 -------------------------------------------------------------

    @torch.no_grad()
    def explain(self, samples: int = 48) -> KANExplanation:
        """导出每条边的函数曲线，供前端绘制。

        定义域取**该输入特征自己的网格有效范围**（去掉两侧各 k 个外扩节点），
        而不是固定的 grid_range —— 训练中网格会跟着数据移动，
        用固定定义域的话曲线会有一大段是平的，看不出信息。
        """
        k, g = self.spline_order, self.grid_size
        lo = self.grid[:, k]  # (in,)
        hi = self.grid[:, -(k + 1)]  # (in,)  = grid[:, G+k]

        t = torch.linspace(0, 1, samples, device=self.grid.device)
        xs = lo.unsqueeze(-1) + t.unsqueeze(0) * (hi - lo).unsqueeze(-1)  # (in, samples)

        bases = self._b_splines_per_feature(xs)  # (in, samples, G+k)
        spline = torch.einsum("inj,oij->oin", bases, self.scaled_spline_weight)
        base = torch.einsum("oi,in->oin", self.base_weight, self.base_activation(xs))
        phi = base + spline

        return KANExplanation(
            xs=xs,
            phi=phi,
            base=base,
            spline=spline,
            # 论文 Eq.2.17 的 |φ|₁。原定义是在数据集样本上取平均，
            # 这里在定义域上均匀采样取平均 —— 对可视化够用，而且不受批次影响。
            importance=phi.abs().mean(dim=-1),
        )

    def _b_splines_per_feature(self, x: Tensor) -> Tensor:
        """和 `b_splines` 同样的递推，但 x 的形状是 (in_features, n)：
        第 i 行用 grid[i] 上的节点。

        可视化时需要对每个输入特征各画一条曲线（定义域不同），
        没法拼成一个 batch 直接调用 b_splines。
        """
        grid = self.grid  # (in, W)
        x = x.unsqueeze(-1)  # (in, n, 1)
        bases = ((x >= grid[:, None, :-1]) & (x < grid[:, None, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, None, : -(k + 1)])
                / (grid[:, None, k:-1] - grid[:, None, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, None, k + 1 :] - x)
                / (grid[:, None, k + 1 :] - grid[:, None, 1:(-k)])
                * bases[:, :, 1:]
            )
        return bases.contiguous()
