"""★ 你自己的模型写在这里 ★

保存这个文件，浏览器不用重启、不用刷新 —— 下拉框里会自动出现新模型。

------------------------------------------------------------------------------
怎么写
------------------------------------------------------------------------------

    from .registry import register
    import torch.nn as nn

    @register("my-net", title="我的网络")
    class MyNet(nn.Module):
        def __init__(self, in_dim, out_dim, cfg):
            super().__init__()
            ...

        def forward(self, x):
            ...

构造函数固定收三个参数：

    in_dim   输入特征数（等于你当前在页面上勾选的特征个数）
    out_dim  输出维度。分类和回归都是 1，所以输出层直接写成 Linear(..., 1)
    cfg      超参对象，从它里面取你在页面上调的东西：
                cfg.activation          "tanh" / "relu" / "sigmoid" / ...
                cfg.learning_rate       学习率
                cfg.regularization      "none" / "l1" / "l2"
                cfg.regularization_rate
                cfg.is_regression       True 表示当前是回归任务
                cfg.problem             "classification" / "regression"

------------------------------------------------------------------------------
可视化是怎么来的
------------------------------------------------------------------------------

不用你做任何事 —— 引擎会在前向传播时挂 hook，自动把每个层画出来：

    nn.Linear            → 边画成粗细/颜色（权重矩阵）
    激活/归一化/Dropout   → 就地变换，不单独占一列
    认不出来的层          → 节点热力图照画，边退化成灰虚线

所以：**随便写，最差情况也只是画得不精细，不会白屏。**

如果你想让自定义层画出更丰富的东西（比如 KAN 那样，把每条边画成一条函数曲线），
给它加一个 `explain()` 方法，返回边的可视化信息。
约定见 playground/layers.py 里的实现示例。

------------------------------------------------------------------------------
装饰器参数
------------------------------------------------------------------------------

    @register(
        "my-net",                    # 唯一 key
        title="我的网络",             # 下拉框里显示的名字
        description="...",           # 悬停提示
        problem=None,                # None=通用；"classification" / "regression"
        configurable_shape=False,    # 是否响应前端的加层/加神经元按钮
        defaults={"learningRate": 0.001},   # 选中这个模型时建议的超参
    )

------------------------------------------------------------------------------
"""

from __future__ import annotations

# from .registry import register
# import torch.nn as nn
# from .config import make_activation


# 示例：带残差连接的 MLP。取消注释即可在页面上选到。
#
# @register("residual", title="Residual MLP", defaults={"learningRate": 0.003})
# class ResidualMLP(nn.Module):
#     """两个残差块。观察点：残差让深层网络也能训得动。"""
#
#     def __init__(self, in_dim, out_dim, cfg):
#         super().__init__()
#         width = 8
#         self.stem = nn.Linear(in_dim, width)
#         self.act = make_activation(cfg.activation)
#         self.block1 = nn.Linear(width, width)
#         self.block2 = nn.Linear(width, width)
#         self.head = nn.Linear(width, out_dim)
#
#     def forward(self, x):
#         h = self.act(self.stem(x))
#         h = h + self.act(self.block1(h))
#         h = h + self.act(self.block2(h))
#         return self.head(h)
