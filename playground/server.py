"""FastAPI + WebSocket 服务。

一个进程同时托管静态页面和 WebSocket —— 没有构建步骤，没有跨域问题，
`uv run main.py` 之后浏览器直接打开就能用。

## 协议

客户端 → 服务端
    {"cmd": "init", "config": {...}}     首次连接，或整体重设
    {"cmd": "config", "config": {...}}   改配置（只传要改的字段）
    {"cmd": "play"} / {"cmd": "pause"}   连续训练 / 停下
    {"cmd": "step"}                      单步
    {"cmd": "reset"}                     重新初始化权重（不换数据）
    {"cmd": "regenerate"}                换一批数据
    {"cmd": "reload_models"}             重新扫描模型文件

服务端 → 客户端
    {"type": "static", ...}   选项、特征、数据集、模型列表（只在需要时发）
    {"type": "data", ...}     数据点（只在数据重新生成后发）
    {"type": "frame", ...}    每步的可视化状态
    {"type": "error", ...}
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import random
from pathlib import Path

import torch
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from . import serialize
from .config import DEFAULTS, OPTIONS, Config
from .datasets import DATASETS, DENSITY, FEATURES, grid_line
from .engine import Engine
from .registry import ModelRegistry, default_registry

log = logging.getLogger("playground")

WEB_DIR = Path(__file__).resolve().parent.parent / "web"

# 目标帧率。原版是每个动画帧跑一个 epoch（约 60/s），这里不去刻意追，
# 跑得动就多跑，跑不动就让帧率掉下来 —— 训练步数不受影响（每帧固定一个 epoch）。
TARGET_FPS = 60
FRAME_INTERVAL = 1.0 / TARGET_FPS


class Session:
    """一个浏览器标签页对应一个 Session。

    单个 Engine 直接放在这里；P4 的「多实验对比」就是把它换成一个 Engine 列表，
    每个 Engine 同步推进一步 —— 引擎本身不依赖 Web，所以不需要额外机制。
    """

    def __init__(self, registry: ModelRegistry) -> None:
        self.registry = registry
        self.engine = Engine(registry)
        self.playing = False
        self._task: asyncio.Task | None = None

    # --- 消息构造 -----------------------------------------------------------

    def static_message(self) -> dict:
        assert self.engine.cfg is not None
        return _static_message(self.registry, self.engine.cfg)

    def data_message(self) -> dict:
        assert self.engine.train is not None and self.engine.test is not None
        assert self.engine.cfg is not None
        spec = DATASETS[self.engine.cfg.dataset]

        message = {
            "type": "data",
            "mode": spec.mode,
            "domain": list(spec.domain),
            "train": _encode_points(self.engine.train.points),
            "test": _encode_points(self.engine.test.points),
        }
        if spec.is_1d:
            # 拟合图要画"真实函数"作为参考线，前端算不了，这里一并送过去。
            x_axis = grid_line(spec.domain)
            message["gridX"] = serialize.encode_line(x_axis)
            message["truth"] = serialize.encode_line(
                torch.tensor([spec.truth(float(v)) for v in x_axis])
            )
        return message

    def frame(self) -> dict:
        graph, payload = self.engine.snapshot()
        message = {
            "type": "frame",
            "mode": "1d" if self.engine.is_1d else "2d",
            "iter": payload["iter"],
            "lossTrain": payload["lossTrain"],
            "lossTest": payload["lossTest"],
            "graph": serialize.encode_graph(graph),
        }
        if self.engine.is_1d:
            message["curves"] = [
                serialize.encode_curve_columns(a) for a in payload["curves"]
            ]
        else:
            message["heatmaps"] = [serialize.encode_heatmap(h) for h in payload["heatmaps"]]
        return message

    def advance(self) -> dict:
        """跑一个 epoch 并编码成一条帧消息。"""
        self.engine.step()
        return self.frame()

    # --- 生命周期 -----------------------------------------------------------

    async def configure(self, payload: dict) -> None:
        cfg = _config_from_payload(payload, self.engine.cfg, self.registry)
        self.engine.configure(cfg)

    def reload_models(self) -> None:
        """重新扫描模型文件 —— 热重载的入口。

        模型类可能整个换了（旧类还引用着旧的代码对象），所以已建好的网络对象
        必须丢掉重建。数据保留 —— 你改的是结构，不是数据集。

        如果当前选的模型在新版本里没了（改名/删掉了），退回到第一个可用模型，
        而不是抛异常把页面搞崩。
        """
        self.registry.reload()
        assert self.engine.cfg is not None
        if self.registry.get(self.engine.cfg.model) is None:
            available = next(iter(self.registry.specs), None)
            if available is None:
                return
            self.engine.configure(self.engine.cfg.replace(model=available))
            return
        self.engine.reset()

    # --- 播放循环 -----------------------------------------------------------

    def start(self, ws: WebSocket) -> None:
        if self._task is None or self._task.done():
            self.playing = True
            self._task = asyncio.create_task(self._run(ws))

    def stop(self) -> None:
        self.playing = False
        if self._task is not None:
            self._task.cancel()
            self._task = None

    async def _run(self, ws: WebSocket) -> None:
        """连续训练：每帧一个 epoch，然后把状态发出去。

        训练放在线程里跑：torch 的算子会释放 GIL，所以这样不会把事件循环
        卡住 —— 否则你点「暂停」得等当前帧算完才响应。
        """
        try:
            while self.playing:
                started = asyncio.get_running_loop().time()

                # 训练过程中盯着模型文件 —— 你在 PyCharm 里 Ctrl+S，这边
                # 下一个动画帧就生效，不用重启也不用刷新页面。
                if self.registry.reload_if_stale():
                    self.reload_models()
                    await ws.send_json(self.static_message())

                frame = await asyncio.to_thread(self.advance)
                await ws.send_json(frame)

                elapsed = asyncio.get_running_loop().time() - started
                if elapsed < FRAME_INTERVAL:
                    await asyncio.sleep(FRAME_INTERVAL - elapsed)
                else:
                    await asyncio.sleep(0)  # 跑得比目标帧率慢，让出控制权
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # 训练中出错不该静默死掉
            # 这里捕获得宽是对的：训练循环里可能抛任何东西（用户自定义层、
            # torch 内部错误、序列化失败……），而**唯一不能接受的**是让它
            # 静默退出、页面停在一个不动的画面上。停下来 + 把错误发出去。
            self.playing = False
            log.exception("训练循环出错")
            await _safe_send(ws, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})


# --- 编码辅助 ----------------------------------------------------------------


def _encode_points(points) -> list[list[float]]:
    return [[round(p.x, 4), round(p.y, 4), round(p.label, 4)] for p in points]


# 前端用 camelCase（跟着原版的字段名走，读起来也更像 JS），
# 而 Config 是 Python 的 snake_case。这张表是两者之间的**唯一**桥梁。
#
# 曾经这里的 bug 是：字段名直接透传给 Config.replace()，于是前端只要
# 动一下"学习率"下拉框就是 TypeError、连接直接断掉。所以它的值必须显式写出来，
# 不能靠 key 自动转换 —— 表驱动至少能一眼看全。
_FIELDS: dict[str, tuple[str, type | tuple[type, ...]]] = {
    "dataset": ("dataset", str),
    "model": ("model", str),
    "activation": ("activation", str),
    "regularization": ("regularization", str),
    "optimizer": ("optimizer", str),
    "device": ("device", str),
    # 协议里的 noise 一律是归一化的 0–0.5（和 datasets.generate 的入参一致）。
    # 界面上那个 0–50 的滑条是**前端的显示单位**，由前端自己换算 ——
    # 之前这里是"clamp(0, 0.5)"，而前端发的是 0–50，于是滑条实际上只有
    # "关"和"最大"两档。
    "noise": ("noise", (int, float)),
    "percTrainData": ("perc_train", (int, float)),
    "seed": ("seed", (int, float)),
    "learningRate": ("learning_rate", (int, float)),
    "regularizationRate": ("regularization_rate", (int, float)),
    "batchSize": ("batch_size", int),
}


def _model_defaults(spec) -> dict:
    """把模型自带的建议超参翻译成 Config 的字段名。

    `spec.defaults` 用的是前端那套 camelCase 名字（和客户端发过来的一致），
    所以复用同一张 `_FIELDS` 映射表 —— 两处名字必须对齐，共用一张表就不会走岔。
    """
    defaults: dict = {}
    for wire_name, value in spec.defaults.items():
        entry = _FIELDS.get(wire_name)
        if entry is None or isinstance(value, bool):
            continue  # 含内部字段（如 _file）和未知键
        field_name, expected = entry
        if isinstance(value, expected):
            defaults[field_name] = value
    return defaults


def _config_from_payload(
    payload: dict,
    current: Config | None,
    registry: ModelRegistry | None = None,
) -> Config:
    """把前端发来的字段合并进当前配置。

    只认识的字段才合并，而且类型对不上就忽略 —— 前端多发字段、发错类型
    都不该让整条连接断掉。
    """
    base = current or Config()

    # 换模型时把它建议的超参一起带上（KAN 需要 adam + 更小的学习率，
    # 用 MLP 的默认 SGD/0.03 基本学不动）。放在服务端做而不是只靠前端，
    # 是因为 `?model=kan` 这种 URL 直接进来的路径也要生效 ——
    # 否则分享出去的链接会给别人一套错误的超参。
    switching_model = (
        "model" in payload
        and isinstance(payload["model"], str)
        and payload["model"] != base.model
    )
    if switching_model and registry is not None:
        spec = registry.get(payload["model"])
        if spec is not None:
            base = base.replace(**_model_defaults(spec))

    kwargs: dict = {}

    for wire_name, (field_name, expected) in _FIELDS.items():
        if wire_name not in payload:
            continue
        value = payload[wire_name]
        # bool 是 int 的子类，得先挡掉，否则 True 会被当成合法的 batchSize
        if isinstance(value, bool):
            continue
        if isinstance(value, expected):
            kwargs[field_name] = value

    if "features" in payload and isinstance(payload["features"], (list, tuple)):
        selected = tuple(f for f in payload["features"] if f in FEATURES)
        # 一个特征都不剩的话模型就没有输入了，保持原样
        if selected:
            kwargs["features"] = selected

    if "networkShape" in payload and isinstance(payload["networkShape"], (list, tuple)):
        shape = [
            max(1, min(8, int(n)))  # 原版：每层 1–8 个神经元
            for n in payload["networkShape"]
            if isinstance(n, (int, float)) and not isinstance(n, bool)
        ][:6]  # 原版：最多 6 个隐藏层
        kwargs["network_shape"] = tuple(shape)

    if "noise" in kwargs:
        kwargs["noise"] = max(0.0, min(0.5, float(kwargs["noise"])))

    return base.replace(**kwargs)


def _dataset_thumbnail(dataset_id: str) -> list[list[float]]:
    """数据集缩略图的点。

    原版是在浏览器里用 dataGenerator(200, 0) 现算的（200 个点、无噪声）。
    这里由服务端算好一次性发过去 —— 反正生成逻辑本来就在服务端，
    没必要在前端再实现一遍（两份实现迟早会对不上）。
    """
    spec = DATASETS[dataset_id]
    rng = random.Random(0)  # 固定种子，缩略图不该每次刷新都变
    points = spec.generator(rng, 200, 0.0)
    return [[round(p.x, 3), round(p.y, 3), round(p.label, 3)] for p in points]


def _static_message(registry: ModelRegistry, cfg: Config) -> dict:
    return {
        "type": "static",
        # 把**生效的**配置发回去。前端启动时发的是空配置，一切以服务端为准 ——
        # 否则前端会拿着 undefined 去和下拉框的默认值较劲。
        "config": cfg.to_dict(),
        "features": [
            {"id": f.id, "label": f.label, "isInput": f.is_input} for f in FEATURES.values()
        ],
        "datasets": [
            {
                "id": d.id,
                "title": d.title,
                "problem": d.problem,
                "mode": d.mode,
                "domain": list(d.domain),
                "numSamples": d.num_samples,
                # 这个数据集允许用哪些输入特征 —— 前端靠它决定输入列画几个方块
                "features": list(d.features),
                "thumbnail": _dataset_thumbnail(d.id),
            }
            for d in DATASETS.values()
        ],
        "options": OPTIONS,
        "defaults": DEFAULTS,
        "density": DENSITY,
        "models": [
            {
                "key": spec.key,
                "title": spec.title,
                "description": spec.description,
                "problem": spec.problem,
                "configurableShape": spec.configurable_shape,
                "defaults": {k: v for k, v in spec.defaults.items() if not k.startswith("_")},
            }
            for spec in registry.specs.values()
        ],
        "modelErrors": registry.errors,
    }


async def _safe_send(ws: WebSocket, message: dict) -> None:
    """往一个可能已经断开的连接发消息 —— 发不出去就算了。

    这里失败是常态而不是异常：浏览器关掉标签页、暂停时任务被取消，
    都会让 send 抛。既不值得让调用方去处理，也不值得往日志里刷屏。
    """
    with contextlib.suppress(Exception):
        await ws.send_json(message)


# --- 应用 --------------------------------------------------------------------


def create_app() -> FastAPI:
    app = FastAPI(title="PyTorch Playground")
    registry = default_registry()
    session = Session(registry)

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        await ws.accept()
        try:
            while True:
                message = await ws.receive_json()
                await _handle(ws, session, message)
        except WebSocketDisconnect:
            session.stop()
        except Exception as exc:
            session.stop()
            # 一定要往控制台打一份。之前这里只把错误发给浏览器，结果
            # "改一下学习率就断连"这种 bug 在终端里完全看不见。
            log.exception("WebSocket 处理出错")
            await _safe_send(ws, {"type": "error", "message": f"{type(exc).__name__}: {exc}"})

    @app.get("/")
    async def index() -> FileResponse:
        return FileResponse(WEB_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")
    return app


async def _handle(ws: WebSocket, session: Session, message: dict) -> None:
    cmd = message.get("cmd")

    if cmd == "init":
        session.stop()
        await session.configure(message.get("config", {}))
        await ws.send_json(session.static_message())
        await ws.send_json(session.data_message())
        await ws.send_json(session.frame())

    elif cmd == "config":
        was_playing = session.playing
        session.stop()
        await session.configure(message.get("config", {}))
        await ws.send_json(session.static_message())
        await ws.send_json(session.data_message())
        await ws.send_json(session.frame())
        if was_playing:
            session.start(ws)

    elif cmd == "reset":
        session.stop()
        session.engine.reset()
        await ws.send_json(session.frame())

    elif cmd == "regenerate":
        # 换一个种子重新生成数据 —— 对应原版的 Regenerate 按钮
        session.stop()
        await session.configure({"seed": random.random()})
        await ws.send_json(session.data_message())
        await ws.send_json(session.frame())

    elif cmd == "step":
        session.stop()
        await ws.send_json(await asyncio.to_thread(session.advance))

    elif cmd == "play":
        session.start(ws)

    elif cmd == "pause":
        session.stop()

    elif cmd == "reload_models":
        session.stop()
        session.reload_models()
        await ws.send_json(session.static_message())

    else:
        await _safe_send(ws, {"type": "error", "message": f"未知指令: {cmd!r}"})
