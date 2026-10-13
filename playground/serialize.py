"""把引擎的状态编码成能在 WebSocket 上传的东西。

## 为什么热力图要量化 + base64

主决策边界是 100×100 = 10000 个数。如果按 JSON 数字发（`[-0.42, 0.11, ...]`），
一帧就是 200KB 左右，60fps 下 12MB/s —— 本地也会卡。

热力图的值域本来就是 [-1, 1]（分类的标签是 ±1，tanh 输出也在 (-1,1)），
量化成 uint8 刚好够用（256 级）—— 而且原版本来就把颜色量化成 30 档，
精度远没有损失。base64 之后一帧降到约 13KB。

神经元小热力图是 10×10，本来就小，但走同一套编码，前端只写一份解码逻辑。
"""

from __future__ import annotations

import base64
from typing import Any

import torch
from torch import Tensor

from .adapters.base import (
    EdgePayload,
    FunctionCurve,
    Opaque,
    ProductInteraction,
    ScalarWeight,
)
from .introspection import Column, DisplayGraph, Edge

# 曲线采样点数。KAN 的一条边就是一条曲线，50 个点足够画得平滑，
# 而且量化到小数点后 4 位后一条边只占几百字节。
CURVE_SAMPLES = 48


# --- 热力图 ------------------------------------------------------------------


def encode_heatmap(grid: Tensor) -> dict[str, Any]:
    """(channels, h, w) → base64 的 uint8 数据。

    值域 [-1, 1] 线性映射到 [0, 255]，超出范围的截断（回归任务的输出会超出）。
    前端拿到后反过来映射回 [-1, 1] 再查颜色表。
    """
    grid = grid.detach().to("cpu")
    channels, height, width = grid.shape
    quantized = (grid.clamp(-1.0, 1.0) + 1.0).mul(127.5).round().to(torch.uint8)
    return {
        "channels": channels,
        "height": height,
        "width": width,
        "data": base64.b64encode(quantized.numpy().tobytes()).decode("ascii"),
    }


# --- 边 ----------------------------------------------------------------------


def encode_payload(payload: EdgePayload) -> dict[str, Any]:
    """编码一条边的负载。前端的渲染分支由 `kind` 决定。"""
    if isinstance(payload, ScalarWeight):
        return {
            "kind": "scalar",
            "value": round(payload.value, 5),
            "dead": payload.dead,
        }
    if isinstance(payload, FunctionCurve):
        return {
            "kind": "curve",
            "xs": _round_list(payload.xs),
            "ys": _round_list(payload.ys),
            "base": _round_list(payload.base) if payload.base else None,
            "spline": _round_list(payload.spline) if payload.spline else None,
            "importance": round(payload.importance, 5),
        }
    if isinstance(payload, ProductInteraction):
        return {
            "kind": "product",
            "terms": list(payload.terms),
            "weight": round(payload.weight, 5),
        }
    if isinstance(payload, Opaque):
        return {"kind": "opaque", "label": payload.label}
    # 走到这里说明新加了 payload 类型但忘了在这里处理。
    # 降级成 opaque 而不是抛异常 —— 一条边画不出来不该让整帧失败。
    return {"kind": "opaque", "label": type(payload).__name__}


def _round_list(values: list[float], digits: int = 4) -> list[float]:
    return [round(float(v), digits) for v in values]


def encode_edge(edge: Edge) -> dict[str, Any]:
    return {"src": edge.src, "dst": edge.dst, "payload": encode_payload(edge.payload)}


def encode_column(column: Column) -> dict[str, Any]:
    return {
        "title": column.title,
        "kind": column.kind,
        # active 只对输入列有意义（这个特征有没有被喂给模型）
        "nodes": [{"label": node.label, "active": node.active} for node in column.nodes],
    }


def encode_graph(graph: DisplayGraph) -> dict[str, Any]:
    return {
        "columns": [encode_column(c) for c in graph.columns],
        "edges": [[encode_edge(e) for e in layer] for layer in graph.edges],
    }
