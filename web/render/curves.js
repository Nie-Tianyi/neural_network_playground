// 一维模式（函数拟合）的渲染。
//
// 二维模式的神经元画的是 100×100 网格上的热力图；一维模式下每个神经元
// 只有一条 `激活值 vs x` 的曲线，所以在方块里直接画那条曲线。
//
// **纵向标度固定为 ±1**，和二维热力图同一个约定：贴着中线的平线就真的表示
// "这个神经元没在输出什么"，不会因为自动缩放而把噪声放大成结构。

import { decodeBase64 } from "./heatmap.js";

// encoded → Map<channelIndex, Float32Array>（值域 [-1,1]）
const cache = new WeakMap();

/** 取出某通道的曲线数值。 */
export function curveValues(encoded, channelIndex) {
  let perChannel = cache.get(encoded);
  if (!perChannel) {
    perChannel = new Map();
    cache.set(encoded, perChannel);
  }
  const cached = perChannel.get(channelIndex);
  if (cached) return cached;

  const { channels, length, data } = encoded;
  const channel = channels === 1 ? 0 : Math.min(channelIndex, channels - 1);
  const bytes = decodeBase64(data);
  const offset = channel * length;

  const values = new Float32Array(length);
  for (let i = 0; i < length; i++) {
    values[i] = (bytes[offset + i] / 255) * 2 - 1;
  }
  perChannel.set(channelIndex, values);
  return values;
}

/**
 * 把一个神经元的激活曲线画进它的方块里。
 *
 * 纵轴固定 ±1（超出截断），所以"这条线有多高"是可以跨神经元比较的。
 */
export function drawNodeCurve(ctx, values, cx, cy, size) {
  const half = size / 2;
  const left = cx - half;
  const top = cy - half;
  const toY = (v) => cy - Math.max(-1, Math.min(1, v)) * half * 0.86;

  ctx.save();
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  // 零线：判断这个神经元的输出整体偏正还是偏负
  ctx.globalAlpha = 0.3;
  ctx.strokeStyle = "#183D4E";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(left, cy);
  ctx.lineTo(left + size, cy);
  ctx.stroke();

  // 这里**不要**照搬 KAN 边曲线那套"描白边"的画法：那套是为了让曲线压在
  // 别的线上也看得清，而节点方块里背后什么都没有 —— 3px 的白边只会把
  // 1.5px 的深色线自己吃掉，结果方块看起来是空的。（这个坑踩过：
  // 一度以为 drawNodeCurves 根本没被调用，换成红线才看清是线太淡。）
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "#183D4E";
  ctx.lineWidth = 1.8;
  ctx.beginPath();
  for (let i = 0; i < values.length; i++) {
    const px = left + (i / (values.length - 1)) * size;
    const py = toY(values[i]);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
  ctx.restore();
}

// --- 拟合图（输出面板）------------------------------------------------------

const TRUTH_COLOR = "#9aa8b0";
const PREDICTION_COLOR = "#183d4e";
const AXIS_COLOR = "#d8dfe3";

export class FitPlot {
  /** @param canvas 300×300 的 canvas；坐标系和二维模式的决策边界图一致。 */
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.gridX = null;
    this.truth = null;
  }

  /** 换数据集时调用：更新横轴和真实函数。 */
  setData({ gridX, truth, domain }) {
    this.gridX = gridX;
    this.truth = truth;
    this.domain = domain;
  }

  /**
   * @param prediction 模型在网格上的输出（原始浮点，可能超出 ±1）
   * @param points     数据点 [[x, y, label], ...]
   */
  draw(prediction, points) {
    const { ctx, canvas } = this;
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);
    if (!this.gridX || !prediction) return;

    const [xLo, xHi] = this.domain;

    // 纵轴按"真实函数 ∪ 预测 ∪ 数据点"的实际范围自动定 ——
    // 拟合图和决策边界图不一样，它要如实显示 x³/9 这种能到 ±3 的输出，
    // 固定 ±1 会把两端削平。留 8% 的边距避免曲线贴边。
    let lo = Infinity;
    let hi = -Infinity;
    const consider = (v) => {
      if (!Number.isFinite(v)) return;
      if (v < lo) lo = v;
      if (v > hi) hi = v;
    };
    for (const v of this.truth) consider(v);
    for (const v of prediction) consider(v);
    for (const p of points) consider(p[2]);
    if (lo === Infinity) return;
    if (hi - lo < 1e-6) {
      hi += 1;
      lo -= 1;
    }
    const pad = (hi - lo) * 0.08;
    lo -= pad;
    hi += pad;

    const toX = (x) => ((x - xLo) / (xHi - xLo)) * width;
    const toY = (v) => height - ((v - lo) / (hi - lo)) * height;

    // 零线（只在纵轴跨过 0 时才画，否则它没有意义）
    if (lo < 0 && hi > 0) {
      ctx.strokeStyle = AXIS_COLOR;
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, toY(0));
      ctx.lineTo(width, toY(0));
      ctx.stroke();
    }

    // 数据点先画，曲线压在上面
    for (const [x, , label] of points) {
      ctx.beginPath();
      ctx.arc(toX(x), toY(label), 2.5, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
      ctx.strokeStyle = "#183D4E";
      ctx.lineWidth = 1.2;
      ctx.stroke();
    }

    // 真实函数：虚线灰，作为参考
    this.strokeSeries(
      this.gridX,
      this.truth,
      toX,
      toY,
      TRUTH_COLOR,
      1.5,
      [6, 4],
    );

    // 模型预测：实线深色
    this.strokeSeries(this.gridX, prediction, toX, toY, PREDICTION_COLOR, 2, null);
  }

  strokeSeries(xs, ys, toX, toY, color, lineWidth, dash) {
    const { ctx } = this;
    ctx.save();
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.lineWidth = lineWidth;
    ctx.lineJoin = "round";
    if (dash) ctx.setLineDash(dash);
    for (let i = 0; i < xs.length; i++) {
      const px = toX(xs[i]);
      const py = toY(ys[i]);
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
    ctx.stroke();
    ctx.restore();
  }
}
