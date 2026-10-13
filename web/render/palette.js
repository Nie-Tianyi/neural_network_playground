// 配色 —— 逐字照抄原版。
//
// 整个应用只有三个颜色，所有视觉都从这一条色带出来：
//
//     负 ──── #f59322（橙）   零 ──── #e8eaeb（灰）   正 ──── #0877bd（蓝）
//
// 决策边界、神经元热力图、权重连线、数据点、bias 小方块全都用它。
// 统一色带的好处是：同一块橙色在图上任何位置都是同一个含义。

export const NEG = "#f59322";
export const ZERO = "#e8eaeb";
export const POS = "#0877bd";

// 原版把色带量化成 30 档（d3.range(0, 1+1e-9, 1/30) 实际给出 31 个取值）。
// 量化而非连续取色，是为了让边界上的渐变看起来是"色块"而不是"雾"。
const NUM_SHADES = 31;

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// 原版用 d3.scale.linear().domain([0,.5,1]).range([NEG, ZERO, POS])，
// d3 对颜色字符串默认走 RGB 线性插值，所以这里也按 RGB 插。
function mix(a, b, t) {
  return [
    Math.round(a[0] + (b[0] - a[0]) * t),
    Math.round(a[1] + (b[1] - a[1]) * t),
    Math.round(a[2] + (b[2] - a[2]) * t),
  ];
}

const NEG_RGB = hexToRgb(NEG);
const ZERO_RGB = hexToRgb(ZERO);
const POS_RGB = hexToRgb(POS);

function rampAt(t) {
  return t < 0.5 ? mix(NEG_RGB, ZERO_RGB, t * 2) : mix(ZERO_RGB, POS_RGB, (t - 0.5) * 2);
}

/** 色带的 31 个离散档位，索引 0 = 最负、30 = 最正。 */
export const SHADES = Array.from({ length: NUM_SHADES }, (_, i) => rampAt(i / (NUM_SHADES - 1)));

/**
 * 数值 [-1, 1] → 颜色档位索引。
 * 超出范围截断 —— 回归任务的输出经常超出（比如 plane 能到 ±1.2），
 * 截断后仍然是最饱和的颜色。
 */
export function shadeIndex(value) {
  const t = Math.max(-1, Math.min(1, value));
  const idx = Math.floor(((t + 1) / 2) * NUM_SHADES);
  return Math.min(NUM_SHADES - 1, Math.max(0, idx));
}

/** 数值 [-1, 1] → CSS 颜色。 */
export function colorFor(value) {
  const [r, g, b] = SHADES[shadeIndex(value)];
  return `rgb(${r},${g},${b})`;
}

/**
 * 权重连线的线宽。原版是 clamp(|w|, 0, 5) → [1, 10]px 的**线性**映射
 * （不是 log —— log 虽然对动态范围更友好，但会让"权重翻倍"的视觉变化
 * 变得不明显，反而不利于观察训练过程）。
 */
export function linkWidth(weight) {
  const w = Math.abs(weight);
  return 1 + (Math.min(w, 5) / 5) * 9;
}

/** 热力图写入 canvas 时的透明度。原版硬编码 160/255。 */
export const HEATMAP_ALPHA = 160;
