// 热力图渲染。
//
// 服务端发过来的是 base64 编码的 uint8，形状 (channels, height, width)：
// 值域 [-1,1] 线性映射到 [0,255]。这里反过来映射回 [-1,1] 再查色带。
//
// 一列神经元打包成一个 channels 维的数组发过来，而不是每个神经元单发一条消息 ——
// 一次 base64 比 N 次省掉大量 JSON 开销。切片在解码这层做，并且按通道缓存，
// 所以同一帧内既不会重复 atob，也不会重复逐像素换色。

import { HEATMAP_ALPHA, SHADES, shadeIndex } from "./palette.js";

function decodeBase64(data) {
  const binary = atob(data);
  const bytes = new Uint8Array(binary.length);
  for (let i = 0; i < binary.length; i++) bytes[i] = binary.charCodeAt(i);
  return bytes;
}

// encoded 对象 → Map<缓存键, ImageData>
const channelCache = new WeakMap();

/**
 * 取出某个通道并转成 ImageData（带缓存）。
 *
 * `discretize` 对应界面上的"离散化输出"：在查色带**之前**把值二值化成 ±1，
 * 于是边界变成纯橙/纯蓝两色，中间过渡消失 —— 更容易看清"分界线画在哪"。
 * 它纯粹是显示选项，不影响训练，也不影响 loss。
 */
export function channelImage(encoded, channelIndex, discretize = false) {
  let perChannel = channelCache.get(encoded);
  if (!perChannel) {
    perChannel = new Map();
    channelCache.set(encoded, perChannel);
  }

  const key = `${channelIndex}:${discretize ? 1 : 0}`;
  const cached = perChannel.get(key);
  if (cached) return cached;

  const { width, height, channels, data } = encoded;
  // 兼容单通道（输出层就是单通道）和打包的多通道
  const channel = channels === 1 ? 0 : Math.min(channelIndex, channels - 1);
  const bytes = decodeBase64(data);
  const planeSize = width * height;
  const offset = channel * planeSize;

  const image = new ImageData(width, height);
  for (let p = 0; p < planeSize; p++) {
    let value = (bytes[offset + p] / 255) * 2 - 1;
    if (discretize) value = value >= 0 ? 1 : -1;
    const [r, g, b] = SHADES[shadeIndex(value)];
    const o = p * 4;
    image.data[o] = r;
    image.data[o + 1] = g;
    image.data[o + 2] = b;
    image.data[o + 3] = HEATMAP_ALPHA;
  }

  perChannel.set(key, image);
  return image;
}

/** 把热力图整块铺满一个 canvas —— 主决策边界图用这个。 */
export function drawHeatmap(canvas, encoded, channelIndex = 0, discretize = false) {
  const ctx = canvas.getContext("2d");
  if (!encoded) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    return;
  }
  const image = channelImage(encoded, channelIndex, discretize);
  if (canvas.width !== image.width || canvas.height !== image.height) {
    canvas.width = image.width;
    canvas.height = image.height;
  }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.putImageData(image, 0, 0);
}

/**
 * 把某个通道缩放着画到共享 canvas 的指定区域 —— 神经元缩略图用这个。
 *
 * 网络图上最多 48 个神经元。给每个都建 canvas 元素的话，每秒 60 帧要重排
 * 48 个 DOM 节点。所以所有缩略图都画到**同一块盖在 SVG 上的 canvas** 里。
 */
export function blitHeatmap(ctx, encoded, channelIndex, x, y, size) {
  if (!encoded) return;
  const image = channelImage(encoded, channelIndex);
  const { width, height, data } = image;

  // putImageData 只能按原尺寸落笔，不会缩放；所以先在离屏 canvas 上放好，
  // 再用 drawImage 缩放过来。
  const scratch = getScratch(width, height);
  scratch.ctx.putImageData(new ImageData(data, width, height), 0, 0);
  ctx.imageSmoothingEnabled = true;
  ctx.drawImage(scratch.canvas, x, y, size, size);
}

let scratchCanvas = null;
let scratchCtx = null;

function getScratch(width, height) {
  if (!scratchCanvas) {
    scratchCanvas = document.createElement("canvas");
    scratchCtx = scratchCanvas.getContext("2d");
  }
  if (scratchCanvas.width !== width || scratchCanvas.height !== height) {
    scratchCanvas.width = width;
    scratchCanvas.height = height;
  }
  return { canvas: scratchCanvas, ctx: scratchCtx };
}
