// 网络图渲染。
//
// 分成两层：
//   <svg>     连线 + 节点方块 + 文字标签   —— 需要命中检测（悬停看权重、点输入层切特征）
//   <canvas>  所有神经元的小热力图        —— 每帧都变，走 canvas 才不会掉帧
//
// 几何参数照抄原版：节点 30×30 的方块（不是圆），纵向步距 55px，
// 输入列中心距左边 65px，隐藏层从 118px 起均分。

import { colorFor, linkWidth } from "./palette.js";
import { blitHeatmap } from "./heatmap.js";
import { curveValues, drawNodeCurve } from "./curves.js";

const SVG_NS = "http://www.w3.org/2000/svg";
const RECT_SIZE = 30;
const VERTICAL_STEP = RECT_SIZE + 25; // 55
// 输入列的中心。左边要留出空间放 "sin(X₂)" 这类右对齐、向左溢出的标签。
const INPUT_X = 65;
const LABEL_FONT = 13;

// 边函数曲线（KAN）的迷你图尺寸区间。实际尺寸按列间距和这一层的边数自适应 ——
// 定死尺寸试过，34×22 在宽屏上小得根本看不出是条函数曲线，
// 而那个尺寸本来就是整个 KAN 可视化的意义所在。
const SPARKLINE_MIN_WIDTH = 34;
const SPARKLINE_MAX_WIDTH = 68;

/** 把 "X_1^2" 这类标签拆成 普通/下标/上标 三段，用 tspan 渲染。 */
function labelParts(label) {
  const parts = [];
  let buffer = "";
  for (let i = 0; i < label.length; i++) {
    const ch = label[i];
    if ((ch === "_" || ch === "^") && i + 1 < label.length) {
      if (buffer) parts.push({ text: buffer, shift: 0 });
      buffer = "";
      parts.push({ text: label[i + 1], shift: ch === "_" ? "sub" : "super" });
      i++;
    } else {
      buffer += ch;
    }
  }
  if (buffer) parts.push({ text: buffer, shift: 0 });
  return parts;
}

/** 贝塞尔 S 曲线 —— 从源节点右缘到目标节点左缘。 */
function linkPath(x1, y1, x2, y2) {
  const mid = (x1 + x2) / 2;
  return `M${x1},${y1} C${mid},${y1} ${mid},${y2} ${x2},${y2}`;
}

export class NetworkRenderer {
  constructor(svg, canvas, options = {}) {
    this.svg = svg;
    this.container = svg.parentElement; // #network，用它量宽度
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.onToggleFeature = options.onToggleFeature || (() => {});
    this.onHoverLink = options.onHoverLink || (() => {});

    this.graph = null;
    // 最近一帧的可视化数据：二维是热力图数组，一维是曲线数组，靠 this.mode 区分
    this.visual = null;
    this.mode = "2d";
    this.layout = null;
    this.signature = "";
    this.lastWidth = 0;

    // 容器尺寸变化时重新布局。用 ResizeObserver 而不是 window.resize ——
    // 侧边栏折叠、字体加载都可能改变容器宽度而窗口尺寸不变。
    if (typeof ResizeObserver !== "undefined") {
      this._observer = new ResizeObserver(() => this._invalidate());
      this._observer.observe(this.container);
    }
  }

  _invalidate() {
    // 只在**宽度**变化时重排：容器高度会随 SVG 高度一起变（它包着 SVG），
    // 不加这个判断就会"改高度 → 触发观察 → 再改高度"地转圈。
    if (this.measureWidth() === this.lastWidth) return;
    this.signature = "";
    // 暂停状态下不会有新帧送进来，所以这里主动重画一次。
    // 必须带上 mode —— 一维模式的视觉数据和二维模式是不同的东西，
    // 漏掉的话窗口一缩放就会用热力图的解读方式去画曲线数组。
    if (this.graph) this.render(this.graph, this.visual, this.mode);
  }

  /**
   * 每次拿到新一帧就调用。结构没变时只更新颜色/线宽，不重建 DOM。
   *
   * `visual` 是二维模式的热力图数组或一维模式的曲线数组，由 `mode` 决定怎么解读。
   */
  render(graph, visual, mode = "2d") {
    this.graph = graph;
    this.visual = visual;
    this.mode = mode;

    // 容器还没布局好（宽度为 0）就攒着，等 ResizeObserver 报出真实尺寸再画。
    if (this.measureWidth() === 0) return;

    const signature = structureSignature(graph);
    if (signature !== this.signature || this.measureWidth() !== this.lastWidth) {
      this.signature = signature;
      this.layout = this.computeLayout(graph);
      this.buildDom(graph);
    } else {
      this.layout = this.computeLayout(graph);
    }

    this.updateLinks(graph);
    if (mode === "1d") this.drawNodeCurves(visual);
    else this.drawHeatmaps(visual);
    this.drawEdgeCurves(graph);
  }

  /**
   * 一维模式下的节点：把该神经元的激活曲线画进方块。
   *
   * 二维模式一个神经元是平面上的一整张图，一维模式下它只是 `激活值 vs x`
   * 一条线 —— 信息量少得多，但也因此可以直接把线画出来，比热力图更直观。
   */
  drawNodeCurves(curves) {
    const { width, height } = this.layout;
    this.ctx.clearRect(0, 0, width, height);
    curves.forEach((encoded, columnIndex) => {
      if (!encoded) return;
      this.layout.columns[columnIndex].nodes.forEach((node, nodeIndex) => {
        drawNodeCurve(
          this.ctx,
          curveValues(encoded, nodeIndex),
          node.cx,
          node.cy,
          RECT_SIZE - 2,
        );
      });
    });
  }

  /**
   * 量容器宽度。0 表示还没布局好（此时调用方应当跳过这一帧）。
   *
   * 两点必须注意：
   *   1. 不能用 `svg.clientWidth` —— SVG 元素上这个属性在 Chrome 里恒为 0，
   *      于是会退回到兜底尺寸，而实际元素宽度由 CSS 决定，两者对不上就画歪了。
   *      量父 div 的 getBoundingClientRect 才可靠。
   *   2. **不要加下限**。给一个"最小宽度"看起来更安全，但只要它大于容器实际宽度，
   *      设成 px 的 SVG 就会撑破容器，整个页面出现横向滚动条。
   *      宁可画得窄，也不能画得溢出。
   */
  measureWidth() {
    const width = this.container.getBoundingClientRect().width;
    return width > 0 ? Math.round(width) : 0;
  }

  // --- 布局 ---------------------------------------------------------------

  computeLayout(graph) {
    const maxNodes = Math.max(...graph.columns.map((c) => c.nodes.length), 1);
    const width = this.measureWidth();
    const height = maxNodes * VERTICAL_STEP + 24;
    const count = graph.columns.length;

    // 所有列在 [INPUT_X, 右缘] 上**等距**分布，一套公式管到底。
    //
    // 之前这里是分开算的：输入列钉在常量 65，隐藏层铺在另一个区间
    // [FEATURE_WIDTH, 输出列-60] 上。结果是相邻列的间距完全不均匀 ——
    // 两层隐藏层时输入列和第一层只隔 53px，而两层隐藏层之间隔了 500px，
    // 首尾挤成一团、中间空一大片。视觉上非常别扭。
    const inputCx = INPUT_X;
    const outputCx = width - RECT_SIZE / 2;
    const step = count > 1 ? (outputCx - inputCx) / (count - 1) : 0;

    const columns = graph.columns.map((column, index) => {
      const cx = inputCx + index * step;

      // 垂直居中：节点少的那一列也要对齐中线，否则图看起来是歪的
      const total = column.nodes.length;
      const offset = (maxNodes - total) / 2;
      const nodes = column.nodes.map((_, i) => ({
        cx,
        cy: (i + offset) * VERTICAL_STEP + VERTICAL_STEP / 2 + 12,
      }));
      return { cx, nodes, column };
    });

    return { width, height, columns };
  }

  // --- DOM 构建（只在结构或尺寸变化时）------------------------------------

  buildDom(graph) {
    const { width, height, columns } = this.layout;
    this.lastWidth = width;

    // 关键：viewBox 和元素的 CSS 尺寸**严格相等**，1:1 映射。
    // 只要两者不一致（哪怕只是比例不同），浏览器就要做一次缩放，
    // 而 SVG 的 preserveAspectRatio 又会引入留白或错位 —— 所以干脆让它们相等，
    // 从根上消除这个问题。
    this.svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
    this.svg.setAttribute("width", width);
    this.svg.setAttribute("height", height);
    this.svg.style.width = `${width}px`;
    this.svg.style.height = `${height}px`;

    this.canvas.width = width;
    this.canvas.height = height;
    this.canvas.style.width = `${width}px`;
    this.canvas.style.height = `${height}px`;

    this.svg.innerHTML = "";

    const linkGroup = document.createElementNS(SVG_NS, "g");
    const nodeGroup = document.createElementNS(SVG_NS, "g");
    this.svg.append(linkGroup, nodeGroup);

    // --- 连线 ---
    this.linkEls = graph.edges.map((edges, layerIndex) => {
      const from = columns[layerIndex];
      const to = columns[layerIndex + 1];
      return edges.map((edge, edgeIndex) => {
        const src = from.nodes[edge.src];
        const dst = to.nodes[edge.dst];

        // 终点纵向按扇形散开，避免所有线收在同一个点。
        // 原版：((index - (len-1)/2) / len) * 12
        const fan = ((edgeIndex - (edges.length - 1) / 2) / Math.max(edges.length, 1)) * 12;

        const path = document.createElementNS(SVG_NS, "path");
        path.setAttribute(
          "d",
          linkPath(src.cx + RECT_SIZE / 2 + 2, src.cy, dst.cx - RECT_SIZE / 2, dst.cy + fan),
        );
        path.setAttribute("fill", "none");
        path.setAttribute("stroke-linecap", "round");

        // 一条透明的粗线专门用来接鼠标 —— 细线根本点不中
        const hit = document.createElementNS(SVG_NS, "path");
        hit.setAttribute("d", path.getAttribute("d"));
        hit.setAttribute("fill", "none");
        hit.setAttribute("stroke", "transparent");
        hit.setAttribute("stroke-width", "8");
        hit.style.cursor = "pointer";
        // 悬停时现查 payload，而不是捕获闭包里的那份 ——
        // DOM 只在结构变化时重建，而权重每一帧都在变，
        // 闭包里那份在你松手看的时候就已经过期了。
        hit.addEventListener("mouseenter", () => {
          const current = this.graph?.edges[layerIndex]?.[edgeIndex]?.payload;
          this.onHoverLink(current, src.cx, src.cy);
        });
        hit.addEventListener("mouseleave", () => this.onHoverLink(null));

        linkGroup.append(path, hit);
        return path;
      });
    });

    // --- 节点 ---
    this.nodeEls = graph.columns.map((column, columnIndex) => {
      return column.nodes.map((node, nodeIndex) => {
        const { cx, cy } = columns[columnIndex].nodes[nodeIndex];
        const rect = document.createElementNS(SVG_NS, "rect");
        rect.setAttribute("x", cx - RECT_SIZE / 2);
        rect.setAttribute("y", cy - RECT_SIZE / 2);
        rect.setAttribute("width", RECT_SIZE);
        rect.setAttribute("height", RECT_SIZE);
        const inactive = node.active === false;
        rect.setAttribute("fill", "#ffffff");
        rect.setAttribute("stroke", inactive ? "#c3ccd2" : "#183D4E");
        rect.setAttribute("stroke-width", column.kind === "input" ? 2 : 1);
        rect.setAttribute("rx", 3);
        if (inactive) rect.setAttribute("stroke-dasharray", "3 2");

        if (column.kind === "input") {
          // 原版的交互特点：切换特征靠**点网络图最左列的方块**，不是复选框。
          // 所以未选中的特征也必须画出来 —— 藏起来就没得点了。
          rect.style.cursor = "pointer";
          rect.addEventListener("click", () => this.onToggleFeature(columnIndex, nodeIndex));
        }

        const text = document.createElementNS(SVG_NS, "text");
        // 输入列用 text-anchor: end 把文字右对齐到方块右缘，长标签（sin(X₁)、X₁²）
        // 就向左溢出到节点左侧的空白区，而不是撑破方块。原版就是这么处理的。
        const isInput = column.kind === "input";
        text.setAttribute("x", isInput ? cx + RECT_SIZE / 2 : cx);
        text.setAttribute("y", cy + LABEL_FONT / 2 - 1);
        text.setAttribute("text-anchor", isInput ? "end" : "middle");
        text.setAttribute("font-size", LABEL_FONT);
        text.setAttribute("fill", inactive ? "#a9b4ba" : "#333");
        text.setAttribute("pointer-events", "none");
        for (const part of labelParts(node.label)) {
          const tspan = document.createElementNS(SVG_NS, "tspan");
          tspan.textContent = part.text;
          if (part.shift === "sub") tspan.setAttribute("baseline-shift", "sub");
          if (part.shift === "super") tspan.setAttribute("baseline-shift", "super");
          text.append(tspan);
        }

        nodeGroup.append(rect, text);
        return { rect, text, cx, cy };
      });
    });
  }

  // --- 每帧更新 -----------------------------------------------------------

  updateLinks(graph) {
    graph.edges.forEach((edges, layerIndex) => {
      edges.forEach((edge, edgeIndex) => {
        const el = this.linkEls?.[layerIndex]?.[edgeIndex];
        if (!el) return;
        const payload = edge.payload;
        if (payload.kind === "scalar") {
          el.setAttribute("stroke", colorFor(payload.value));
          el.setAttribute("stroke-width", linkWidth(payload.value));
          // 被 L1 杀死的连接：原版直接不画
          el.setAttribute("opacity", payload.dead ? 0 : 1);
        } else if (payload.kind === "curve") {
          // KAN：线宽固定（参考实现里线宽不编码重要性），透明度编码 |φ|₁
          el.setAttribute("stroke", "#183D4E");
          el.setAttribute("stroke-width", 1.5);
          el.setAttribute("opacity", String(0.15 + 0.85 * Math.min(1, payload.importance)));
        } else if (payload.kind === "product") {
          el.setAttribute("stroke", colorFor(payload.weight));
          el.setAttribute("stroke-width", linkWidth(payload.weight));
        } else {
          el.setAttribute("stroke", "#aaa");
          el.setAttribute("stroke-width", 1);
          el.setAttribute("stroke-dasharray", "4 3");
        }
      });
    });
  }

  drawHeatmaps(heatmaps) {
    const { width, height } = this.layout;
    this.ctx.clearRect(0, 0, width, height);
    heatmaps.forEach((encoded, columnIndex) => {
      if (!encoded) return;
      this.layout.columns[columnIndex].nodes.forEach((node, nodeIndex) => {
        // 未选中的特征仍然画出它的取值分布（淡显），让你能预览这个特征长什么样，
        // 再决定要不要打开它 —— 但明显的淡显不会让人误以为它参与了训练。
        const column = this.graph.columns[columnIndex];
        const inactive = column.kind === "input" && column.nodes[nodeIndex].active === false;
        this.ctx.globalAlpha = inactive ? 0.3 : 1;
        blitHeatmap(
          this.ctx,
          encoded,
          nodeIndex,
          node.cx - RECT_SIZE / 2 + 1,
          node.cy - RECT_SIZE / 2 + 1,
          RECT_SIZE - 2,
        );
        this.ctx.globalAlpha = 1;
      });
    });
  }

  /**
   * 画「边是一个函数」的那些结构（KAN）的函数曲线。
   *
   * 位置选在连线的中点：连线是一条对称的 S 形贝塞尔，它的中点**正好**是
   * 两端节点的中点，所以直接取 (src+dst)/2 即可，不用去解贝塞尔方程。
   *
   * 画在 canvas 覆盖层上（它已经在 SVG 之上、且 pointer-events: none），
   * 所以既能盖住连线、又不影响连线上的悬停命中。
   */
  drawEdgeCurves(graph) {
    graph.edges.forEach((edges, layerIndex) => {
      if (!edges.some((edge) => edge.payload.kind === "curve")) return;

      const from = this.layout.columns[layerIndex];
      const to = this.layout.columns[layerIndex + 1];
      const maxImportance = Math.max(...edges.map((e) => e.payload.importance ?? 0), 1e-9);
      const scale = curveScale(edges);
      const size = this.sparklineSize(from.cx, to.cx);

      // 按重要性升序画：重叠时重要的曲线压在上面，不会被次要的盖掉
      for (const edge of [...edges].sort((a, b) => a.payload.importance - b.payload.importance)) {
        const src = from.nodes[edge.src];
        const dst = to.nodes[edge.dst];
        drawSparkline(
          this.ctx,
          edge.payload,
          (src.cx + dst.cx) / 2,
          (src.cy + dst.cy) / 2,
          size.width,
          size.height,
          scale,
          edge.payload.importance / maxImportance,
        );
      }
    });
  }

  /**
   * 迷你曲线图该画多大。
   *
   * 横向可以奢侈一点（两列之间空档很宽），纵向必须抠。关键约束**不是**
   * 节点间距 55，而是两条边的中点之间能有多近：
   * 源节点在 `a + i·55`、目标节点在 `b + j·55`，中点就落在
   * `(a+b)/2 + (i+j)·27.5` —— 因为 i+j 是整数，**相邻中点永远差 27.5**，
   * 和两边各有多少节点无关。
   *
   * 所以高度必须小于 27.5。按 55 去算的话，8 个盒子会糊成一整块白板，
   * 反而把网络结构盖住 —— 这个坑踩过一次了。
   */
  sparklineSize(fromCx, toCx) {
    const gap = Math.max(toCx - fromCx, SPARKLINE_MIN_WIDTH);
    const width = Math.max(SPARKLINE_MIN_WIDTH, Math.min(SPARKLINE_MAX_WIDTH, gap * 0.42));
    // 高度用满 27.5 的间距：描白边的画法下，相邻曲线轻微相切也能分辨，
    // 而矮一档就看不出函数形状了 —— 形状才是这里要传达的信息。
    const height = Math.max(16, Math.min(width * 0.55, (VERTICAL_STEP / 2) * 0.95));
    return { width, height };
  }
}

/**
 * 整层共用一个纵向标度，这样边与边之间的**相对幅度**是可见的。
 *
 * 如果每条曲线各自归一化，一个几乎为零的边看起来也会和主导边一样"有形状"，
 * 那是最容易误导人的画法。共用一个标度则"平直的线"就真的是平的 ——
 * 配合透明度编码的重要性，"又平又淡"就准确表达了"这条边没在干活"。
 */
function curveScale(edges) {
  let max = 0;
  for (const edge of edges) {
    for (const value of edge.payload.ys) {
      const magnitude = Math.abs(value);
      if (magnitude > max) max = magnitude;
    }
  }
  return max > 1e-6 ? max : 1;
}

function drawSparkline(ctx, payload, cx, cy, width, height, scale, normalizedImportance) {
  const ys = payload.ys;
  if (!ys || ys.length < 2) return;

  const x0 = cx - width / 2;
  const y0 = cy - height / 2;
  const halfHeight = height / 2;
  const toY = (value) => cy - (value / scale) * halfHeight * 0.92;

  // 参考实现的做法：**透明度**编码重要性，线宽保持不变。
  // tanh 在 0..1 中段更陡，让中等重要性的边也能拉开差距。
  const alpha = 0.25 + 0.75 * Math.tanh(3 * normalizedImportance);

  const tracePath = () => {
    ctx.beginPath();
    for (let i = 0; i < ys.length; i++) {
      const px = x0 + (i / (ys.length - 1)) * width;
      const py = Math.max(y0 + 1, Math.min(y0 + height - 1, toY(ys[i])));
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    }
  };

  ctx.save();
  ctx.lineJoin = "round";
  ctx.lineCap = "round";

  // φ = 0 的基准线：判断这条边整体偏正还是偏负
  ctx.globalAlpha = alpha * 0.3;
  ctx.strokeStyle = "#183D4E";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0, cy);
  ctx.lineTo(x0 + width, cy);
  ctx.stroke();

  // 先描一层白色粗边，再在上面画曲线。
  //
  // 一开始这里画的是白底方框，结果一层 8 条边的小方框会连成一整块白板，
  // 把网络结构整个盖住 —— 而连线本身也是有用的信息（它说明哪条边连向哪个节点）。
  // 描白边只压住曲线自己那一圈，两者都能看清。
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "rgba(255,255,255,0.92)";
  ctx.lineWidth = 3.6;
  tracePath();
  ctx.stroke();

  ctx.globalAlpha = alpha;
  ctx.strokeStyle = "#183D4E";
  ctx.lineWidth = 1.7;
  tracePath();
  ctx.stroke();
  ctx.restore();
}

function structureSignature(graph) {
  // 节点的 active 也要进签名：切换特征时列宽不变，但方块的样式必须重画。
  const columns = graph.columns
    .map((c) => `${c.kind}:${c.nodes.map((n) => (n.active === false ? "0" : "1")).join("")}`)
    .join("|");
  const edges = graph.edges.map((e) => e.length).join(",");
  return `${columns}#${edges}`;
}
