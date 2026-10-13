// 主控制器：WebSocket 连接、控件绑定、各视图的调度。
//
// 数据流是单向的：任何交互 → 发一条 cmd 给服务端 → 服务端回 frame → 重画。
// 前端不维护训练状态（唯一的例外是 loss 曲线的历史点，那是纯显示用的）。

import { colorFor } from "./render/palette.js";
import { drawHeatmap } from "./render/heatmap.js";
import { LossChart } from "./render/losschart.js";
import { NetworkRenderer } from "./render/network.js";
import { FitPlot, curveValues } from "./render/curves.js";

const state = {
  config: {},
  info: null, // 服务端发来的静态信息
  train: [],
  test: [],
  showTestData: true,
  discretize: false,
  lastIter: -1,
  playing: false,
  autoplayStarted: false,
  // "2d" 画决策边界，"1d" 画函数拟合。由服务端在 data/frame 消息里给，
  // 前端不自己推导 —— 数据集的分辨率、网格点数、输出面板布局都是跟着它走的。
  mode: "2d",
  domain: [-6, 6],
  gridX: null,
  truth: null,
};

let ws = null;
let reconnectTimer = null;
// 最近一帧。切换纯显示选项时用它原地重画，不用再向服务端要一次。
let lastFrame = null;

/**
 * 从 URL 查询参数读初始配置，例如
 *   ?dataset=spiral&shape=8,8&model=mlp&lr=0.1&play=1
 * 方便把一次实验直接存成书签或分享出去（原版也把状态写进 URL）。
 * 只支持这几个字段，其余一律以服务端默认值为准。
 */
function initialConfigFromUrl() {
  const params = new URLSearchParams(location.search);
  const config = {};
  if (params.has("dataset")) config.dataset = params.get("dataset");
  if (params.has("model")) config.model = params.get("model");
  if (params.has("lr")) config.learningRate = Number(params.get("lr"));
  if (params.has("activation")) config.activation = params.get("activation");
  if (params.has("shape")) {
    config.networkShape = params
      .get("shape")
      .split(",")
      .map((n) => parseInt(n, 10))
      .filter((n) => Number.isFinite(n) && n > 0);
  }
  if (params.has("features")) {
    // 不在这里校验合法性：此时还没收到 static 消息，拿不到特征清单；
    // 服务端会过滤掉不认识的名字，这里原样传过去即可。
    // 顺序就是模型输入列的顺序，所以要保留。
    config.features = params.get("features").split(",").filter(Boolean);
  }
  return config;
}

// ── WebSocket ────────────────────────────────────────────────────────────

function connect() {
  const proto = location.protocol === "https:" ? "wss:" : "ws:";
  ws = new WebSocket(`${proto}//${location.host}/ws`);

  ws.onopen = () => {
    setStatus("");
    send({ cmd: "init", config: { ...initialConfigFromUrl(), ...state.config } });
  };

  ws.onmessage = (event) => handleMessage(JSON.parse(event.data));

  ws.onclose = () => {
    setStatus("连接断开，正在重连…", "error");
    clearTimeout(reconnectTimer);
    reconnectTimer = setTimeout(connect, 1500);
  };

  ws.onerror = () => ws.close();
}

function send(message) {
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(message));
}

function handleMessage(message) {
  switch (message.type) {
    case "static":
      state.info = message;
      renderStatic(message);
      break;
    case "data":
      state.train = message.train;
      state.test = message.test;
      state.mode = message.mode || "2d";
      state.domain = message.domain || [-6, 6];
      state.gridX = message.gridX || null;
      state.truth = message.truth || null;
      applyMode(state.mode);
      if (state.mode === "1d") fitPlot.setData(message);
      drawPoints();
      break;
    case "frame":
      renderFrame(message);
      // ?play=1 让页面打开就自动开始训练。演示和截图都方便。
      if (!state.autoplayStarted && new URLSearchParams(location.search).get("play") === "1") {
        state.autoplayStarted = true;
        send({ cmd: "play" });
      }
      break;
    case "error":
      setStatus(message.message, "error");
      break;
  }
}

function setStatus(text, kind = "") {
  const el = document.getElementById("status");
  el.textContent = text;
  el.className = `status ${kind}`;
}

// ── 静态信息：数据集缩略图、下拉框、模型列表 ─────────────────────────────

function renderStatic(info) {
  // 一切以服务端的生效配置为准。前端启动时发的是空配置，如果这里不接回来，
  // 后续读 state.config.learningRate 之类全是 undefined。
  state.config = info.config;
  renderDatasetThumbs(info.datasets);
  renderSelect("lr", info.options.learningRate, state.config.learningRate, (v) =>
    send({ cmd: "config", config: { learningRate: Number(v) } }),
  );
  renderSelect("activation", info.options.activation, state.config.activation, (v) =>
    send({ cmd: "config", config: { activation: v } }),
  );
  renderSelect("optimizer", info.options.optimizer, state.config.optimizer, (v) =>
    send({ cmd: "config", config: { optimizer: v } }),
  );
  renderSelect("reg", info.options.regularization, state.config.regularization, (v) =>
    send({ cmd: "config", config: { regularization: v } }),
  );
  renderSelect("regRate", info.options.regularizationRate, state.config.regularizationRate, (v) =>
    send({ cmd: "config", config: { regularizationRate: Number(v) } }),
  );
  renderModelSelect(info.models);
  renderModelErrors(info.modelErrors);
  syncControlsFromConfig(state.config);
  updateShapeControls();
}

/**
 * 把服务端的生效配置回填到各个控件上。
 *
 * 没有这一步的话，用 URL 参数（?dataset=spiral&shape=8,8）打开页面时，
 * 模型确实按参数跑了，但控件还显示着默认值 —— 界面在骗人。
 */
function syncControlsFromConfig(config) {
  const setSlider = (id, displayValue) => {
    const slider = document.getElementById(id);
    slider.value = String(displayValue);
    document.getElementById(`val-${id}`).textContent = String(displayValue);
  };
  // noise 在配置里是 0–0.5，滑条上是 0–50
  setSlider("noise", Math.round((config.noise ?? 0) * 100));
  setSlider("ratio", Math.round(config.percTrainData ?? 50));
  setSlider("batch", config.batchSize ?? 10);

  const select = (id, value) => {
    const el = document.getElementById(id);
    if (value !== undefined && value !== null) el.value = String(value);
  };
  select("lr", config.learningRate);
  select("activation", config.activation);
  select("optimizer", config.optimizer);
  select("reg", config.regularization);
  select("regRate", config.regularizationRate);
  select("model", config.model);

  applyProblemFilter(config.problem || "classification");
  for (const thumb of document.querySelectorAll(".thumb")) {
    thumb.classList.toggle("active", thumb.dataset.id === config.dataset);
  }
}

function renderSelect(id, options, current, onChange) {
  const select = document.getElementById(id);
  select.innerHTML = "";
  for (const option of options) {
    const el = document.createElement("option");
    el.value = String(option);
    el.textContent = String(option);
    select.append(el);
  }
  if (current !== undefined && current !== null) select.value = String(current);
  select.onchange = () => onChange(select.value);
}

function renderModelSelect(models) {
  const select = document.getElementById("model");
  select.innerHTML = "";
  for (const model of models) {
    const el = document.createElement("option");
    el.value = model.key;
    el.textContent = model.title;
    el.title = model.description || "";
    select.append(el);
  }
  if (state.config.model) select.value = state.config.model;

  const updateDesc = () => {
    const spec = models.find((m) => m.key === select.value);
    document.getElementById("model-desc").textContent = spec?.description || "";
    // 选中模型自带建议超参（比如 KAN 建议更小的学习率）时，一并应用
    if (spec?.defaults && Object.keys(spec.defaults).length) {
      send({ cmd: "config", config: { model: select.value, ...spec.defaults } });
    } else {
      send({ cmd: "config", config: { model: select.value } });
    }
  };

  select.onchange = updateDesc;
  const spec = models.find((m) => m.key === select.value);
  document.getElementById("model-desc").textContent = spec?.description || "";
}

function renderModelErrors(errors) {
  const el = document.getElementById("model-errors");
  const entries = Object.entries(errors || {});
  el.textContent = entries.length
    ? entries.map(([file, err]) => `⚠ ${file} 加载失败：\n${err}`).join("\n\n")
    : "";
}

function renderDatasetThumbs(datasets) {
  const container = document.getElementById("datasets");
  container.innerHTML = "";

  for (const dataset of datasets) {
    const div = document.createElement("div");
    div.className = "thumb";
    div.dataset.id = dataset.id;
    div.dataset.problem = dataset.problem;
    div.dataset.mode = dataset.mode;
    if (dataset.id === state.config.dataset) div.classList.add("active");

    const canvas = document.createElement("canvas");
    canvas.width = 60;
    canvas.height = 60;
    if (dataset.mode === "1d") drawThumbnail1D(canvas, dataset.thumbnail, dataset.domain);
    else drawThumbnail(canvas, dataset.thumbnail);

    const label = document.createElement("span");
    label.textContent = dataset.title;

    div.append(canvas, label);
    div.onclick = () => {
      state.config.dataset = dataset.id;
      send({ cmd: "config", config: { dataset: dataset.id } });
    };
    container.append(div);
  }

  applyProblemFilter(state.config.problem || "classification");
}

function drawThumbnail(canvas, points) {
  const ctx = canvas.getContext("2d");
  const size = canvas.width;
  ctx.clearRect(0, 0, size, size);
  for (const [x, y, label] of points) {
    ctx.fillStyle = colorFor(label);
    // 缩略图统一用主图的朝向（y 向上）。原版这里 y 轴是反的，
    // 于是螺旋缩略图和主图是镜像的 —— 属于原版的无心之失，不复刻。
    ctx.fillRect(
      (size * (x + 6)) / 12 - 2,
      (size * (6 - y)) / 12 - 2,
      4,
      4,
    );
  }
}

/**
 * 一维数据集的缩略图：把 f(x) 画成一条折线。
 *
 * 二维缩略图按 (x, y) 散点画，而一维数据的 y 恒为 0 —— 那样画出来会是
 * 贴着中线的一条彩色直线，什么也看不出来。
 */
function drawThumbnail1D(canvas, points, domain) {
  const ctx = canvas.getContext("2d");
  const size = canvas.width;
  const [lo, hi] = domain || [-3, 3];
  ctx.clearRect(0, 0, size, size);

  const labels = points.map((p) => p[2]);
  let yLo = Math.min(...labels);
  let yHi = Math.max(...labels);
  if (yHi - yLo < 1e-6) {
    yHi += 1;
    yLo -= 1;
  }
  const pad = (yHi - yLo) * 0.12;
  yLo -= pad;
  yHi += pad;

  const toPx = (x) => ((x - lo) / (hi - lo)) * size;
  const toPy = (y) => size - ((y - yLo) / (yHi - yLo)) * size;

  // 按 x 排序再连线 —— 缩略图的点是无序的，连错了会得到一团乱麻
  const sorted = [...points].sort((a, b) => a[0] - b[0]);
  ctx.beginPath();
  ctx.strokeStyle = "#183D4E";
  ctx.lineWidth = 1.5;
  ctx.lineJoin = "round";
  for (let i = 0; i < sorted.length; i++) {
    const px = toPx(sorted[i][0]);
    const py = toPy(sorted[i][2]);
    if (i === 0) ctx.moveTo(px, py);
    else ctx.lineTo(px, py);
  }
  ctx.stroke();
}

function applyProblemFilter(problem) {
  for (const thumb of document.querySelectorAll(".thumb")) {
    thumb.style.display = thumb.dataset.problem === problem ? "" : "none";
  }
  for (const button of document.querySelectorAll("#problem-tabs button")) {
    button.classList.toggle("active", button.dataset.problem === problem);
  }
}

// ── 帧渲染 ───────────────────────────────────────────────────────────────

const network = new NetworkRenderer(
  document.getElementById("net-svg"),
  document.getElementById("net-heat"),
  {
    onToggleFeature: (_, nodeIndex) => toggleFeature(nodeIndex),
    onHoverLink: (payload, x, y) => showHovercard(payload, x, y),
  },
);

const lossChart = new LossChart(document.getElementById("losschart"));
const fitPlot = new FitPlot(document.getElementById("fitplot"));

/** 只重画、不推进任何状态。用于切换纯显示选项（显示测试数据、离散化）。 */
function requestFrameRedraw() {
  if (lastFrame) renderFrame(lastFrame, { redrawOnly: true });
}

function renderFrame(frame, { redrawOnly = false } = {}) {
  lastFrame = frame;

  document.getElementById("iter").textContent = String(frame.iter).padStart(6, "0");
  document.getElementById("loss-train").textContent = frame.lossTrain.toFixed(3);
  document.getElementById("loss-test").textContent = frame.lossTest.toFixed(3);
  setStatus("");

  if (!redrawOnly) {
    // iter 归零说明引擎重置过（Reset / 换数据集 / 换模型），曲线也跟着重来。
    // 调学习率之类的不会重置 iter，所以曲线不会被清掉 —— 这是有意的。
    if (frame.iter <= state.lastIter && frame.iter <= 1) lossChart.reset();
    state.lastIter = frame.iter;
    lossChart.addPoint(frame.lossTrain, frame.lossTest);
  }

  const mode = frame.mode || "2d";
  if (mode === "1d") {
    network.render(frame.graph, frame.curves, "1d");
    // 输出层的曲线就是模型学到的函数，直接画到拟合图上和真实函数对比
    const output = frame.curves[frame.curves.length - 1];
    fitPlot.draw(Array.from(curveValues(output, 0)), visiblePoints());
  } else {
    network.render(frame.graph, frame.heatmaps, "2d");
    // 最后一列是输出层，它的全分辨率热力图就是主决策边界
    drawHeatmap(
      document.getElementById("heatmap"),
      frame.heatmaps[frame.heatmaps.length - 1],
      0,
      state.discretize,
    );
  }
}

/** 当前该显示的数据点：训练集 + （勾选时）测试集。 */
function visiblePoints() {
  return state.showTestData ? [...state.train, ...state.test] : state.train;
}

/**
 * 按数据集模式切换界面。
 *
 * 二维模式有"颜色 = 输出值"这套映射，所以有色标、有离散化开关；
 * 一维模式画的是曲线，这些都没有意义，换成曲线图例。
 */
function applyMode(mode) {
  const is1d = mode === "1d";
  document.getElementById("heatmap-stack").hidden = is1d;
  document.getElementById("fitplot").hidden = !is1d;
  document.getElementById("colormap").hidden = is1d;
  document.getElementById("fit-legend").hidden = !is1d;
  document.getElementById("discretize-row").hidden = is1d;
}

function drawPoints() {
  const canvas = document.getElementById("points");
  const ctx = canvas.getContext("2d");
  const size = canvas.width;
  ctx.clearRect(0, 0, size, size);

  // 一维模式的数据点画在拟合图里（那里才有正确的坐标轴），这里留空
  if (state.mode === "1d") return;

  const [lo, hi] = state.domain;
  const span = hi - lo;
  const toPx = (x) => ((x - lo) / span) * size;
  const toPy = (y) => ((hi - y) / span) * size;

  const draw = (points, isTest) => {
    ctx.strokeStyle = isTest ? "black" : "white";
    ctx.lineWidth = 1;
    ctx.globalAlpha = isTest ? 0.6 : 0.8;
    for (const [x, y, label] of points) {
      ctx.beginPath();
      ctx.arc(toPx(x), toPy(y), 3, 0, Math.PI * 2);
      ctx.fillStyle = colorFor(label);
      ctx.fill();
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  };

  draw(state.train, false);
  if (state.showTestData) draw(state.test, true);
}

function showHovercard(payload, x, y) {
  const card = document.getElementById("hovercard");
  if (!payload) {
    card.style.display = "none";
    return;
  }
  if (payload.kind === "scalar") {
    card.textContent = `Weight ${payload.value.toFixed(3)}${payload.dead ? "  (已剪枝)" : ""}`;
  } else if (payload.kind === "curve") {
    // KAN 的边是一个函数，所以要说清"在哪个区间上"以及"有多重要"。
    // 重要性 |φ|₁ 是这一层的**局部**指标（论文 Eq.2.17），不是对最终输出的归因。
    const lo = payload.xs[0].toFixed(2);
    const hi = payload.xs[payload.xs.length - 1].toFixed(2);
    card.textContent = `φ(x)  x ∈ [${lo}, ${hi}]   |φ|₁ = ${payload.importance.toFixed(3)}`;
  } else {
    card.textContent = payload.label || payload.kind;
  }
  const rect = document.getElementById("network").getBoundingClientRect();
  card.style.display = "block";
  card.style.left = `${rect.left + x + 14}px`;
  card.style.top = `${rect.top + y}px`;
}

// ── 交互 ─────────────────────────────────────────────────────────────────

function toggleFeature(nodeIndex) {
  const features = [...(state.config.features || [])];
  // 输入列的下标是**全部 7 个特征**里的位置，不是已选特征的序号
  const all = state.info.features.map((f) => f.id);
  const featureId = all[nodeIndex];
  if (!featureId) return;

  const at = features.indexOf(featureId);
  if (at >= 0) {
    if (features.length <= 1) return; // 至少留一个特征，否则模型没有输入
    features.splice(at, 1);
  } else {
    features.push(featureId);
  }
  state.config.features = features;
  send({ cmd: "config", config: { features } });
}

function currentShape() {
  return [...(state.config.networkShape || [4])];
}

function sendShape(shape) {
  state.config.networkShape = shape;
  send({ cmd: "config", config: { networkShape: shape } });
}

function updateShapeControls() {
  const spec = state.info?.models.find((m) => m.key === state.config.model);
  const enabled = !!spec?.configurableShape;
  const controls = document.getElementById("shape-controls");
  controls.style.opacity = enabled ? "1" : "0.4";
  for (const button of controls.querySelectorAll("button")) button.disabled = !enabled;
  document.getElementById("num-layers").textContent = String(currentShape().length);
  document.getElementById("num-neurons").textContent = String(currentShape().at(-1) ?? 0);
}

function bindControls() {
  const on = (id, event, handler) => document.getElementById(id).addEventListener(event, handler);

  on("btn-play", "click", () => send({ cmd: "play" }));
  on("btn-pause", "click", () => send({ cmd: "pause" }));
  on("btn-step", "click", () => send({ cmd: "step" }));
  on("btn-reset", "click", () => send({ cmd: "reset" }));
  on("btn-regenerate", "click", () => send({ cmd: "regenerate" }));

  on("ratio", "input", (event) => {
    document.getElementById("val-ratio").textContent = event.target.value;
    send({ cmd: "config", config: { percTrainData: Number(event.target.value) } });
  });
  on("noise", "input", (event) => {
    // 滑条是 0–50 的显示单位，协议里一律用归一化的 0–0.5
    document.getElementById("val-noise").textContent = event.target.value;
    send({ cmd: "config", config: { noise: Number(event.target.value) / 100 } });
  });
  on("batch", "input", (event) => {
    document.getElementById("val-batch").textContent = event.target.value;
    send({ cmd: "config", config: { batchSize: Number(event.target.value) } });
  });

  on("show-test-data", "change", (event) => {
    state.showTestData = event.target.checked;
    drawPoints();
    // 一维模式的数据点是画在拟合图里的，得让它重画一次
    if (state.mode === "1d") requestFrameRedraw();
  });
  on("discretize", "change", (event) => {
    state.discretize = event.target.checked;
    // 纯显示选项，本地重画就行。之前这里是发 step 让服务端重发一帧，
    // 结果是"勾一下离散化"顺手多训了一个 epoch —— 调显示选项不该改变训练状态。
    requestFrameRedraw();
  });

  for (const button of document.querySelectorAll("#problem-tabs button")) {
    button.addEventListener("click", () => {
      const problem = button.dataset.problem;
      applyProblemFilter(problem);
      // 切问题类型必然要换数据集：默认挑该类型的第一个
      const first = state.info.datasets.find((d) => d.problem === problem);
      if (first) {
        state.config.dataset = first.id;
        send({ cmd: "config", config: { dataset: first.id } });
      }
    });
  }

  on("btn-add-layer", "click", () => {
    const shape = currentShape();
    if (shape.length >= 6) return; // 原版最多 6 个隐藏层
    sendShape([...shape, 2]);
    updateShapeControls();
  });
  on("btn-remove-layer", "click", () => {
    const shape = currentShape();
    if (shape.length === 0) return;
    sendShape(shape.slice(0, -1));
    updateShapeControls();
  });
  on("btn-add-neuron", "click", () => {
    const shape = currentShape();
    if (shape.length === 0) return;
    if (shape[shape.length - 1] >= 8) return; // 原版每层最多 8 个
    shape[shape.length - 1] += 1;
    sendShape(shape);
    updateShapeControls();
  });
  on("btn-remove-neuron", "click", () => {
    const shape = currentShape();
    if (shape.length === 0) return;
    if (shape[shape.length - 1] <= 1) return;
    shape[shape.length - 1] -= 1;
    sendShape(shape);
    updateShapeControls();
  });

  // 窗口缩放由 NetworkRenderer 内部的 ResizeObserver 处理（它盯着容器而不是
  // 窗口，侧边栏折叠之类的变化也能覆盖到）。
}

// ── 启动 ─────────────────────────────────────────────────────────────────

bindControls();
connect();
