// loss 曲线。照抄原版的行为：
//   * 横轴 = epoch 序号，最新一个点永远贴右缘（x 轴的 domain 是 [1, 数据点个数]）
//   * 纵轴随数据自适应，线性（不是对数）
//   * 每个 epoch 一个点，不抽稀、不做移动平均
//   * train 灰 #777、test 黑，线宽 1.5px，没有坐标轴、没有网格线
//
// 没有坐标轴刻度是有意的：这张图的用途是看"两条线的相对关系"（有没有分开、
// 有没有震荡），而不是读具体数值 —— 精确数值在旁边的文字里有。

const TRAIN_COLOR = "#777";
const TEST_COLOR = "black";

export class LossChart {
  constructor(canvas) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d");
    this.reset();
  }

  reset() {
    this.lossTrain = [];
    this.lossTest = [];
    this.redraw();
  }

  addPoint(lossTrain, lossTest) {
    this.lossTrain.push(lossTrain);
    this.lossTest.push(lossTest);
    this.redraw();
  }

  redraw() {
    const { ctx, canvas } = this;
    const width = canvas.width;
    const height = canvas.height;
    ctx.clearRect(0, 0, width, height);

    const n = this.lossTrain.length;
    if (n < 2) return;

    const all = [...this.lossTrain, ...this.lossTest];
    let minY = Math.min(...all);
    let maxY = Math.max(...all);
    if (maxY - minY < 1e-9) {
      // 完全平的时候给一点上下留白，否则除以 0
      maxY = minY + 1;
      minY -= 1;
    }
    const pad = (maxY - minY) * 0.08;
    minY -= pad;
    maxY += pad;

    const xAt = (i) => (i / (n - 1)) * width;
    const yAt = (v) => height - ((v - minY) / (maxY - minY)) * height;

    ctx.lineWidth = 1.5;
    ctx.lineJoin = "round";
    this.drawLine(this.lossTrain, xAt, yAt, TRAIN_COLOR);
    this.drawLine(this.lossTest, xAt, yAt, TEST_COLOR);
  }

  drawLine(values, xAt, yAt, color) {
    const { ctx } = this;
    ctx.beginPath();
    ctx.strokeStyle = color;
    ctx.moveTo(xAt(0), yAt(values[0]));
    for (let i = 1; i < values.length; i++) {
      ctx.lineTo(xAt(i), yAt(values[i]));
    }
    ctx.stroke();
  }
}
