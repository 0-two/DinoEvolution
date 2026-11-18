/**
 * A small line chart for the explainer widgets: thin line, labelled axes, one highlighted point.
 * Same look as the other charts (light grid, mono text, accent for the point being discussed).
 */
import { svg } from "./svg";

export interface MiniChartOptions {
  xLabel: string;
  yLabel: string;
  xTicks: number[];
  yTicks: number[];
  xFormat?: (v: number) => string;
  yFormat?: (v: number) => string;
  /** Log scale on x, for quantities that span orders of magnitude. */
  logX?: boolean;
  /** A horizontal reference line, e.g. 1 for "same as the parent". */
  reference?: { y: number; label: string };
}

const W = 380;
const H = 210;
const M = { top: 18, right: 16, bottom: 40, left: 52 };

export class MiniChart {
  private readonly root: SVGSVGElement;
  private points: [number, number][] = [];
  private highlight: number | null = null;

  constructor(
    container: HTMLElement,
    private readonly opts: MiniChartOptions,
    label: string,
  ) {
    this.root = svg("svg", {
      class: "chart-svg mini",
      viewBox: `0 0 ${W} ${H}`,
      role: "img",
      "aria-label": label,
    });
    container.appendChild(this.root);
    this.draw();
  }

  setPoints(points: [number, number][]): void {
    this.points = points;
    this.draw();
  }

  /** Highlight the point whose x is nearest `x`, drawn in the accent colour. */
  setHighlight(x: number | null): void {
    this.highlight = x;
    this.draw();
  }

  private sx(v: number): number {
    const { xTicks, logX } = this.opts;
    const lo = Math.min(...xTicks);
    const hi = Math.max(...xTicks);
    const t = logX ? (Math.log(v) - Math.log(lo)) / (Math.log(hi) - Math.log(lo)) : (v - lo) / (hi - lo);
    return M.left + t * (W - M.left - M.right);
  }

  private sy(v: number): number {
    const lo = Math.min(...this.opts.yTicks);
    const hi = Math.max(...this.opts.yTicks);
    return M.top + (1 - (v - lo) / (hi - lo)) * (H - M.top - M.bottom);
  }

  private draw(): void {
    const root = this.root;
    const { xTicks, yTicks, xLabel, yLabel, reference } = this.opts;
    const xf = this.opts.xFormat ?? ((v: number) => String(v));
    const yf = this.opts.yFormat ?? ((v: number) => String(v));
    root.replaceChildren();
    for (const v of yTicks) {
      svg("line", { x1: M.left, x2: W - M.right, y1: this.sy(v), y2: this.sy(v), class: "grid" }, root);
      svg("text", { x: M.left - 8, y: this.sy(v) + 4, "text-anchor": "end", class: "tick" }, root, yf(v));
    }
    for (const v of xTicks) {
      svg("text", { x: this.sx(v), y: H - 22, "text-anchor": "middle", class: "tick" }, root, xf(v));
    }
    svg("text", { x: W - M.right, y: H - 6, "text-anchor": "end", class: "axis-label" }, root, xLabel);
    svg("text", { x: 0, y: 10, class: "axis-label" }, root, yLabel);
    if (reference !== undefined) {
      svg("line", { x1: M.left, x2: W - M.right, y1: this.sy(reference.y), y2: this.sy(reference.y), class: "ref-line" }, root);
      svg("text", { x: W - M.right, y: this.sy(reference.y) - 5, "text-anchor": "end", class: "tick" }, root, reference.label);
    }
    if (this.points.length === 0) return;
    const d = this.points.map(([x, y], i) => `${i === 0 ? "M" : "L"}${this.sx(x).toFixed(1)} ${this.sy(y).toFixed(1)}`).join(" ");
    svg("path", { d, class: "line-mean" }, root);
    for (const [x, y] of this.points) svg("circle", { cx: this.sx(x), cy: this.sy(y), r: 2.5, class: "dot-ink" }, root);
    if (this.highlight !== null) {
      const hl = this.highlight;
      const nearest = this.points.reduce((best, p) => (Math.abs(p[0] - hl) < Math.abs(best[0] - hl) ? p : best));
      svg("circle", { cx: this.sx(nearest[0]), cy: this.sy(nearest[1]), r: 5, class: "dot-best" }, root);
    }
  }
}
