/**
 * Fitness per generation: best and mean, drawn as thin lines with labels at their ends in
 * place of a legend. Redrawn once per generation at the container's current pixel width, so
 * text keeps its size on narrow screens.
 *
 * The accent colour is used for the best line only. Text stays in ink: the accent is below
 * the 4.5:1 contrast needed for small text.
 */
import type { GenerationStat } from "../evolve/trainer";
import { formatInt, niceCeil, svg } from "./svg";

const HEIGHT = 260;
const MARGIN = { top: 30, right: 92, bottom: 38, left: 52 };

export class FitnessChart {
  private readonly root: SVGSVGElement;
  private last: readonly GenerationStat[] = [];

  constructor(private readonly container: HTMLElement) {
    this.root = svg("svg", {
      class: "chart-svg",
      role: "img",
      "aria-label": "Line chart of best and mean fitness by generation",
    });
    container.appendChild(this.root);
    new ResizeObserver(() => this.render(this.last)).observe(container);
  }

  render(history: readonly GenerationStat[]): void {
    this.last = history;
    const width = Math.max(280, this.container.clientWidth);
    const root = this.root;
    root.replaceChildren();
    root.setAttribute("viewBox", `0 0 ${width} ${HEIGHT}`);
    root.setAttribute("width", String(width));
    root.setAttribute("height", String(HEIGHT));

    const plotW = width - MARGIN.left - MARGIN.right;
    const plotH = HEIGHT - MARGIN.top - MARGIN.bottom;
    const maxGen = Math.max(10, history.length);
    const yMax = niceCeil(Math.max(10, ...history.map((h) => h.best)));
    const x = (gen: number) => MARGIN.left + (plotW * (gen - 1)) / (maxGen - 1);
    const y = (v: number) => MARGIN.top + plotH * (1 - v / yMax);

    // Horizontal guides at 0, half and full scale. Light, so the lines stay the subject.
    for (const frac of [0, 0.5, 1]) {
      const value = yMax * frac;
      svg("line", { x1: MARGIN.left, x2: MARGIN.left + plotW, y1: y(value), y2: y(value), class: "grid" }, root);
      svg("text", { x: MARGIN.left - 8, y: y(value) + 4, "text-anchor": "end", class: "tick" }, root, formatInt(value));
    }
    svg("text", { x: MARGIN.left, y: HEIGHT - 4, class: "axis-label" }, root, "generation");
    svg("text", { x: 0, y: 10, class: "axis-label" }, root, "fitness (score)");
    for (const gen of ticksFor(maxGen)) {
      svg("text", { x: x(gen), y: HEIGHT - 20, "text-anchor": "middle", class: "tick" }, root, String(gen));
    }

    if (history.length === 0) {
      svg("text", { x: MARGIN.left + 8, y: MARGIN.top + plotH / 2, class: "tick" }, root, "waiting for generation 1");
      return;
    }

    const path = (pick: (h: GenerationStat) => number) =>
      history.map((h, i) => `${i === 0 ? "M" : "L"}${x(h.generation).toFixed(1)} ${y(pick(h)).toFixed(1)}`).join(" ");
    svg("path", { d: path((h) => h.mean), class: "line-mean" }, root);
    svg("path", { d: path((h) => h.best), class: "line-best" }, root);

    const end = history[history.length - 1];
    const labelX = x(end.generation) + 8;
    // Keep the two end labels apart when the lines are close.
    let bestY = y(end.best) + 4;
    let meanY = y(end.mean) + 4;
    if (Math.abs(bestY - meanY) < 15) {
      const mid = (bestY + meanY) / 2;
      bestY = mid - 8;
      meanY = mid + 8;
    }
    svg("circle", { cx: x(end.generation), cy: y(end.best), r: 3, class: "dot-best" }, root);
    svg("text", { x: labelX, y: bestY, class: "end-label" }, root, `best ${formatInt(end.best)}`);
    svg("text", { x: labelX, y: meanY, class: "end-label muted" }, root, `mean ${formatInt(end.mean)}`);
  }
}

/** Round generation numbers for the x axis. Generation 1 is the left edge and has no tick. */
function ticksFor(maxGen: number): number[] {
  const step = niceCeil(maxGen / 5);
  const ticks: number[] = [];
  for (let g = step; g <= maxGen; g += step) ticks.push(g);
  return ticks;
}
