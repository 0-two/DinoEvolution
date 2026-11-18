/**
 * Makes the report figures interactive. The figures are plain SVG files made by the Python
 * figure script; they carry their own data in attributes, so this file needs no chart code.
 *
 *  - `data-tip` on a mark: hover or focus shows its text.
 *  - `data-series` groups with `data-points`: hovering a plot area (or using the arrow keys on
 *    it) draws a crosshair and lists each visible series' value at that x.
 *  - `setSeriesVisible` hides or shows one series, its band and its end label.
 */
import { crosshairText, nearestIndex, xFromPointer, type Point } from "./figureMath";

export class Tooltip {
  readonly el: HTMLElement;

  constructor(parent: HTMLElement) {
    this.el = document.createElement("div");
    this.el.className = "tooltip";
    this.el.hidden = true;
    parent.appendChild(this.el);
  }

  show(text: string, clientX: number, clientY: number): void {
    this.el.textContent = text;
    this.el.hidden = false;
    const { width, height } = this.el.getBoundingClientRect();
    const x = Math.min(clientX + 14, window.innerWidth - width - 8);
    const y = clientY + 16 + height > window.innerHeight ? clientY - height - 12 : clientY + 16;
    this.el.style.left = `${Math.max(8, x)}px`;
    this.el.style.top = `${Math.max(8, y)}px`;
  }

  hide(): void {
    this.el.hidden = true;
  }
}

export function setSeriesVisible(svg: SVGSVGElement, id: string, visible: boolean): void {
  const display = visible ? "" : "none";
  const selector = `[data-series="${CSS.escape(id)}"], [data-series-label="${CSS.escape(id)}"]`;
  for (const el of svg.querySelectorAll<SVGElement>(selector)) el.style.display = display;
}

export function seriesOf(svg: SVGSVGElement): { id: string; label: string }[] {
  const seen = new Map<string, string>();
  for (const g of svg.querySelectorAll<SVGGElement>("g[data-series]")) {
    const id = g.dataset.series ?? "";
    if (!seen.has(id)) seen.set(id, g.dataset.label ?? id);
  }
  return [...seen].map(([id, label]) => ({ id, label }));
}

function toSvgPoint(svg: SVGSVGElement, clientX: number, clientY: number): DOMPoint {
  const pt = svg.createSVGPoint();
  pt.x = clientX;
  pt.y = clientY;
  const ctm = svg.getScreenCTM();
  return ctm === null ? pt : pt.matrixTransform(ctm.inverse());
}

function parsePoints(g: SVGGElement): Point[] {
  const cached = pointCache.get(g);
  if (cached !== undefined) return cached;
  const points = JSON.parse(g.dataset.points ?? "[]") as Point[];
  pointCache.set(g, points);
  return points;
}
const pointCache = new WeakMap<SVGGElement, Point[]>();

export function enhanceFigure(svg: SVGSVGElement, tooltip: Tooltip): void {
  // Marks with their own text.
  for (const el of svg.querySelectorAll<SVGElement>("[data-tip]")) {
    el.setAttribute("tabindex", "0");
    // A focusable mark needs a role and a name. The name is the same text the tooltip shows.
    el.setAttribute("role", "img");
    el.setAttribute("aria-label", el.dataset.tip ?? "");
    const show = (e: Event) => {
      const text = el.dataset.tip ?? "";
      if (e instanceof PointerEvent) tooltip.show(text, e.clientX, e.clientY);
      else {
        const r = el.getBoundingClientRect();
        tooltip.show(text, r.right, r.top);
      }
    };
    el.addEventListener("pointermove", show);
    el.addEventListener("focus", show);
    el.addEventListener("pointerleave", () => tooltip.hide());
    el.addEventListener("blur", () => tooltip.hide());
  }

  // Line plots: a crosshair across the series of one panel.
  const crosshair = document.createElementNS("http://www.w3.org/2000/svg", "line");
  crosshair.setAttribute("class", "f-ref");
  crosshair.style.display = "none";
  crosshair.style.pointerEvents = "none";
  svg.appendChild(crosshair);

  for (const hit of svg.querySelectorAll<SVGRectElement>("rect[data-plot]")) {
    const [bx, by, bw, bh] = (hit.dataset.box ?? "0,0,0,0").split(",").map(Number);
    const xlim = (hit.dataset.xlim ?? "0,1").split(",").map(Number) as [number, number];
    const panel = hit.dataset.panel ?? "0";
    hit.setAttribute("tabindex", "0");
    hit.setAttribute("role", "group"); // aria-label is only allowed on elements that have a role
    hit.setAttribute("aria-label", "Plot. Use the left and right arrow keys to read values.");
    let keyX: number | null = null;

    const update = (xValue: number, clientX: number, clientY: number) => {
      const rows: { label: string; value: number }[] = [];
      let xShown = xValue;
      const groups = svg.querySelectorAll<SVGGElement>(`g[data-series][data-panel="${panel}"]`);
      for (const g of groups) {
        if (g.style.display === "none") continue;
        const points = parsePoints(g);
        const i = nearestIndex(points, xValue);
        if (i < 0) continue;
        xShown = points[i][0];
        rows.push({ label: g.dataset.label ?? "", value: points[i][1] });
      }
      if (rows.length === 0) return;
      const xLabel = groups[0]?.dataset.xlabel ?? "x";
      const sx = bx + ((xShown - xlim[0]) / (xlim[1] - xlim[0])) * bw;
      crosshair.setAttribute("x1", String(sx));
      crosshair.setAttribute("x2", String(sx));
      crosshair.setAttribute("y1", String(by));
      crosshair.setAttribute("y2", String(by + bh));
      crosshair.style.display = "";
      tooltip.show(crosshairText(xLabel, xShown, rows), clientX, clientY);
    };

    hit.addEventListener("pointermove", (e) => {
      const p = toSvgPoint(svg, e.clientX, e.clientY);
      keyX = xFromPointer(p.x, { x: bx, w: bw }, xlim);
      update(keyX, e.clientX, e.clientY);
    });
    hit.addEventListener("keydown", (e) => {
      if (e.key !== "ArrowLeft" && e.key !== "ArrowRight") return;
      e.preventDefault();
      const step = (xlim[1] - xlim[0]) / 40;
      keyX = Math.min(xlim[1], Math.max(xlim[0], (keyX ?? xlim[0]) + (e.key === "ArrowRight" ? step : -step)));
      const r = hit.getBoundingClientRect();
      update(keyX, r.left + ((keyX - xlim[0]) / (xlim[1] - xlim[0])) * r.width, r.top + r.height / 2);
    });
    const leave = () => {
      crosshair.style.display = "none";
      tooltip.hide();
    };
    hit.addEventListener("pointerleave", leave);
    hit.addEventListener("blur", leave);
  }
}
