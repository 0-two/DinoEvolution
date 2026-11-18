/**
 * Draws the followed dino's network as nodes and lines.
 *
 * Line width is weight size. A solid line is a positive weight, a dashed line a negative one.
 * A node fills in as it activates: the inner disc grows with the size of the activation and
 * is dark for positive values and grey for negative ones. The output the network picked is
 * filled in the accent colour.
 *
 * The lines depend only on the genome, so they are rebuilt when the followed dino changes.
 * The nodes change every tick and are updated in place.
 */
import {
  HIDDEN,
  INPUTS,
  INPUT_LABELS,
  OUTPUTS,
  OUTPUT_LABELS,
  w1,
  w2,
  type Activations,
} from "../evolve/network";
import { svg } from "./svg";

const WIDTH = 360;
const HEIGHT = 270;
const X_IN = 112;
const X_HID = 190;
const X_OUT = 268;
const TOP = 18;
const BOTTOM = HEIGHT - 18;
const R = 7;
const MAX_LINE = 3.2;

function columnY(count: number): number[] {
  if (count === 1) return [(TOP + BOTTOM) / 2];
  return Array.from({ length: count }, (_, i) => TOP + ((BOTTOM - TOP) * i) / (count - 1));
}

export class NetworkGraph {
  readonly root: SVGSVGElement;
  private readonly lines: SVGGElement;
  private readonly inFill: SVGCircleElement[] = [];
  private readonly hidFill: SVGCircleElement[] = [];
  private readonly outFill: SVGCircleElement[] = [];
  private readonly inY = columnY(INPUTS);
  private readonly hidY = columnY(HIDDEN);
  private readonly outY = columnY(OUTPUTS);
  private genome: Float64Array | null = null;

  constructor(container: HTMLElement) {
    this.root = svg("svg", {
      viewBox: `0 0 ${WIDTH} ${HEIGHT}`,
      role: "img",
      "aria-label":
        "Network of the followed dino: nine inputs, eight hidden units, three outputs. " +
        "Line width shows weight size.",
      class: "network-svg",
    });
    this.lines = svg("g", { class: "net-lines" }, this.root);
    this.buildNodes();
    container.appendChild(this.root);
  }

  /** Redraw the lines for a new genome. Does nothing if it is the one already shown. */
  setGenome(genome: Float64Array): void {
    if (genome === this.genome) return;
    this.genome = genome;
    this.lines.replaceChildren();
    for (let h = 0; h < HIDDEN; h++) {
      for (let i = 0; i < INPUTS; i++) {
        this.line(X_IN, this.inY[i], X_HID, this.hidY[h], w1(genome, h, i));
      }
    }
    for (let o = 0; o < OUTPUTS; o++) {
      for (let h = 0; h < HIDDEN; h++) {
        this.line(X_HID, this.hidY[h], X_OUT, this.outY[o], w2(genome, o, h));
      }
    }
  }

  /** Fill the nodes from the activations of the latest forward pass. */
  update(act: Activations, action: number): void {
    for (let i = 0; i < INPUTS; i++) this.fill(this.inFill[i], act.inputs[i], false);
    for (let h = 0; h < HIDDEN; h++) this.fill(this.hidFill[h], act.hidden[h], false);
    for (let o = 0; o < OUTPUTS; o++) this.fill(this.outFill[o], o === action ? 1 : 0, true);
  }

  private line(x1: number, y1: number, x2: number, y2: number, weight: number): void {
    const width = Math.min(MAX_LINE, 0.3 + 1.1 * Math.abs(weight));
    const attrs: Record<string, string | number> = { x1, y1, x2, y2, "stroke-width": width.toFixed(2) };
    if (weight < 0) attrs["stroke-dasharray"] = `${(2 + width).toFixed(1)} ${(1.5 + width).toFixed(1)}`;
    svg("line", attrs, this.lines);
  }

  private fill(node: SVGCircleElement, value: number, chosen: boolean): void {
    const mag = Math.min(1, Math.abs(value));
    const r = (R - 1) * mag;
    if (node.dataset.r !== r.toFixed(2)) {
      node.dataset.r = r.toFixed(2);
      node.setAttribute("r", r.toFixed(2));
    }
    const cls = chosen ? "fill-accent" : value < 0 ? "fill-neg" : "fill-pos";
    if (node.getAttribute("class") !== cls) node.setAttribute("class", cls);
  }

  private buildNodes(): void {
    const label = (x: number, y: number, text: string, anchor: "start" | "end") =>
      svg("text", { x, y: y + 4, "text-anchor": anchor, class: "net-label" }, this.root, text);
    const node = (x: number, y: number, store: SVGCircleElement[]) => {
      svg("circle", { cx: x, cy: y, r: R, class: "node-ring" }, this.root);
      store.push(svg("circle", { cx: x, cy: y, r: 0, class: "fill-pos" }, this.root));
    };
    for (let i = 0; i < INPUTS; i++) {
      node(X_IN, this.inY[i], this.inFill);
      label(X_IN - R - 8, this.inY[i], INPUT_LABELS[i], "end");
    }
    for (let h = 0; h < HIDDEN; h++) node(X_HID, this.hidY[h], this.hidFill);
    for (let o = 0; o < OUTPUTS; o++) {
      node(X_OUT, this.outY[o], this.outFill);
      label(X_OUT + R + 8, this.outY[o], OUTPUT_LABELS[o], "start");
    }
  }
}
