/** Tiny helper for building SVG by hand. Charts and the network drawing use it. */
const NS = "http://www.w3.org/2000/svg";

export type Attrs = Record<string, string | number>;

export function svg<K extends keyof SVGElementTagNameMap>(
  tag: K,
  attrs: Attrs = {},
  parent?: Element,
  text?: string,
): SVGElementTagNameMap[K] {
  const node = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, String(v));
  if (text !== undefined) node.textContent = text;
  parent?.appendChild(node);
  return node;
}

/** Round up to 1, 2 or 5 times a power of ten, for axis limits. */
export function niceCeil(value: number): number {
  if (value <= 0) return 1;
  const exp = Math.floor(Math.log10(value));
  const base = Math.pow(10, exp);
  const m = value / base;
  const nice = m <= 1 ? 1 : m <= 2 ? 2 : m <= 5 ? 5 : 10;
  return nice * base;
}

export function formatInt(n: number): string {
  return Math.round(n).toLocaleString("en-US");
}
