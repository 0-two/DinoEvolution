/**
 * The pure parts of the figure interactions: finding the point nearest the pointer and
 * formatting numbers for the tooltip. No DOM here, so Vitest can test it directly.
 */

export type Point = [number, number];

/** Index of the point whose x is closest to `x`. Points are sorted by x. */
export function nearestIndex(points: readonly Point[], x: number): number {
  let lo = 0;
  let hi = points.length - 1;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (points[mid][0] <= x) lo = mid;
    else hi = mid;
  }
  if (points.length === 0) return -1;
  return Math.abs(points[lo][0] - x) <= Math.abs(points[hi][0] - x) ? lo : hi;
}

/** Convert a pointer position inside a plot box to a value on the x axis. */
export function xFromPointer(px: number, box: { x: number; w: number }, xlim: [number, number]): number {
  const t = Math.min(1, Math.max(0, (px - box.x) / box.w));
  return xlim[0] + t * (xlim[1] - xlim[0]);
}

/** Whole numbers with thousands separators, small numbers with up to two decimals. */
export function formatValue(v: number): string {
  const abs = Math.abs(v);
  const digits = abs >= 100 ? 0 : abs >= 10 ? 1 : 2;
  return v.toLocaleString("en-US", { maximumFractionDigits: digits });
}

export interface SeriesPoint {
  label: string;
  value: number;
}

/** Tooltip text for a crosshair: the x position, then one row per visible series. */
export function crosshairText(xLabel: string, x: number, rows: readonly SeriesPoint[]): string {
  const width = Math.max(0, ...rows.map((r) => r.label.length));
  const lines = rows.map((r) => `${r.label.padEnd(width)}  ${formatValue(r.value)}`);
  return [`${xLabel} ${formatValue(x)}`, ...lines].join("\n");
}
