"""A tiny SVG writer for the report figures.

Why not matplotlib: the design system asks for direct labels at line ends, no legend, no chart
junk, thin rules and a fixed palette. That is easier to get exactly right by writing the few
shapes needed than by undoing a plotting library's defaults. It also means the results page can
inline these same files and add hover and toggle behaviour without a second chart renderer.

Styling is by CSS class, with colours taken from variables set on the root ``svg.fig`` element.
A standalone file therefore looks right on its own (light palette, paper background), and the
web page overrides the variables to follow its light and dark themes.

All text is monospace, so label widths are predictable: ``CHAR_W`` pixels per character at 11 px.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence

PAPER, INK, INK2, RULE, ACCENT = "#F3F1EC", "#161616", "#6B6862", "#D9D5CC", "#D9480F"
CHAR_W = 6.6  # IBM Plex Mono advance at 11 px, rounded up a little

STYLE = f"""
svg.fig {{ --f-paper: {PAPER}; --f-ink: {INK}; --f-ink2: {INK2}; --f-rule: {RULE}; --f-accent: {ACCENT}; }}
svg.fig .f-bg {{ fill: var(--f-paper); }}
svg.fig text {{ font-family: "IBM Plex Mono", ui-monospace, Consolas, monospace; font-size: 11px; fill: var(--f-ink2); }}
svg.fig .f-title {{ font-size: 13px; fill: var(--f-ink); }}
svg.fig .f-label {{ letter-spacing: 0.08em; text-transform: uppercase; }}
svg.fig .f-end {{ font-size: 12px; fill: var(--f-ink); }}
svg.fig .f-note {{ fill: var(--f-ink2); }}
svg.fig .f-grid {{ stroke: var(--f-rule); stroke-width: 1; fill: none; }}
svg.fig .f-axis {{ stroke: var(--c, var(--f-ink2)); stroke-width: 1; fill: none; pointer-events: none; }}
svg.fig .f-line {{ fill: none; stroke: var(--c, var(--f-ink)); stroke-width: 1.5; stroke-linejoin: round; }}
svg.fig .f-band {{ fill: var(--c, var(--f-ink)); stroke: none; opacity: 0.16; }}
svg.fig .f-dot {{ fill: var(--c, var(--f-ink)); stroke: none; }}
svg.fig .f-ring {{ fill: none; stroke: var(--c, var(--f-ink)); stroke-width: 1.25; pointer-events: all; }}
svg.fig .f-ref {{ stroke: var(--f-ink2); stroke-width: 1; stroke-dasharray: 2 3; fill: none; }}
svg.fig .s-ga {{ --c: var(--f-accent); }}
svg.fig .s-cmaes, svg.fig .s-neat, svg.fig .s-ink {{ --c: var(--f-ink); }}
svg.fig .s-ppo, svg.fig .s-rule_fixed, svg.fig .s-rule_scaled, svg.fig .s-ink2 {{ --c: var(--f-ink2); }}
svg.fig .s-accent {{ --c: var(--f-accent); }}
svg.fig .f-line.s-neat {{ stroke-dasharray: 6 3; }}
svg.fig .f-line.dash {{ stroke-dasharray: 6 3; }}
svg.fig .f-line.s-rule_fixed {{ stroke-dasharray: 2 3; stroke-width: 1; }}
svg.fig .f-line.s-rule_scaled {{ stroke-dasharray: 8 3 2 3; stroke-width: 1; }}
svg.fig .f-box {{ fill: none; stroke: var(--f-ink2); stroke-width: 1; }}
svg.fig .f-arrow {{ fill: var(--f-ink2); stroke: none; }}
svg.fig .f-cell-on {{ fill: var(--f-ink); }}
svg.fig .f-cell-duck {{ fill: var(--f-accent); }}
svg.fig .f-cell-off {{ fill: var(--f-rule); opacity: 0.55; }}
svg.fig .f-dim {{ opacity: 0.12; }}
svg.fig [data-series] {{ cursor: pointer; }}
svg.fig .f-hit {{ fill: transparent; stroke: none; }}
"""

ALGO_LABEL = {
    "ga": "GA",
    "cmaes": "CMA-ES",
    "neat": "NEAT",
    "ppo": "PPO",
    "rule_fixed": "rule, fixed",
    "rule_scaled": "rule, scaled",
}


def esc(text: object) -> str:
    return html.escape(str(text), quote=True)


def fmt(v: float, decimals: int = 1) -> str:
    return f"{v:.{decimals}f}".rstrip("0").rstrip(".") if decimals else f"{v:.0f}"


def thousands(v: float) -> str:
    return f"{v:,.0f}"


def nice_ceil(value: float) -> float:
    if value <= 0:
        return 1.0
    exp = math.floor(math.log10(value))
    base = 10.0**exp
    m = value / base
    for step in (1, 2, 2.5, 5, 10):
        if m <= step:
            return step * base
    return 10 * base


def axis_top(value: float, target: int = 4) -> float:
    """Smallest multiple of a round tick step that covers ``value``. Tighter than ``nice_ceil``,
    which jumps from 12,000 to 20,000 and leaves a third of the plot empty."""
    ticks = nice_ticks(0, value, target)
    step = ticks[1] - ticks[0] if len(ticks) > 1 else max(value, 1.0)
    return math.ceil(value / step - 1e-9) * step


def nice_ticks(lo: float, hi: float, target: int = 5) -> list[float]:
    """Round tick values covering [lo, hi]."""
    span = hi - lo
    if span <= 0:
        return [lo]
    raw = span / target
    exp = math.floor(math.log10(raw))
    base = 10.0**exp
    step = next(s * base for s in (1, 2, 2.5, 5, 10) if s * base >= raw)
    start = math.ceil(lo / step - 1e-9) * step
    out, v = [], start
    while v <= hi + 1e-9:
        out.append(round(v, 10))
        v += step
    return out


def spread_labels(ys: Sequence[float], min_gap: float, lo: float, hi: float) -> list[float]:
    """Move label positions apart so that neighbours are at least ``min_gap`` px, in order."""
    order = sorted(range(len(ys)), key=lambda i: ys[i])
    pos = [ys[i] for i in order]
    for k in range(1, len(pos)):
        pos[k] = max(pos[k], pos[k - 1] + min_gap)
    overflow = pos[-1] - hi if pos else 0
    if overflow > 0:
        pos = [p - overflow for p in pos]
        for k in range(len(pos) - 2, -1, -1):
            pos[k] = min(pos[k], pos[k + 1] - min_gap)
    pos = [max(p, lo) for p in pos]
    out = [0.0] * len(ys)
    for k, i in enumerate(order):
        out[i] = pos[k]
    return out


class Svg:
    """Accumulates elements and renders a standalone SVG document."""

    def __init__(self, width: float, height: float, title: str, description: str) -> None:
        self.width, self.height = width, height
        self.title, self.description = title, description
        self.parts: list[str] = []

    def add(self, markup: str) -> None:
        self.parts.append(markup)

    def line(self, x1: float, y1: float, x2: float, y2: float, cls: str, extra: str = "") -> None:
        self.add(
            f'<line x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}" class="{cls}" {extra}/>'
        )

    def rect(self, x: float, y: float, w: float, h: float, cls: str, extra: str = "") -> None:
        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{w:.2f}" height="{h:.2f}" class="{cls}" {extra}/>'
        )

    def circle(self, cx: float, cy: float, r: float, cls: str, extra: str = "") -> None:
        self.add(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r}" class="{cls}" {extra}/>')

    def path(self, points: Sequence[tuple[float, float]], cls: str, extra: str = "") -> None:
        if len(points) < 2:
            return
        d = "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in points)
        self.add(f'<path d="{d}" class="{cls}" {extra}/>')

    def polygon(self, points: Sequence[tuple[float, float]], cls: str, extra: str = "") -> None:
        d = "M" + " L".join(f"{x:.2f} {y:.2f}" for x, y in points) + " Z"
        self.add(f'<path d="{d}" class="{cls}" {extra}/>')

    def text(
        self,
        x: float,
        y: float,
        s: object,
        cls: str = "",
        anchor: str = "start",
        extra: str = "",
    ) -> None:
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" text-anchor="{anchor}" class="{cls}" {extra}>{esc(s)}</text>'
        )

    def render(self) -> str:
        head = (
            f'<svg xmlns="http://www.w3.org/2000/svg" class="fig" viewBox="0 0 {self.width:g} '
            f'{self.height:g}" width="{self.width:g}" height="{self.height:g}" role="img" '
            f'aria-labelledby="t d">'
            f'<title id="t">{esc(self.title)}</title><desc id="d">{esc(self.description)}</desc>'
            f"<style>{STYLE}</style>"
            f'<rect class="f-bg" width="{self.width:g}" height="{self.height:g}"/>'
        )
        return head + "".join(self.parts) + "</svg>\n"


class Axes:
    """A rectangular plotting area with linear scales, light grid lines and axis labels."""

    def __init__(
        self,
        svg: Svg,
        x: float,
        y: float,
        w: float,
        h: float,
        xlim: tuple[float, float],
        ylim: tuple[float, float],
    ) -> None:
        self.svg, self.x, self.y, self.w, self.h = svg, x, y, w, h
        self.xlim, self.ylim = xlim, ylim

    def sx(self, v: float) -> float:
        lo, hi = self.xlim
        return self.x + self.w * (v - lo) / (hi - lo)

    def sy(self, v: float) -> float:
        lo, hi = self.ylim
        return self.y + self.h * (1 - (v - lo) / (hi - lo))

    def frame(
        self,
        xticks: Sequence[float],
        yticks: Sequence[float],
        xlabel: str,
        ylabel: str,
        xfmt=thousands,
        yfmt=thousands,
        grid: bool = True,
        xtick_labels: Sequence[str] | None = None,
    ) -> None:
        s = self.svg
        for v in yticks:
            if grid:
                s.line(self.x, self.sy(v), self.x + self.w, self.sy(v), "f-grid")
            s.text(self.x - 8, self.sy(v) + 4, yfmt(v), "", "end")
        s.line(self.x, self.sy(self.ylim[0]), self.x + self.w, self.sy(self.ylim[0]), "f-axis")
        for i, v in enumerate(xticks):
            label = xtick_labels[i] if xtick_labels else xfmt(v)
            s.line(
                self.sx(v), self.sy(self.ylim[0]), self.sx(v), self.sy(self.ylim[0]) + 4, "f-axis"
            )
            s.text(self.sx(v), self.y + self.h + 17, label, "", "middle")
        s.text(self.x + self.w, self.y + self.h + 34, xlabel, "f-label", "end")
        s.text(self.x - 52, self.y - 6, ylabel, "f-label", "start")
