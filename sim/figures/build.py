"""Build every figure and table of the report from ``results/``.

    uv run python -m sim.figures.build            # reads results/main and results/ablations
    uv run python -m sim.figures.build --suite quick --out-dir results/quick/report

Nothing here contains a result typed in by hand. Figures are SVG in ``report/figures``, tables
are Markdown and JSON in ``report/tables``. A figure whose data is missing is skipped with a
message, so the script can run on a partial results folder.

Style rules, from the design brief: thin lines, direct labels at line ends instead of legends,
light gridlines, axes with units, confidence intervals as pale bands in the line's own colour,
one accent colour (used for the GA, the algorithm in the live demo).
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from sim.analysis import curves, probe
from sim.experiments.aggregate import load_runs, steps_to_threshold, summarise
from sim.experiments.runner import RESULTS_ROOT, finite
from sim.figures.svgkit import (
    ALGO_LABEL,
    Axes,
    Svg,
    axis_top,
    esc,
    nice_ceil,
    nice_ticks,
    spread_labels,
    thousands,
)
from sim.stats import mean_ci95

REPO = RESULTS_ROOT.parent
NEWLINE = chr(10)
LEARNED = ["ga", "cmaes", "neat", "ppo"]
WIDTH = 720
MARKERS = {"ga": "circle", "cmaes": "square", "neat": "diamond", "ppo": "triangle",
           "rule_fixed": "ring", "rule_scaled": "ring"}  # fmt: skip


def label(run_id: str) -> str:
    return ALGO_LABEL.get(run_id, run_id)


def group(runs: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in runs:
        out[r["run_id"]].append(r)
    return dict(out)


def marker(
    svg: Svg, x: float, y: float, shape: str, cls: str, tip: str = "", r: float = 4.0
) -> None:
    extra = f'data-tip="{esc(tip)}"' if tip else ""
    if shape == "square":
        svg.rect(x - r, y - r, 2 * r, 2 * r, f"f-dot {cls}", extra)
    elif shape == "diamond":
        svg.polygon(
            [(x, y - r * 1.3), (x + r * 1.3, y), (x, y + r * 1.3), (x - r * 1.3, y)],
            f"f-dot {cls}",
            extra,
        )
    elif shape == "triangle":
        svg.polygon(
            [(x, y - r * 1.2), (x + r * 1.15, y + r * 0.9), (x - r * 1.15, y + r * 0.9)],
            f"f-dot {cls}",
            extra,
        )
    elif shape == "ring":
        svg.circle(x, y, r, f"f-ring {cls}", extra)
    else:
        svg.circle(x, y, r, f"f-dot {cls}", extra)


def points_attr(xs: list[float], ys: list[float]) -> str:
    """Compact data for the page's hover crosshair."""
    pairs = [[round(x, 4), round(y, 2)] for x, y in zip(xs, ys, strict=True)]
    return f"data-points='{json.dumps(pairs, separators=(',', ':'))}'"


# --- 1. learning curves ----------------------------------------------------------------------
def learning_curves(main: list[dict[str, Any]]) -> str | None:
    g = group(main)
    algos = [a for a in LEARNED if a in g]
    if not algos:
        return None
    panels = [
        ("best", "Best fitness in each generation"),
        ("mean", "Mean fitness of each generation"),
    ]
    stats = {(a, k): curves.curve_stats(g[a], k) for a in algos for k, _ in panels}
    x_max = max(s["steps_m"][-1] for s in stats.values())
    ceiling = max(r["eval"]["max_ticks"] for a in algos for r in g[a])
    ceiling_score = _ceiling(ceiling)

    left, right, top, ph, gap = 64, 120, 40, 190, 70
    height = top + 2 * ph + gap + 66
    svg = Svg(WIDTH, height, "Learning curves",
              "Best and mean training fitness against environment steps for GA, CMA-ES, NEAT and PPO, "
              "mean and 95% interval over seeds.")  # fmt: skip
    pw = WIDTH - left - right
    n_runs = {a: len(g[a]) for a in algos}
    counts = sorted(set(n_runs.values()))
    seed_text = str(counts[0]) if len(counts) == 1 else f"{counts[0]} to {counts[-1]}"
    for p, (key, title) in enumerate(panels):
        y0 = top + p * (ph + gap)
        y_max = nice_ceil(max(max(stats[(a, key)]["hi"]) for a in algos) * 1.05)
        y_max = max(y_max, ceiling_score * 1.0 if key == "best" else 0)
        ax = Axes(svg, left, y0, pw, ph, (0, x_max), (0, y_max))
        svg.text(left, y0 - 22, title, "f-title")
        ax.frame(nice_ticks(0, x_max, 5), nice_ticks(0, y_max, 4), "environment steps (millions)",
                 "fitness (score)", xfmt=lambda v: f"{v:g}")  # fmt: skip
        if key == "best":
            yy = ax.sy(ceiling_score)
            svg.line(left, yy, left + pw, yy, "f-ref")
            svg.text(
                left + pw,
                yy - 5,
                f"ceiling {thousands(ceiling_score)} ({ceiling:,} ticks)",
                "f-note",
                "end",
            )
        ends = []
        for a in algos:
            s = stats[(a, key)]
            xs = [ax.sx(v) for v in s["steps_m"]]
            band = [(x, ax.sy(v)) for x, v in zip(xs, s["hi"], strict=True)]
            band += [(x, ax.sy(v)) for x, v in zip(reversed(xs), reversed(s["lo"]), strict=True)]
            svg.add(f'<g data-series="{a}" data-panel="{p}" data-label="{label(a)}" '
                    f'data-xlabel="M steps" {points_attr(s["steps_m"], s["mean"])}>')  # fmt: skip
            svg.polygon(band, f"f-band s-{a}")
            svg.path(list(zip(xs, [ax.sy(v) for v in s["mean"]], strict=True)), f"f-line s-{a}")
            svg.add("</g>")
            ends.append((a, xs[-1], ax.sy(s["mean"][-1]), s["mean"][-1]))
        ys = spread_labels([e[2] + 4 for e in ends], 15, y0 + 8, y0 + ph)
        for (a, x, _, v), ly in zip(ends, ys, strict=True):
            svg.text(
                x + 8, ly, f"{label(a)} {thousands(v)}", "f-end", extra=f'data-series-label="{a}"'
            )
        svg.add(f'<rect class="f-hit" x="{left}" y="{y0}" width="{pw}" height="{ph}" data-plot="1" '
                f'data-panel="{p}" data-box="{left},{y0},{pw},{ph}" data-xlim="0,{x_max:.4f}" data-ylim="0,{y_max}"/>')  # fmt: skip
    svg.text(left, height - 8,
             f"Mean and 95% interval over {seed_text} seeds. "
             "Training episodes, three seeds per genome.", "f-note")  # fmt: skip
    return svg.render()


def _ceiling(max_ticks: int) -> float:
    from sim.game import DinoGame

    g = DinoGame(0, max_ticks, collisions=False)
    while not g.done:
        g.step(0)
    return g.score


# --- 2. final scores on unseen seeds ---------------------------------------------------------
def final_scores(main: list[dict[str, Any]]) -> str | None:
    g = group(main)
    if not g:
        return None
    rows = summarise(main)
    order = [r["run_id"] for r in sorted(rows, key=lambda r: r["test_mean"])]
    left, right, top, ph, bottom = 64, 24, 44, 270, 56
    height = top + ph + bottom
    svg = Svg(WIDTH, height, "Final scores on unseen seeds",
              "Mean test score of each run's best genome on 100 unseen seeds, by algorithm.")  # fmt: skip
    all_means = [r["test"]["mean"] for rs in g.values() for r in rs]
    cap = _ceiling(main[0]["test"]["max_ticks"])
    y_max = axis_top(max(max(all_means), 1) * 1.05, 5)
    ax = Axes(svg, left, top, WIDTH - left - right, ph, (0, len(order)), (0, y_max))
    svg.text(left, top - 24, "Test score of each run's best genome, 100 unseen seeds", "f-title")
    ax.frame([], nice_ticks(0, y_max, 5), "", "score (full 36,000-tick game)")
    if cap <= y_max:
        svg.line(left, ax.sy(cap), left + ax.w, ax.sy(cap), "f-ref")
        svg.text(left + ax.w, ax.sy(cap) - 5, f"ceiling {thousands(cap)}", "f-note", "end")
    for i, rid in enumerate(order):
        cx = ax.sx(i + 0.5)
        algo = g[rid][0]["algorithm"]
        means = [r["test"]["mean"] for r in g[rid]]
        m, half = mean_ci95(means)
        for r in sorted(g[rid], key=lambda r: r["seed"]):
            jitter = ((r["seed"] % 10) - 4.5) * 4.2
            marker(svg, cx + jitter, ax.sy(r["test"]["mean"]), MARKERS[algo], f"s-{algo}",
                   f"{label(rid)} seed {r['seed']}: {thousands(r['test']['mean'])}", 3.4)  # fmt: skip
        svg.line(cx - 26, ax.sy(m), cx + 26, ax.sy(m), f"f-axis s-{algo}", 'style="stroke-width:2"')
        if not math.isnan(half):
            svg.line(cx, ax.sy(m - half), cx, ax.sy(m + half), f"f-axis s-{algo}")
        svg.text(cx, top + ph + 18, label(rid), "f-end", "middle")
        svg.text(cx, top + ph + 34, f"mean {thousands(m)}", "", "middle")
    svg.text(
        left,
        height - 6,
        "One dot per run. Bar: mean over runs. Whisker: 95% interval of the mean.",
        "f-note",
    )
    return svg.render()


# --- 3. sample efficiency --------------------------------------------------------------------
def sample_efficiency(main: list[dict[str, Any]]) -> str | None:
    g = group(main)
    algos = [a for a in LEARNED if a in g]
    if not algos:
        return None
    thresholds = [1000.0, 3000.0]
    left, right, top, row_h, panel_gap = 96, 24, 44, 38, 80
    ph = row_h * len(algos)
    height = top + len(thresholds) * (ph + panel_gap) + 8
    svg = Svg(WIDTH, height, "Environment steps to reach a score",
              "Steps until the best genome of a generation reached 1,000 and 3,000 points, per run.")  # fmt: skip
    all_steps = [r["env_steps"] for a in algos for r in g[a]]
    x_max = nice_ceil(max(all_steps) / 1e6 * 1.02)
    pw = WIDTH - left - right - 90
    for p, thr in enumerate(thresholds):
        y0 = top + p * (ph + panel_gap)
        svg.text(
            left,
            y0 - 22,
            f"Steps until a generation's best fitness reaches {thousands(thr)}",
            "f-title",
        )
        ax = Axes(svg, left, y0, pw, ph, (0, x_max), (0, len(algos)))
        for v in nice_ticks(0, x_max, 5):
            svg.line(ax.sx(v), y0, ax.sx(v), y0 + ph, "f-grid")
            svg.text(ax.sx(v), y0 + ph + 17, f"{v:g}", "", "middle")
        svg.text(left + pw, y0 + ph + 34, "environment steps (millions)", "f-label", "end")
        svg.text(left + pw + 22, y0 - 4, "reached", "f-label")
        for i, a in enumerate(algos):
            cy = y0 + (i + 0.5) * row_h
            svg.text(left - 10, cy + 4, label(a), "f-end", "end")
            hits = []
            for r in sorted(g[a], key=lambda r: r["seed"]):
                s = steps_to_threshold(r, thr)
                if s is not None:
                    hits.append(s)
                    jitter = ((r["seed"] % 5) - 2) * 4.5
                    marker(svg, ax.sx(s / 1e6), cy + jitter, MARKERS[a], f"s-{a}",
                           f"{label(a)} seed {r['seed']}: {s / 1e6:.2f}M steps", 3.4)  # fmt: skip
            if hits:
                med = statistics.median(hits) / 1e6
                svg.line(
                    ax.sx(med),
                    cy - 13,
                    ax.sx(med),
                    cy + 13,
                    f"f-axis s-{a}",
                    'style="stroke-width:2"',
                )
            svg.text(left + pw + 22, cy + 4, f"{len(hits)} of {len(g[a])}", "f-end")
    svg.text(
        left, height - 6, "One dot per run that got there. Bar: median of those runs.", "f-note"
    )
    return svg.render()


# --- 4. generalisation gap -------------------------------------------------------------------
def generalisation(rows: list[dict[str, Any]]) -> str | None:
    rows = [r for r in rows if r["algorithm"] in LEARNED]
    if not rows:
        return None
    left, top, size = 64, 44, 330
    height = top + size + 64
    svg = Svg(WIDTH, height, "Training score against test score",
              "For each run, the best genome's training fitness against its mean score on 100 unseen "
              "seeds, both at the 12,000-tick training cap.")  # fmt: skip
    lim = nice_ceil(
        max(max(r["train_fitness"] for r in rows), max(r["test_at_train_cap"] for r in rows)) * 1.04
    )
    ax = Axes(svg, left, top, size, size, (0, lim), (0, lim))
    svg.text(left, top - 24, "Best training fitness against score on unseen seeds", "f-title")
    ax.frame(
        nice_ticks(0, lim, 5),
        nice_ticks(0, lim, 5),
        "training fitness (3 seeds)",
        "test score (100 unseen seeds)",
    )
    svg.line(ax.sx(0), ax.sy(0), ax.sx(lim), ax.sy(lim), "f-ref")
    svg.text(ax.sx(lim) - 4, ax.sy(lim) + 14, "no gap", "f-note", "end")
    ends = []
    for a in LEARNED:
        pts = [r for r in rows if r["algorithm"] == a]
        if not pts:
            continue
        for r in pts:
            marker(svg, ax.sx(r["train_fitness"]), ax.sy(r["test_at_train_cap"]), MARKERS[a], f"s-{a}",
                   f"{label(a)} seed {r['seed']}: train {r['train_fitness']:.0f}, test {r['test_at_train_cap']:.0f}", 3.6)  # fmt: skip
        mx = statistics.fmean(r["train_fitness"] for r in pts)
        my = statistics.fmean(r["test_at_train_cap"] for r in pts)
        ends.append((a, my, mx))
    ys = spread_labels([ax.sy(my) + 4 for _, my, _ in ends], 32, top + 10, top + size)
    for (a, my, mx), ly in zip(ends, ys, strict=True):
        svg.text(left + size + 14, ly, label(a), "f-end")
        svg.text(
            left + size + 14,
            ly + 13,
            f"gap {mx - my:+,.0f} on average".replace("+", "").replace("-", "-"),
            "f-note",
        )
    svg.text(
        left,
        height - 6,
        "One mark per run. Distance below the dotted line is what the best genome lost on unseen seeds.",
        "f-note",
    )
    return svg.render()


# --- 5. ablations ----------------------------------------------------------------------------
ABLATION_PANELS = [
    ("Population size", [("ga-pop50", "50"), ("ga-default", "150"), ("ga-pop300", "300")]),
    ("Mutation size", [("ga-sigma0_1", "0.1"), ("ga-default", "0.3"), ("ga-sigma0_6", "0.6")]),
    ("Elitism", [("ga-noelite", "off"), ("ga-default", "on")]),
    ("Seeds per fitness, K", [("ga-k1", "1"), ("ga-default", "3"), ("ga-k5", "5")]),
]


def ablations(abl: list[dict[str, Any]]) -> str | None:
    g = group(abl)
    if "ga-default" not in g:
        return None
    pw, ph, top, gx, gy = 240, 150, 56, 120, 96
    left = 64
    height = top + 2 * ph + gy + 40
    svg = Svg(WIDTH, height, "GA ablations",
              "Test score of the GA's best genome under changes to population size, mutation size, "
              "elitism and the number of evaluation seeds.")  # fmt: skip
    all_means = [r["test"]["mean"] for rs in g.values() for r in rs]
    y_max = axis_top(max(all_means) * 1.05, 3)
    for i, (title, items) in enumerate(ABLATION_PANELS):
        col, row = i % 2, i // 2
        x0, y0 = left + col * (pw + gx - 60 + 60), top + row * (ph + gy)
        ax = Axes(svg, x0, y0, pw, ph, (0, len(items)), (0, y_max))
        svg.text(x0, y0 - 24, title, "f-title")
        ax.frame([], nice_ticks(0, y_max, 3), "", "score" if col == 0 else "")
        for j, (rid, name) in enumerate(items):
            cx = ax.sx(j + 0.5)
            runs = g.get(rid, [])
            cls = "s-ga" if rid == "ga-default" else "s-ink"
            if not runs:
                continue
            means = [r["test"]["mean"] for r in runs]
            m, half = mean_ci95(means)
            for r in sorted(runs, key=lambda r: r["seed"]):
                jitter = ((r["seed"] % 10) - 4.5) * 3.6
                marker(svg, cx + jitter, ax.sy(r["test"]["mean"]), "circle", cls,
                       f"{rid} seed {r['seed']}: {thousands(r['test']['mean'])}", 2.8)  # fmt: skip
            svg.line(
                cx - 22, ax.sy(m), cx + 22, ax.sy(m), f"f-axis {cls}", 'style="stroke-width:2"'
            )
            if not math.isnan(half):
                svg.line(cx, ax.sy(max(0, m - half)), cx, ax.sy(m + half), f"f-axis {cls}")
            svg.text(cx, y0 + ph + 17, name, "f-end", "middle")
    svg.text(
        left,
        height - 6,
        "One dot per run, 10 runs per setting. Bar: mean. Whisker: 95% interval. Accent: default settings.",
        "f-note",
    )
    return svg.render()


# --- 6. decision maps ------------------------------------------------------------------------
def decision_maps(main: list[dict[str, Any]], root: Path) -> str | None:
    g = group(main)
    cols = [a for a in [*LEARNED, "rule_scaled"] if a in g]
    if not cols:
        return None
    pw, cell_h = 104.0, 7.0
    cell_w = pw / len(probe.DISTANCES)
    ph = len(probe.SPEEDS) * cell_h
    left, top, hgap, vgap = 100, 100, 16, 60
    width = left + len(cols) * (pw + hgap)
    kinds = list(probe.OBSTACLES)
    height = top + len(kinds) * (ph + vgap) + 8
    svg = Svg(max(WIDTH, width), height, "Decision maps",
              "Action chosen by each algorithm's best genome for an obstacle at a given distance and "
              "game speed. Dark: jump. Accent: duck. Pale: nothing.")  # fmt: skip
    svg.text(
        16, 20, "What the best genome of each algorithm does as an obstacle approaches", "f-title"
    )
    svg.text(
        16,
        38,
        "Across: distance to the obstacle, 0 to 400 px. Down: game speed, 6 at the top to 13 at the bottom.",
        "f-note",
    )
    svg.text(16, 54, "Dark: jump. Accent: duck. Pale: nothing.", "f-note")
    for k, kind in enumerate(kinds):
        y0 = top + k * (ph + vgap)
        svg.text(left - 10, y0 + ph / 2 + 4, kind, "f-end", "end")
    for c, a in enumerate(cols):
        best = max(g[a], key=lambda r: r["test"]["mean"])
        payload = json.loads((REPO / best["genome_file"]).read_text(encoding="utf-8"))
        x0 = left + c * (pw + hgap)
        svg.text(x0, top - 20, label(a), "f-end")
        svg.text(
            x0, top - 6, f"seed {best['seed']}, test {thousands(best['test']['mean'])}", "f-note"
        )
        for k, kind in enumerate(kinds):
            y0 = top + k * (ph + vgap)
            grid = probe.decision_map(payload, kind)
            for r, row in enumerate(grid):
                # One rect per run of equal actions: the maps would otherwise be thousands of cells.
                start = 0
                for col in range(1, len(row) + 1):
                    if col == len(row) or row[col] != row[start]:
                        action = row[start]
                        cls = {0: "f-cell-off", 1: "f-cell-on", 2: "f-cell-duck"}[action]
                        d0, d1 = probe.DISTANCES[start], probe.DISTANCES[col - 1]
                        tip = (
                            f"{label(a)}, {kind}, speed {probe.SPEEDS[r]:g}: "
                            f"{probe.ACTION_WORDS[action]} at {d0} to {d1} px"
                        )
                        svg.rect(
                            x0 + start * cell_w,
                            y0 + r * cell_h,
                            (col - start) * cell_w + 0.35,
                            cell_h - 0.8,
                            cls,
                            f'data-tip="{esc(tip)}"',
                        )
                        start = col
            for d, anchor in ((0, "start"), (200, "middle"), (400, "end")):
                svg.text(
                    x0 + (d / 5) * cell_w + (0 if d == 0 else 2 if d == 400 else 0),
                    y0 + ph + 15,
                    str(d),
                    "",
                    anchor,
                )
    return svg.render()


# --- 7. NEAT growth --------------------------------------------------------------------------
def neat_growth(main: list[dict[str, Any]]) -> str | None:
    g = group(main)
    if "neat" not in g or "best_hidden_nodes" not in g["neat"][0]["history"][0]:
        return None
    keys = [
        ("best_hidden_nodes", "Hidden nodes"),
        ("best_connections", "Enabled connections"),
        ("species", "Species"),
    ]
    left, top, pw, ph, gap = 52, 56, 170, 170, 54
    height = top + ph + 90
    svg = Svg(WIDTH, height, "NEAT topology growth",
              "Hidden nodes and connections of the best genome, and the number of species, by generation.")  # fmt: skip
    stats = {k: curves.by_generation(g["neat"], k) for k, _ in keys}
    n_gen = len(stats[keys[0][0]]["generation"])
    for i, (k, title) in enumerate(keys):
        s = stats[k]
        x0 = left + i * (pw + gap)
        y_max = nice_ceil(max(max(s["hi"]), 1) * 1.1)
        ax = Axes(svg, x0, top, pw, ph, (1, n_gen), (0, y_max))
        svg.text(x0, top - 24, title, "f-title")
        ax.frame(
            nice_ticks(1, n_gen, 4),
            nice_ticks(0, y_max, 4),
            "generation",
            "",
            xfmt=lambda v: f"{v:g}",
            yfmt=lambda v: f"{v:g}",
        )
        xs = [ax.sx(v) for v in s["generation"]]
        lo = [max(0.0, v) for v in s["lo"]]  # counts cannot be negative
        band = [(x, ax.sy(v)) for x, v in zip(xs, s["hi"], strict=True)]
        band += [(x, ax.sy(v)) for x, v in zip(reversed(xs), reversed(lo), strict=True)]
        svg.add(
            f'<g data-series="neat" data-panel="{i}" data-label="{title}" data-xlabel="generation" '
            f"{points_attr(s['generation'], s['mean'])}>"
        )
        svg.polygon(band, "f-band s-neat")
        svg.path(list(zip(xs, [ax.sy(v) for v in s["mean"]], strict=True)), "f-line s-neat")
        svg.add("</g>")
        svg.text(xs[-1], ax.sy(s["mean"][-1]) - 8, f"{s['mean'][-1]:.1f}", "f-end", "end")
        svg.add(f'<rect class="f-hit" x="{x0}" y="{top}" width="{pw}" height="{ph}" data-plot="1" '
                f'data-panel="{i}" data-box="{x0},{top},{pw},{ph}" data-xlim="1,{n_gen}" data-ylim="0,{y_max}"/>')  # fmt: skip
    svg.text(
        left,
        height - 8,
        f"Mean and 95% interval over {len(g['neat'])} runs, up to the shortest run ({n_gen} generations).",
        "f-note",
    )
    return svg.render()


# --- 8. the evolutionary loop -----------------------------------------------------------------
def loop_diagram() -> str:
    """The generation loop of the GA with the parameters it actually uses. Drawn from
    ``ga.DEFAULTS`` so the numbers on the diagram cannot drift from the code."""
    from sim.algorithms import ga
    from sim.evaluation import EvalConfig

    p = ga.DEFAULTS
    k = EvalConfig().k_seeds
    steps = [
        ("1  Evaluate", [f"play {k} seeds,", "fitness = mean score"]),
        (
            "2  Select",
            [f"tournament of {p['tournament_size']},", f"{p['elite_count']} elites kept"],
        ),
        (
            "3  Cross over",
            ["each weight from", f"either parent, p {p['crossover_rate']:g}"],
        ),
        (
            "4  Mutate",
            [
                f"weights: p {p['mutation_rate']:g},",
                f"gaussian sd {p['mutation_sigma']:g}",
            ],
        ),
    ]
    width, height = WIDTH, 250
    svg = Svg(width, height, "The evolutionary loop",
              "Four steps repeated every generation: evaluate each genome on several seeds, select "
              "parents, cross them over, mutate the children.")  # fmt: skip
    svg.text(16, 22, f"One generation: {p['pop_size']} genomes of 96 weights", "f-title")
    box_w, box_h, gap, left, top = 150, 78, 30, 16, 58
    for i, (title, lines) in enumerate(steps):
        x = left + i * (box_w + gap)
        svg.rect(x, top, box_w, box_h, "f-box")
        svg.text(x + 10, top + 24, title, "f-end")
        for j, line in enumerate(lines):
            svg.text(x + 10, top + 46 + j * 15, line, "f-note")
        if i < len(steps) - 1:
            ax = x + box_w
            svg.line(ax, top + box_h / 2, ax + gap - 7, top + box_h / 2, "f-axis")
            svg.polygon(
                [
                    (ax + gap, top + box_h / 2),
                    (ax + gap - 8, top + box_h / 2 - 4),
                    (ax + gap - 8, top + box_h / 2 + 4),
                ],
                "f-arrow",
            )
    # Return path from the last box back to the first.
    x_last = left + 3 * (box_w + gap) + box_w / 2
    x_first = left + box_w / 2
    y_bottom = top + box_h
    y_return = y_bottom + 34
    svg.path(
        [(x_last, y_bottom), (x_last, y_return), (x_first, y_return), (x_first, y_bottom + 7)],
        "f-axis",
    )
    svg.polygon(
        [(x_first, y_bottom), (x_first - 4, y_bottom + 8), (x_first + 4, y_bottom + 8)], "f-arrow"
    )
    svg.text(
        (x_first + x_last) / 2,
        y_return + 18,
        "the children are the next generation",
        "f-note",
        "middle",
    )
    svg.text(
        16,
        height - 14,
        "Fitness is the mean over seeds, so one lucky obstacle sequence cannot decide who breeds.",
        "f-note",
    )
    return svg.render()


# --- tables ----------------------------------------------------------------------------------
def md_table(headers: list[str], rows: list[list[str]], align: list[str] | None = None) -> str:
    align = align or ["l"] + ["r"] * (len(headers) - 1)
    sep = ["---:" if a == "r" else ":---" for a in align]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def parse_md_table(md: str) -> dict[str, Any]:
    """Headers, alignment and rows of a pipe table written by ``md_table``. The web page renders
    tables from this, so the page and the Markdown always show the same numbers."""
    lines = [ln for ln in md.strip().splitlines() if ln.startswith("|")]
    cells = [[c.strip() for c in ln.strip("|").split(" | ")] for ln in lines]
    align = ["r" if c.endswith(":") else "l" for c in cells[1]]
    return {"headers": cells[0], "align": align, "rows": cells[2:]}


def pm(mean: float, half: float) -> str:
    return f"{thousands(mean)}" if math.isnan(half) else f"{thousands(mean)} ± {thousands(half)}"


def headline_table(main: list[dict[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    g = group(main)
    out_rows, md = [], []
    for rid in ["rule_fixed", "rule_scaled", "ppo", "cmaes", "ga", "neat"]:
        if rid not in g:
            continue
        rs = g[rid]
        means = [r["test"]["mean"] for r in rs]
        m, half = mean_ci95(means)
        hits = [(steps_to_threshold(r), r) for r in rs]
        hit_steps = [s for s, _ in hits if s is not None]
        hit_gens = [
            next(row["generation"] for row in r["history"] if row["best"] >= 1000)
            for s, r in hits
            if s is not None
        ]
        learned = rid in LEARNED
        wall = statistics.fmean(r["wall_clock_s"] for r in rs)
        rec = {
            "run_id": rid, "n_runs": len(rs), "best_run": max(means), "mean": m, "ci95": half,
            "reached": len(hit_steps), "steps_to_1000": statistics.median(hit_steps) if hit_steps else None,
            "generations_to_1000": statistics.median(hit_gens) if hit_gens else None, "wall_clock_s": wall,
        }  # fmt: skip
        out_rows.append(rec)
        steps_cell = (
            "not reached"
            if not hit_steps
            else f"{statistics.median(hit_steps) / 1e6:.2f} M ({len(hit_steps)} of {len(rs)} runs)"
        )
        gen_cell = (
            "-"
            if not hit_gens or rid == "ppo" or not learned
            else f"{statistics.median(hit_gens):.0f}"
        )
        md.append(
            [
                label(rid),
                thousands(max(means)),
                pm(m, half),
                steps_cell if learned else "-",
                gen_cell,
                f"{wall:,.0f}",
            ]
        )
    table = md_table(
        [
            "Algorithm",
            "Best run",
            "Mean ± 95% CI",
            "Steps to 1,000 points",
            "Generations",
            "Wall-clock per run (s)",
        ],
        md,
    )
    return table, out_rows


def ablation_table(abl: list[dict[str, Any]]) -> str:
    g = group(abl)
    md = []
    for title, items in ABLATION_PANELS:
        for rid, name in items:
            if rid not in g:
                continue
            rs = g[rid]
            m, half = mean_ci95([r["test"]["mean"] for r in rs])
            hits = [s for r in rs if (s := steps_to_threshold(r)) is not None]
            md.append([title, name + (" (default)" if rid == "ga-default" else ""),
                       pm(m, half), f"{len(hits)} of {len(rs)}",
                       f"{statistics.median(hits) / 1e6:.2f} M" if hits else "-"])  # fmt: skip
    return md_table(
        [
            "Factor",
            "Setting",
            "Test score, mean ± 95% CI",
            "Runs reaching 1,000",
            "Median steps to 1,000",
        ],
        md,
        ["l", "l", "r", "r", "r"],
    )


def generalisation_table(rows: list[dict[str, Any]]) -> str:
    md = []
    for a in LEARNED:
        pts = [r for r in rows if r["algorithm"] == a]
        if not pts:
            continue
        train = statistics.fmean(r["train_fitness"] for r in pts)
        test, half = mean_ci95([r["test_at_train_cap"] for r in pts])
        full, fhalf = mean_ci95([r["test_full"] for r in pts])
        capped = statistics.fmean(r["reached_cap_fraction"] for r in pts)
        md.append(
            [
                label(a),
                thousands(train),
                pm(test, half),
                f"{train - test:,.0f}",
                f"{capped * 100:.0f}%",
                pm(full, fhalf),
            ]
        )
    return md_table(
        [
            "Algorithm",
            "Best training fitness",
            "Test at training cap",
            "Gap",
            "Test seeds played to the cap",
            "Test at 36,000 ticks",
        ],
        md,
    )


# --- driver ----------------------------------------------------------------------------------
def write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.relative_to(REPO) if path.is_relative_to(REPO) else path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="main")
    ap.add_argument("--ablations", default="ablations")
    ap.add_argument("--out-dir", type=Path, default=REPO / "report")
    args = ap.parse_args()

    main_runs = load_runs(args.suite)
    abl_runs = load_runs(args.ablations)
    gen_path = RESULTS_ROOT / args.suite / "generalisation.json"
    gen_rows = json.loads(gen_path.read_text(encoding="utf-8")) if gen_path.exists() else []
    fig_dir, tab_dir = args.out_dir / "figures", args.out_dir / "tables"

    figures = {
        "learning_curves": lambda: learning_curves(main_runs),
        "final_scores": lambda: final_scores(main_runs),
        "sample_efficiency": lambda: sample_efficiency(main_runs),
        "generalisation": lambda: generalisation(gen_rows),
        "ablations": lambda: ablations(abl_runs),
        "decision_maps": lambda: decision_maps(main_runs, RESULTS_ROOT),
        "neat_growth": lambda: neat_growth(main_runs),
        "loop": loop_diagram,
    }
    for name, build in figures.items():
        svg = build() if (main_runs or abl_runs or name == "loop") else None
        if svg is None:
            print(f"skipped {name}: no data")
            continue
        write(fig_dir / f"{name}.svg", svg)

    tables: dict[str, Any] = {}
    if main_runs:
        table, rows = headline_table(main_runs)
        write(tab_dir / "headline.md", table)
        text = json.dumps(finite(rows), indent=1, allow_nan=False)
        write(tab_dir / "headline.json", text + NEWLINE)
        tables["headline"] = parse_md_table(table)
    if abl_runs:
        table = ablation_table(abl_runs)
        write(tab_dir / "ablations.md", table)
        tables["ablations"] = parse_md_table(table)
    if gen_rows:
        table = generalisation_table(gen_rows)
        write(tab_dir / "generalisation.md", table)
        tables["generalisation"] = parse_md_table(table)
    if main_runs:
        # Index for the web app's Watch page: which genome file belongs to which run and score.
        index = [
            {
                "run_id": r["run_id"],
                "algorithm": r["algorithm"],
                "seed": r["seed"],
                "test_mean": r["test"]["mean"],
                "genome_file": r["genome_file"],
            }
            for r in sorted(main_runs, key=lambda r: (r["run_id"], r["seed"]))
        ]
        write(tab_dir / "genomes.json", json.dumps(finite(index), indent=1) + NEWLINE)
    if tables:
        write(tab_dir / "tables.json", json.dumps(tables, indent=1, ensure_ascii=False) + NEWLINE)


if __name__ == "__main__":
    main()
