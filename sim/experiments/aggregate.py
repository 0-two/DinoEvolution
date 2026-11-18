"""Collect a suite's run files into one summary table.

Reads ``results/<suite>/runs/*/seed_*.json`` and writes ``results/<suite>/summary.csv``.
Every number in the README and report tables comes from this file or the run files it reads;
nothing is typed in by hand.

Columns:

``test_mean``, ``test_ci95``  mean over runs of each run's mean score on the unseen test seeds,
                              with the half-width of the 95% t-interval across runs
``best_run``                  highest single-run test mean
``reached_1000``              how many runs had a generation whose best fitness reached 1,000
``steps_to_1000``             median environment steps to that point, among the runs that got there
``env_steps``, ``wall_s``     mean total environment steps and wall-clock seconds per run
"""

from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

from sim.experiments.runner import RESULTS_ROOT
from sim.stats import mean_ci95

THRESHOLD = 1000.0
COLUMNS = [
    "run_id",
    "algorithm",
    "n_runs",
    "test_mean",
    "test_ci95",
    "best_run",
    "reached_1000",
    "steps_to_1000",
    "env_steps",
    "wall_s",
]


def load_runs(suite: str, root: Path = RESULTS_ROOT) -> list[dict[str, Any]]:
    paths = sorted((root / suite / "runs").glob("*/seed_*.json"))
    return [json.loads(p.read_text(encoding="utf-8")) for p in paths]


def steps_to_threshold(run: dict[str, Any], threshold: float = THRESHOLD) -> int | None:
    """Environment steps at the first generation whose best fitness reaches ``threshold``."""
    for row in run["history"]:
        if row["best"] >= threshold:
            return int(row["env_steps"])
    return None


def summarise(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[run["run_id"]].append(run)
    rows = []
    for run_id, group in groups.items():
        means = [r["test"]["mean"] for r in group]
        mean, half = mean_ci95(means)
        reached = [s for r in group if (s := steps_to_threshold(r)) is not None]
        rows.append(
            {
                "run_id": run_id,
                "algorithm": group[0]["algorithm"],
                "n_runs": len(group),
                "test_mean": mean,
                "test_ci95": half,
                "best_run": max(means),
                "reached_1000": len(reached),
                "steps_to_1000": statistics.median(reached) if reached else None,
                "env_steps": statistics.fmean(r["env_steps"] for r in group),
                "wall_s": statistics.fmean(r["wall_clock_s"] for r in group),
            }
        )
    return rows


def write_summary(suite: str, root: Path = RESULTS_ROOT) -> list[dict[str, Any]]:
    rows = summarise(load_runs(suite, root))
    out = root / suite / "summary.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: ("" if v is None else v) for k, v in row.items()})
    return rows


def format_table(rows: list[dict[str, Any]]) -> str:
    def num(v: Any, spec: str) -> str:
        return "-" if v is None or v != v else format(v, spec)

    lines = [
        f"{'run':18s} {'n':>3s} {'test mean':>10s} {'+-95%':>8s} {'best run':>9s} "
        f"{'reached':>8s} {'steps to 1000':>14s} {'wall s':>8s}"
    ]
    for r in rows:
        lines.append(
            f"{r['run_id']:18s} {r['n_runs']:3d} {num(r['test_mean'], '10.0f')} "
            f"{num(r['test_ci95'], '8.0f')} {num(r['best_run'], '9.0f')} "
            f"{r['reached_1000']:>4d}/{r['n_runs']:<3d} {num(r['steps_to_1000'], '14,.0f')} "
            f"{num(r['wall_s'], '8.0f')}"
        )
    return "\n".join(lines)
