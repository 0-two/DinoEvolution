"""Learning curves on a common axis of environment steps, with confidence bands.

Every run logs its history at its own steps (one row per generation, or per window for PPO).
To average over seeds the histories are resampled as step functions: the value at a grid point
is the value of the last row at or before it. The grid stops at the smallest final step count
among the runs, so every run contributes at every point and no curve is extrapolated.
"""

from __future__ import annotations

import bisect
import math
from typing import Any

from sim.stats import mean_ci95


def resample(run: dict[str, Any], key: str, grid: list[float]) -> list[float | None]:
    """Value of ``key`` for the last history row at or before each grid step (None before the first)."""
    steps = [row["env_steps"] for row in run["history"]]
    values = [row[key] for row in run["history"]]
    out: list[float | None] = []
    for g in grid:
        i = bisect.bisect_right(steps, g) - 1
        out.append(values[i] if i >= 0 else None)
    return out


def curve_stats(
    runs: list[dict[str, Any]], key: str, n_points: int = 160
) -> dict[str, list[float]]:
    """Mean and 95% interval across runs of ``key`` against environment steps (in millions).

    Returns ``steps_m``, ``mean``, ``lo`` and ``hi``. With one run the interval collapses to
    the mean.
    """
    first = max(run["history"][0]["env_steps"] for run in runs)
    last = min(run["history"][-1]["env_steps"] for run in runs)
    grid = [first + (last - first) * i / (n_points - 1) for i in range(n_points)]
    resampled = [resample(run, key, grid) for run in runs]
    out: dict[str, list[float]] = {
        "steps_m": [g / 1e6 for g in grid],
        "mean": [],
        "lo": [],
        "hi": [],
    }
    for i in range(n_points):
        column = [r[i] for r in resampled if r[i] is not None]
        mean, half = mean_ci95(column)
        out["mean"].append(mean)
        out["lo"].append(mean - (0.0 if math.isnan(half) else half))
        out["hi"].append(mean + (0.0 if math.isnan(half) else half))
    return out


def by_generation(runs: list[dict[str, Any]], key: str) -> dict[str, list[float]]:
    """Mean and 95% interval across runs of ``key`` against generation number, up to the
    shortest run."""
    n = min(len(run["history"]) for run in runs)
    out: dict[str, list[float]] = {"generation": [], "mean": [], "lo": [], "hi": []}
    for g in range(n):
        column = [float(run["history"][g][key]) for run in runs]
        mean, half = mean_ci95(column)
        out["generation"].append(g + 1)
        out["mean"].append(mean)
        out["lo"].append(mean - (0.0 if math.isnan(half) else half))
        out["hi"].append(mean + (0.0 if math.isnan(half) else half))
    return out
