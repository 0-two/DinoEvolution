"""Experiment suites, read from YAML.

A suite lists runs. Each run is one algorithm with one set of parameters, repeated over the
suite's seeds. Example::

    suite: quick
    seeds: [0, 1, 2]
    budget_steps: 2000000          # environment steps; ticks of the game, summed over episodes
    eval: {k_seeds: 3, max_ticks: 12000, jump_penalty: 0.0}
    test: {max_ticks: 36000, n_seeds: 100}
    runs:
      - {id: ga, algorithm: ga}
      - {id: ga-pop50, algorithm: ga, params: {pop_size: 50}}
      - {id: ppo, algorithm: ppo, budget_steps: 300000}

A run may override ``budget_steps`` and any key of ``eval``. Run ids become directory names, so
they use letters, digits, dashes and underscores only.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from sim.evaluation import EvalConfig

SUITE_DIR = Path(__file__).resolve().parent / "suites"
ALGORITHMS = ("ga", "cmaes", "neat", "ppo", "rule_fixed", "rule_scaled")


@dataclass(frozen=True)
class RunSpec:
    id: str
    algorithm: str
    params: dict[str, Any]
    budget_steps: int
    eval: EvalConfig


@dataclass(frozen=True)
class Suite:
    name: str
    seeds: tuple[int, ...]
    test_max_ticks: int
    test_n_seeds: int
    runs: tuple[RunSpec, ...]
    raw: dict[str, Any] = field(repr=False, default_factory=dict)


def load_suite(name_or_path: str | Path) -> Suite:
    path = Path(name_or_path)
    if not path.exists():
        path = SUITE_DIR / f"{name_or_path}.yaml"
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    base_eval = raw.get("eval", {})
    default_budget = int(raw["budget_steps"])
    runs = []
    seen: set[str] = set()
    for r in raw["runs"]:
        run_id = str(r["id"])
        if not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
            raise ValueError(f"run id {run_id!r} must use letters, digits, - and _ only")
        if run_id in seen:
            raise ValueError(f"duplicate run id {run_id!r}")
        seen.add(run_id)
        if r["algorithm"] not in ALGORITHMS:
            raise ValueError(f"unknown algorithm {r['algorithm']!r}; choose from {ALGORITHMS}")
        runs.append(
            RunSpec(
                id=run_id,
                algorithm=r["algorithm"],
                params=dict(r.get("params", {})),
                budget_steps=int(r.get("budget_steps", default_budget)),
                eval=EvalConfig(**{**base_eval, **r.get("eval", {})}),
            )
        )
    test = raw.get("test", {})
    return Suite(
        name=str(raw["suite"]),
        seeds=tuple(int(s) for s in raw["seeds"]),
        test_max_ticks=int(test.get("max_ticks", 36_000)),
        test_n_seeds=int(test.get("n_seeds", 100)),
        runs=tuple(runs),
        raw=raw,
    )
