"""Hand-written baseline: jump (or duck) when the next obstacle is closer than d pixels.

This is the sanity floor for every learning algorithm. If an evolved network cannot beat a
one-line rule, something is wrong with the training, not with the rule.

Two variants are tuned and reported:

* ``fixed``: one distance ``d`` for all speeds. This is the rule as the project brief
  describes it.
* ``scaled``: ``d + gain * (speed - START_SPEED)``. The game speeds up, a jump covers more
  ground at higher speed, and a fixed ``d`` lands too late. The variant exists to show how
  much of the remaining gap comes from that one effect.

Run ``python -m sim.algorithms.rule_based`` to tune on ``TUNE_SEEDS`` and score on
``TEST_SEEDS``. The output goes to ``results/baselines/rule_based.json``.
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from sim.evaluation import EvalConfig, Recorder, RunResult
from sim.game import (
    DEFAULT_MAX_TICKS,
    DINO_H,
    DINO_W,
    DINO_X,
    DUCK,
    DUCK_H,
    JUMP,
    NOOP,
    START_SPEED,
    DinoGame,
    rollout,
)
from sim.genome_io import rule_payload
from sim.provenance import git_commit, machine_info
from sim.seeds import TEST_SEEDS, TUNE_SEEDS

RESULT_PATH = Path(__file__).resolve().parents[2] / "results" / "baselines" / "rule_based.json"


@dataclass(frozen=True)
class RuleParams:
    d: float
    gain: float = 0.0  # extra trigger distance per unit of speed above START_SPEED


def rule_action(game: DinoGame, params: RuleParams) -> int:
    """Choose an action from the next obstacle alone.

    Birds whose bottom edge clears the standing dino are ignored, those that clear the
    ducking dino are ducked under, everything else is jumped. While airborne the rule
    does nothing, because a duck in the air would trigger the fast fall.
    """
    if game.airborne:
        return NOOP
    first, _ = game.upcoming()
    if first is None:
        return NOOP
    trigger = params.d + params.gain * (game.speed - START_SPEED)
    if first[0] - (DINO_X + DINO_W) >= trigger:
        return NOOP
    bottom = first[1]
    if bottom >= DINO_H:
        return NOOP
    return DUCK if bottom >= DUCK_H else JUMP


def evaluate(
    params: RuleParams, seeds: tuple[int, ...], max_ticks: int = DEFAULT_MAX_TICKS
) -> list[float]:
    """Score of the rule on each seed."""
    return [rollout(s, lambda g: rule_action(g, params), max_ticks).score for s in seeds]


def tune(
    seeds: tuple[int, ...] = TUNE_SEEDS,
    d_grid: range = range(20, 301, 5),
    gain_grid: tuple[float, ...] = (0.0,),
    max_ticks: int = DEFAULT_MAX_TICKS,
    on_ticks: Callable[[int], None] | None = None,
) -> tuple[RuleParams, float]:
    """Grid search for the highest mean score. Ties go to the smaller d and gain.

    ``on_ticks`` receives the number of game ticks each candidate used, so that callers can
    count the tuning effort in environment steps.
    """
    best = (RuleParams(float(d_grid[0])), -1.0)
    for gain in gain_grid:
        for d in d_grid:
            params = RuleParams(float(d), gain)
            results = [rollout(s, lambda g, p=params: rule_action(g, p), max_ticks) for s in seeds]
            if on_ticks:
                on_ticks(sum(r.ticks for r in results))
            mean = statistics.fmean(r.score for r in results)
            if mean > best[1]:
                best = (params, mean)
    return best


# Grids used when the rule runs as one of the experiment's algorithms. Coarser than the ones in
# main(): the experiment repeats the tuning once per run seed.
RUN_GRIDS = {
    "rule_fixed": (range(20, 301, 10), (0.0,)),
    "rule_scaled": (range(20, 201, 10), (0.0, 8.0, 16.0, 24.0, 32.0, 40.0)),
}
RUN_TUNE_SEEDS = 20


def run(
    params: dict[str, object],
    eval_cfg: EvalConfig,
    budget_steps: int,
    seed: int,
    progress: Callable[[dict[str, object]], None] | None = None,
    *,
    variant: str = "rule_fixed",
) -> RunResult:
    """Tune the rule on ``RUN_TUNE_SEEDS`` seeds taken from the tuning range, offset by the run
    seed so that repeats differ. ``budget_steps`` is ignored: the grid search is the budget, and
    the ticks it simulates are counted as the run's environment steps."""
    rec = Recorder(variant, seed)
    start = TUNE_SEEDS[0] + RUN_TUNE_SEEDS * seed
    seeds = tuple(range(start, start + RUN_TUNE_SEEDS))
    d_grid, gain_grid = RUN_GRIDS[variant]
    best, mean = tune(seeds, d_grid, gain_grid, eval_cfg.max_ticks, rec.add_steps)
    rec.generation([mean], tuned_d=best.d, tuned_gain=best.gain)
    rec.result.best_fitness = mean
    rec.result.best_genome = rule_payload(
        best.d, best.gain, {"algorithm": variant, "seed": seed, "tune_seeds": [seeds[0], seeds[-1]]}
    )
    return rec.finish()


def _summary(scores: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(scores),
        "median": statistics.median(scores),
        "min": min(scores),
        "max": max(scores),
    }


def main() -> None:
    started = time.perf_counter()
    report: dict[str, object] = {
        "commit": git_commit(),
        "machine": machine_info(),
        "tune_seeds": [TUNE_SEEDS[0], TUNE_SEEDS[-1], len(TUNE_SEEDS)],
        "test_seeds": [TEST_SEEDS[0], TEST_SEEDS[-1], len(TEST_SEEDS)],
        "variants": {},
    }
    variants = {
        "fixed": (range(20, 301, 5), (0.0,)),
        "scaled": (range(20, 201, 5), tuple(float(g) for g in range(0, 41, 4))),
    }
    for name, (d_grid, gain_grid) in variants.items():
        params, tune_mean = tune(TUNE_SEEDS, d_grid, gain_grid)
        test_scores = evaluate(params, TEST_SEEDS)
        report["variants"][name] = {  # type: ignore[index]
            "params": asdict(params),
            "tune_mean_score": tune_mean,
            "test": _summary(test_scores),
            "test_scores": test_scores,
        }
        print(
            f"{name:7s} d={params.d:5.0f} gain={params.gain:4.1f} "
            f"tune {tune_mean:8.1f}  test mean {statistics.fmean(test_scores):8.1f}"
        )
    report["wall_clock_s"] = time.perf_counter() - started
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
