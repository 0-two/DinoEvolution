"""Measure how many simulated ticks per second one core can run.

Three workloads, each in ghost mode (collisions off) so that every run lasts the full
36,000 ticks and passes through every speed band, with obstacles on screen throughout:

* ``step``: the bare game step, replaying a pre-recorded action list.
* ``step+observe``: the step plus building the 9 network inputs every tick. A learning
  agent pays at least this much.
* ``step+rule``: the step plus the rule-based policy, a realistic agent loop.

The brief asks for at least 50,000 ticks/s per core. A neural policy adds its own forward
pass on top of ``step+observe``; that cost is measured when the network exists.

    uv run python -m sim.bench
"""

from __future__ import annotations

import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

from sim.algorithms.rule_based import RuleParams, rule_action
from sim.game import DEFAULT_MAX_TICKS, DinoGame
from sim.provenance import git_commit, machine_info

TARGET_TICKS_PER_S = 50_000
RESULT_PATH = Path(__file__).resolve().parent.parent / "results" / "benchmarks" / "sim_speed.json"
EPISODES_PER_REPEAT = 20
REPEATS = 5


def _ghost(seed: int) -> DinoGame:
    return DinoGame(seed, DEFAULT_MAX_TICKS, collisions=False)


def _record_actions() -> list[int]:
    params = RuleParams(100.0)
    game = _ghost(0)
    actions = []
    while not game.done:
        a = rule_action(game, params)
        actions.append(a)
        game.step(a)
    return actions


def _bare_step(actions: list[int]) -> int:
    ticks = 0
    for seed in range(EPISODES_PER_REPEAT):
        game = _ghost(seed)
        step = game.step
        for a in actions:
            step(a)
        ticks += game.tick
    return ticks


def _step_observe() -> int:
    ticks = 0
    for seed in range(EPISODES_PER_REPEAT):
        game = _ghost(seed)
        step, observe = game.step, game.observe
        while not game.done:
            observe()
            step(0)
        ticks += game.tick
    return ticks


def _step_rule() -> int:
    params = RuleParams(100.0)
    ticks = 0
    for seed in range(EPISODES_PER_REPEAT):
        game = _ghost(seed)
        step = game.step
        while not game.done:
            step(rule_action(game, params))
        ticks += game.tick
    return ticks


def _measure(workload: Callable[[], int]) -> dict[str, float]:
    workload()  # warm-up
    rates = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        ticks = workload()
        rates.append(ticks / (time.perf_counter() - start))
    return {
        "ticks_per_s_median": statistics.median(rates),
        "ticks_per_s_best": max(rates),
        "ticks_per_s_worst": min(rates),
    }


def main() -> None:
    actions = _record_actions()
    workloads: dict[str, Callable[[], int]] = {
        "step": lambda: _bare_step(actions),
        "step+observe": _step_observe,
        "step+rule": _step_rule,
    }
    results = {name: _measure(fn) for name, fn in workloads.items()}
    report = {
        "commit": git_commit(),
        "machine": machine_info(),
        "episode_ticks": DEFAULT_MAX_TICKS,
        "episodes_per_repeat": EPISODES_PER_REPEAT,
        "repeats": REPEATS,
        "target_ticks_per_s": TARGET_TICKS_PER_S,
        "results": results,
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, indent=2) + "\n")
    for name, r in results.items():
        median = r["ticks_per_s_median"]
        verdict = "meets" if median >= TARGET_TICKS_PER_S else "BELOW"
        print(f"{name:13s} median {median:11,.0f} ticks/s  ({verdict} {TARGET_TICKS_PER_S:,})")
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
