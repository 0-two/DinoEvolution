"""Scoring a policy, and the bookkeeping every algorithm shares.

Fitness is the mean score over K seeds, so one lucky obstacle sequence cannot decide who
breeds. Every game tick that is simulated counts as one environment step. That number is the
common axis for comparing algorithms.

Reward shaping is behind ``EvalConfig.jump_penalty`` and is zero by default. When it is set,
each jump taken from the ground subtracts that many score points from the episode. It exists
to test whether discouraging pointless jumping helps; the reported results use zero.
"""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable, Iterable
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from sim.game import DEFAULT_MAX_TICKS, JUMP, DinoGame
from sim.seeds import TRAIN_SEED_LIMIT


@dataclass(frozen=True)
class EvalConfig:
    k_seeds: int = 3
    max_ticks: int = DEFAULT_MAX_TICKS
    jump_penalty: float = 0.0


def play(
    act: Callable[[list[float]], int], seed: int, max_ticks: int, jump_penalty: float = 0.0
) -> tuple[float, int]:
    """One episode. Returns (score, ticks played). The score includes any shaping penalty."""
    game = DinoGame(seed, max_ticks)
    observe, step = game.observe, game.step
    jumps = 0
    if jump_penalty:
        while not game.done:
            a = act(observe())
            if a == JUMP and not game.airborne:
                jumps += 1
            step(a)
    else:
        while not game.done:
            step(act(observe()))
    return game.score - jump_penalty * jumps, game.tick


def fitness(
    act: Callable[[list[float]], int], seeds: Iterable[int], cfg: EvalConfig
) -> tuple[float, int]:
    """Mean score over ``seeds`` and the total ticks simulated."""
    scores = []
    steps = 0
    for seed in seeds:
        score, ticks = play(act, seed, cfg.max_ticks, cfg.jump_penalty)
        scores.append(score)
        steps += ticks
    return sum(scores) / len(scores), steps


def draw_seeds(rng: np.random.Generator, count: int) -> list[int]:
    """Training seeds, always below ``TRAIN_SEED_LIMIT`` so they never touch tuning or test."""
    return [int(s) for s in rng.integers(0, TRAIN_SEED_LIMIT, size=count)]


@dataclass
class RunResult:
    """What one algorithm run hands back. The runner adds config, commit and test scores."""

    algorithm: str
    seed: int
    history: list[dict[str, Any]] = field(default_factory=list)
    env_steps: int = 0
    wall_clock_s: float = 0.0
    best_genome: dict[str, Any] | None = None
    best_fitness: float = float("-inf")
    extras: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["best_fitness"] = self.best_fitness
        return out


class Recorder:
    """Collects one history row per generation and tracks steps and time."""

    def __init__(self, algorithm: str, seed: int) -> None:
        self.result = RunResult(algorithm, seed)
        self._start = time.perf_counter()

    def add_steps(self, steps: int) -> None:
        self.result.env_steps += steps

    def generation(self, fitnesses: list[float], **extra: Any) -> dict[str, Any]:
        """Log best, mean and median fitness of the generation that was just evaluated."""
        row = {
            "generation": len(self.result.history) + 1,
            "env_steps": self.result.env_steps,
            "wall_s": time.perf_counter() - self._start,
            "best": max(fitnesses),
            "mean": statistics.fmean(fitnesses),
            "median": statistics.median(fitnesses),
            **extra,
        }
        self.result.history.append(row)
        return row

    def finish(self) -> RunResult:
        self.result.wall_clock_s = time.perf_counter() - self._start
        return self.result
