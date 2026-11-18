"""Run one job: one algorithm, one seed, then test the result on unseen seeds.

A job writes two files under ``results/<suite>/``:

``runs/<run_id>/seed_<n>.json``     everything about the run: configuration, git commit,
                                    seed, per-generation best/mean/median fitness, wall-clock
                                    time, environment steps, and the test scores
``genomes/<run_id>/seed_<n>.json``  the best genome alone, in a format the web app can load

The best genome is tested by reading it back from the exported payload, never from the
in-memory object, so a broken export shows up as a bad score here and not later in the browser.
"""

from __future__ import annotations

import json
import math
import statistics
import time
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from functools import partial
from pathlib import Path
from typing import Any

from sim.evaluation import EvalConfig
from sim.experiments.config import RunSpec
from sim.game import DinoGame
from sim.genome_io import load_game_policy, save
from sim.provenance import git_commit, library_versions, machine_info
from sim.seeds import TEST_SEEDS

RESULTS_ROOT = Path(__file__).resolve().parents[2] / "results"


@dataclass(frozen=True)
class Job:
    suite: str
    run: RunSpec
    seed: int
    test_max_ticks: int
    test_n_seeds: int
    results_root: str = str(RESULTS_ROOT)

    @property
    def run_path(self) -> Path:
        return (
            Path(self.results_root) / self.suite / "runs" / self.run.id / f"seed_{self.seed}.json"
        )

    @property
    def genome_path(self) -> Path:
        return (
            Path(self.results_root)
            / self.suite
            / "genomes"
            / self.run.id
            / f"seed_{self.seed}.json"
        )


def algorithm_runner(name: str) -> Any:
    """Look up ``run(params, eval_cfg, budget_steps, seed, progress)``. Imports are lazy so a
    worker only loads the libraries its algorithm needs."""
    if name == "ga":
        from sim.algorithms import ga

        return ga.run
    if name == "cmaes":
        from sim.algorithms import cmaes

        return cmaes.run
    if name == "neat":
        from sim.algorithms import neat_runner

        return neat_runner.run
    if name == "ppo":
        from sim.algorithms import ppo_baseline

        return ppo_baseline.run
    if name in ("rule_fixed", "rule_scaled"):
        from sim.algorithms import rule_based

        return partial(rule_based.run, variant=name)
    raise ValueError(f"unknown algorithm {name!r}")


def score_on_test_seeds(payload: dict[str, Any], n_seeds: int, max_ticks: int) -> list[float]:
    """Play the exported genome on the first ``n_seeds`` held-out test seeds."""
    act = load_game_policy(payload)
    scores = []
    for seed in TEST_SEEDS[:n_seeds]:
        game = DinoGame(seed, max_ticks)
        while game.step(act(game)):
            pass
        scores.append(game.score)
    return scores


def finite(value: Any) -> Any:
    """Replace NaN and infinities by None, recursively. JSON has no such numbers."""
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {k: finite(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [finite(v) for v in value]
    return value


def execute(job: Job) -> dict[str, Any]:
    """Run ``job`` and write its result files. Returns a short summary for progress output."""
    spec = job.run
    started = datetime.now(UTC).isoformat(timespec="seconds")
    result = algorithm_runner(spec.algorithm)(spec.params, spec.eval, spec.budget_steps, job.seed)
    assert result.best_genome is not None

    t0 = time.perf_counter()
    scores = score_on_test_seeds(result.best_genome, job.test_n_seeds, job.test_max_ticks)
    record = {
        "suite": job.suite,
        "run_id": spec.id,
        "algorithm": spec.algorithm,
        "seed": job.seed,
        "params": spec.params,
        "eval": asdict(spec.eval),
        "budget_steps": spec.budget_steps,
        "commit": git_commit(),
        "machine": machine_info(),
        "libraries": library_versions(),
        "started_utc": started,
        "env_steps": result.env_steps,
        "wall_clock_s": result.wall_clock_s,
        "best_train_fitness": result.best_fitness,
        "history": result.history,
        "extras": result.extras,
        "test": {
            "max_ticks": job.test_max_ticks,
            "seeds": [TEST_SEEDS[0], TEST_SEEDS[job.test_n_seeds - 1], job.test_n_seeds],
            "scores": scores,
            "mean": statistics.fmean(scores),
            "median": statistics.median(scores),
            "wall_clock_s": time.perf_counter() - t0,
        },
        "genome_file": str(job.genome_path.relative_to(Path(job.results_root).parent)).replace(
            "\\", "/"
        ),
    }
    save(result.best_genome, job.genome_path)
    job.run_path.parent.mkdir(parents=True, exist_ok=True)
    job.run_path.write_text(
        json.dumps(finite(record), indent=1, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {
        "run_id": spec.id,
        "seed": job.seed,
        "env_steps": result.env_steps,
        "wall_clock_s": result.wall_clock_s,
        "test_mean": record["test"]["mean"],
    }


__all__ = ["RESULTS_ROOT", "EvalConfig", "Job", "execute", "score_on_test_seeds"]
