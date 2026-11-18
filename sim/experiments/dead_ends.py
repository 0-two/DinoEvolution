"""Re-measure the settings that did not work, so the README can quote numbers from a file.

Three things went wrong during the project and were changed. This script reruns each one and
writes ``results/notes/dead_ends.json``:

``gap_coefficient``   the original game (0.6) against the one used (0.5): how a tuned speed-aware
                      rule scores on the 100 test seeds, and on how many it survives to the cap
``ppo_first_settings`` PPO with learning rate 3e-4 and discount 0.995 against the settings used
                      (1e-3 and 0.99), mean episode score per window over the first 600k steps
``neat_threshold``    NEAT's speciation threshold 3.0 against 1.0: how many species appear

    uv run python -m sim.experiments.dead_ends
"""

from __future__ import annotations

import json
import statistics
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from sim.evaluation import EvalConfig
from sim.experiments.runner import RESULTS_ROOT, finite
from sim.provenance import git_commit, library_versions, machine_info
from sim.seeds import TEST_SEEDS, TUNE_SEEDS

OUT = RESULTS_ROOT / "notes" / "dead_ends.json"
TRAIN_CAP = 12_000
TEST_CAP = 36_000


def gap_coefficient(value: float) -> dict[str, Any]:
    """Tune the speed-scaled rule with GAP_COEFF set to ``value`` and score it on the test seeds."""
    import sim.game as game
    from sim.algorithms import rule_based

    game.GAP_COEFF = value  # read at each spawn, so setting it here changes the whole game
    d_grid, gain_grid = rule_based.RUN_GRIDS["rule_scaled"]
    params, _ = rule_based.tune(TUNE_SEEDS[:20], d_grid, gain_grid, TRAIN_CAP)
    scores = rule_based.evaluate(params, TEST_SEEDS, TEST_CAP)
    ceiling = max(scores)
    return {
        "gap_coefficient": value,
        "tuned_d": params.d,
        "tuned_gain": params.gain,
        "test_mean": statistics.fmean(scores),
        "test_median": statistics.median(scores),
        "seeds_at_cap": sum(s >= ceiling - 1e-6 for s in scores) if ceiling > 0 else 0,
        "n_seeds": len(scores),
        "max_possible": _cap_score(TEST_CAP),
    }


def _cap_score(ticks: int) -> float:
    from sim.game import DinoGame

    g = DinoGame(0, ticks, collisions=False)
    while not g.done:
        g.step(0)
    return g.score


def ppo_windows(params: dict[str, Any], steps: int = 600_000) -> dict[str, Any]:
    from sim.algorithms import ppo_baseline

    result = ppo_baseline.run({**params, "log_every": 100_000}, EvalConfig(3, TRAIN_CAP), steps, 1)
    return {
        "params": params,
        "env_steps": [h["env_steps"] for h in result.history],
        "window_mean": [h["mean"] for h in result.history],
        "window_best": [h["best"] for h in result.history],
    }


def neat_species(threshold: float) -> dict[str, Any]:
    from sim.algorithms import neat_runner

    result = neat_runner.run(
        {"compatibility_threshold": threshold}, EvalConfig(3, TRAIN_CAP), 2_500_000, 2
    )
    species = [h["species"] for h in result.history]
    return {
        "threshold": threshold,
        "generations": len(species),
        "species_min": min(species),
        "species_max": max(species),
        "species_last": species[-1],
        "best_fitness": result.best_fitness,
    }


def main() -> None:
    started = time.perf_counter()
    jobs = {
        "gap_0.6": (gap_coefficient, (0.6,)),
        "gap_0.5": (gap_coefficient, (0.5,)),
        "ppo_first": (ppo_windows, ({"learning_rate": 3e-4, "gamma": 0.995, "ent_coef": 0.01},)),
        "ppo_final": (ppo_windows, ({},)),
        "neat_3.0": (neat_species, (3.0,)),
        "neat_1.0": (neat_species, (1.0,)),
    }
    with ProcessPoolExecutor(max_workers=len(jobs)) as pool:
        futures = {name: pool.submit(fn, *args) for name, (fn, args) in jobs.items()}
        out = {name: f.result() for name, f in futures.items()}
    record = {
        "commit": git_commit(),
        "machine": machine_info(),
        "libraries": library_versions(),
        "wall_clock_s": time.perf_counter() - started,
        "gap_coefficient": {"0.6": out["gap_0.6"], "0.5": out["gap_0.5"]},
        "ppo_first_settings": out["ppo_first"],
        "ppo_final_settings": out["ppo_final"],
        "neat_threshold": {"3.0": out["neat_3.0"], "1.0": out["neat_1.0"]},
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(finite(record), indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {Path(OUT)}")


if __name__ == "__main__":
    main()
