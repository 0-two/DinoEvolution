"""The PPO settings comparison, kept as a script so its numbers are in results/.

PPO's first settings (learning rate 3e-4, discount 0.995) did not learn within 600k steps, so I
compared a few alternatives on one seed for 2 million steps each and kept the best one. This
script reruns that comparison and writes ``results/notes/ppo_sweep.json``. It is light tuning:
one seed, one budget, five settings.

    uv run python -m sim.experiments.ppo_sweep
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any

from sim.evaluation import EvalConfig
from sim.experiments.runner import RESULTS_ROOT, finite
from sim.provenance import git_commit, library_versions, machine_info

OUT = RESULTS_ROOT / "notes" / "ppo_sweep.json"
STEPS = 2_000_000
SEED = 1
TRAIN_CAP = 12_000

# name -> parameter overrides on top of sim.algorithms.ppo_baseline.DEFAULTS
VARIANTS: dict[str, dict[str, Any]] = {
    "first": {"learning_rate": 3e-4, "gamma": 0.995, "ent_coef": 0.01},
    "entropy 0.05": {"ent_coef": 0.05, "gamma": 0.99, "learning_rate": 3e-4},
    "lr 1e-3": {"learning_rate": 1e-3, "ent_coef": 0.01, "gamma": 0.99},
    "16 envs": {
        "n_envs": 16,
        "n_steps": 128,
        "batch_size": 512,
        "ent_coef": 0.02,
        "gamma": 0.99,
        "learning_rate": 3e-4,
    },
    "lr 1e-3, entropy 0.05": {
        "learning_rate": 1e-3,
        "ent_coef": 0.05,
        "n_epochs": 4,
        "gamma": 0.99,
    },
}
CHOSEN = "lr 1e-3"


def run_variant(name: str) -> dict[str, Any]:
    from sim.algorithms import ppo_baseline

    t = time.perf_counter()
    result = ppo_baseline.run(
        {**VARIANTS[name], "log_every": 250_000}, EvalConfig(3, TRAIN_CAP), STEPS, SEED
    )
    return {
        "name": name,
        "params": VARIANTS[name],
        "env_steps": [h["env_steps"] for h in result.history],
        "window_mean": [h["mean"] for h in result.history],
        "window_best": [h["best"] for h in result.history],
        "wall_clock_s": time.perf_counter() - t,
    }


def main() -> None:
    with ProcessPoolExecutor(max_workers=len(VARIANTS)) as pool:
        runs = list(pool.map(run_variant, VARIANTS))
    record = {
        "commit": git_commit(),
        "machine": machine_info(),
        "libraries": library_versions(),
        "seed": SEED,
        "steps_per_variant": STEPS,
        "chosen": CHOSEN,
        "variants": runs,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(finite(record), indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
