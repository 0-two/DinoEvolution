"""Train score against test score, in the same units.

``best_train_fitness`` in a run file is the best genome's mean score on its three training
seeds at the 12,000-tick training cap. The test score in the run file is at the full
36,000 ticks, so the two are not comparable. This module replays every final genome on the
100 test seeds at the training cap as well, which gives a like-for-like generalisation gap.
It also records how many test seeds the genome played to the cap without dying.

    uv run python -m sim.analysis.generalisation --suite main
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

from sim.experiments.aggregate import load_runs
from sim.experiments.runner import RESULTS_ROOT, finite
from sim.game import DinoGame
from sim.genome_io import load_game_policy
from sim.seeds import TEST_SEEDS


def _replay(args: tuple[str, int, dict[str, Any], int, int]) -> dict[str, Any]:
    run_id, seed, payload, max_ticks, n_seeds = args
    act = load_game_policy(payload)
    scores, capped = [], 0
    for s in TEST_SEEDS[:n_seeds]:
        game = DinoGame(s, max_ticks)
        while game.step(act(game)):
            pass
        scores.append(game.score)
        capped += not game.dead
    return {
        "run_id": run_id,
        "seed": seed,
        "test_at_train_cap": statistics.fmean(scores),
        "reached_cap_fraction": capped / len(scores),
    }


def compute(
    suite: str, root: Path = RESULTS_ROOT, workers: int | None = None
) -> list[dict[str, Any]]:
    runs = load_runs(suite, root)
    jobs = []
    for r in runs:
        payload = json.loads((root.parent / r["genome_file"]).read_text(encoding="utf-8"))
        jobs.append(
            (r["run_id"], r["seed"], payload, r["eval"]["max_ticks"], r["test"]["seeds"][2])
        )
    with ProcessPoolExecutor(max_workers=workers or max(1, (os.cpu_count() or 2) - 1)) as pool:
        replayed = list(pool.map(_replay, jobs))
    by_key = {(x["run_id"], x["seed"]): x for x in replayed}
    rows = []
    for r in runs:
        extra = by_key[(r["run_id"], r["seed"])]
        rows.append(
            {
                "run_id": r["run_id"],
                "algorithm": r["algorithm"],
                "seed": r["seed"],
                "train_fitness": r["best_train_fitness"],
                "train_cap": r["eval"]["max_ticks"],
                "test_at_train_cap": extra["test_at_train_cap"],
                "reached_cap_fraction": extra["reached_cap_fraction"],
                "test_full": r["test"]["mean"],
                "test_full_cap": r["test"]["max_ticks"],
            }
        )
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--suite", default="main")
    args = ap.parse_args()
    rows = compute(args.suite)
    out = RESULTS_ROOT / args.suite / "generalisation.json"
    out.write_text(json.dumps(finite(rows), indent=1, allow_nan=False) + "\n", encoding="utf-8")
    print(f"wrote {out} ({len(rows)} runs)")


if __name__ == "__main__":
    main()
