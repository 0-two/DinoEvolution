"""CMA-ES on the same fixed 9-8-3 network as the GA, using the ``cma`` library.

CMA-ES keeps a Gaussian over the 96 weights and adapts its mean, step size and covariance from
the ranking of each sampled population (Hansen 2016). It minimises, so the objective is the
negative mean score. Every sample in an iteration plays the same K seeds; the seeds change
between iterations, which makes the objective noisy. CMA-ES only uses ranks, which tolerates
that noise reasonably well, and no noise handler is enabled here.

Parameters:

``pop_size``  samples per iteration (the library default for 96 dimensions is 17)
``sigma0``    initial step size
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import cma
import numpy as np

from sim.evaluation import EvalConfig, Recorder, RunResult, draw_seeds, fitness
from sim.genome_io import fixed_genome_payload
from sim.network import GENOME_LENGTH, FixedNetwork

DEFAULTS: dict[str, Any] = {"pop_size": 50, "sigma0": 0.5}


def run(
    params: dict[str, Any],
    eval_cfg: EvalConfig,
    budget_steps: int,
    seed: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> RunResult:
    p = {**DEFAULTS, **params}
    rng = np.random.default_rng(seed)
    es = cma.CMAEvolutionStrategy(
        np.zeros(GENOME_LENGTH),
        p["sigma0"],
        {"popsize": p["pop_size"], "seed": seed + 1, "verbose": -9, "verb_log": 0},
    )
    rec = Recorder("cmaes", seed)
    best_genome: np.ndarray | None = None

    while rec.result.env_steps < budget_steps and not es.stop():
        seeds = draw_seeds(rng, eval_cfg.k_seeds)
        candidates = es.ask()
        fit = []
        for x in candidates:
            f, steps = fitness(FixedNetwork(x).act, seeds, eval_cfg)
            fit.append(f)
            rec.add_steps(steps)
        es.tell(candidates, [-f for f in fit])

        i = int(np.argmax(fit))
        if fit[i] > rec.result.best_fitness:
            rec.result.best_fitness = float(fit[i])
            best_genome = np.asarray(candidates[i]).copy()
        row = rec.generation(fit, sigma=float(es.sigma), seeds=seeds)
        if progress:
            progress(row)

    assert best_genome is not None
    rec.result.best_genome = fixed_genome_payload(
        best_genome, {"algorithm": "cmaes", "seed": seed, "train_fitness": rec.result.best_fitness}
    )
    return rec.finish()
