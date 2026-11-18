"""Fixed-topology genetic algorithm on the 9-8-3 network.

Same operators as the browser version in ``web/src/evolve/ga.ts``: elitism, tournament
selection, uniform crossover and Gaussian mutation with optional decay. The population is a
(pop_size, 96) array and every operator is vectorised, so the cost of a run is almost entirely
the game simulation.

Parameters (``params`` in the experiment config):

``pop_size``        genomes per generation
``elite_count``     best genomes copied unchanged; 0 turns elitism off
``tournament_size`` genomes drawn per tournament; the fittest wins
``crossover_rate``  chance that two parents are crossed over, else the child clones the first
``mutation_rate``   chance that a weight is perturbed
``mutation_sigma``  standard deviation of the perturbation at generation 0
``sigma_decay``     multiplier on the standard deviation each generation (1 = off)
``init_sigma``      standard deviation of the initial weights
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from sim.evaluation import EvalConfig, Recorder, RunResult, draw_seeds, fitness
from sim.genome_io import fixed_genome_payload
from sim.network import GENOME_LENGTH, FixedNetwork

DEFAULTS: dict[str, Any] = {
    "pop_size": 150,
    "elite_count": 8,
    "tournament_size": 3,
    "crossover_rate": 0.8,
    "mutation_rate": 0.1,
    "mutation_sigma": 0.3,
    "sigma_decay": 1.0,
    "init_sigma": 0.5,
}


def next_population(
    ranked: np.ndarray, p: dict[str, Any], sigma: float, rng: np.random.Generator
) -> np.ndarray:
    """Build the next generation from ``ranked``, a population sorted best first."""
    n = p["pop_size"]
    elites = min(int(p["elite_count"]), n, len(ranked))
    children = n - elites
    out = np.empty((n, GENOME_LENGTH))
    out[:elites] = ranked[:elites]
    if children <= 0:
        return out

    # Tournament on rank: of ``k`` random picks the lowest index (fittest) wins.
    picks = rng.integers(0, len(ranked), size=(2, children, int(p["tournament_size"])))
    parent_a = ranked[picks[0].min(axis=1)]
    parent_b = ranked[picks[1].min(axis=1)]

    # Uniform crossover for the rows that cross over; the others clone parent A.
    crossed = rng.random(children) < p["crossover_rate"]
    take_b = (rng.random((children, GENOME_LENGTH)) < 0.5) & crossed[:, None]
    kids = np.where(take_b, parent_b, parent_a)

    mutate = rng.random((children, GENOME_LENGTH)) < p["mutation_rate"]
    kids = kids + mutate * rng.normal(0.0, sigma, size=kids.shape)
    out[elites:] = kids
    return out


def run(
    params: dict[str, Any],
    eval_cfg: EvalConfig,
    budget_steps: int,
    seed: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> RunResult:
    p = {**DEFAULTS, **params}
    rng = np.random.default_rng(seed)
    pop = rng.normal(0.0, p["init_sigma"], size=(p["pop_size"], GENOME_LENGTH))
    rec = Recorder("ga", seed)
    best_genome: np.ndarray | None = None
    generation = 0

    while rec.result.env_steps < budget_steps:
        seeds = draw_seeds(rng, eval_cfg.k_seeds)
        fit = np.empty(len(pop))
        for i, genome in enumerate(pop):
            fit[i], steps = fitness(FixedNetwork(genome).act, seeds, eval_cfg)
            rec.add_steps(steps)

        order = np.argsort(-fit, kind="stable")
        if fit[order[0]] > rec.result.best_fitness:
            rec.result.best_fitness = float(fit[order[0]])
            best_genome = pop[order[0]].copy()
        row = rec.generation(fit.tolist(), seeds=seeds)
        if progress:
            progress(row)

        sigma = p["mutation_sigma"] * p["sigma_decay"] ** (generation + 1)
        pop = next_population(pop[order], p, sigma, rng)
        generation += 1

    assert best_genome is not None
    rec.result.best_genome = fixed_genome_payload(
        best_genome, {"algorithm": "ga", "seed": seed, "train_fitness": rec.result.best_fitness}
    )
    return rec.finish()
