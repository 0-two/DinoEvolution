"""NEAT through neat-python (Stanley & Miikkulainen 2002).

The network topology evolves along with the weights. The configuration lives in
``neat_config.ini``. ``params`` may override a few population settings:

``pop_size``                 genomes per generation (overrides the config file)
``compatibility_threshold``  speciation distance (overrides the config file)

Besides best, mean and median fitness the history records the number of species, their sizes
and the size of the best genome, so the report can show how species and topology grow.

neat-python draws its random numbers from Python's global generator, which it seeds from the
``seed`` argument of ``Population``. Run each seed in its own process, as the runner does.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import neat
import numpy as np

from sim.evaluation import EvalConfig, Recorder, RunResult, draw_seeds, fitness
from sim.genome_io import neat_payload

CONFIG_PATH = Path(__file__).with_name("neat_config.ini")


def plain_tanh(z: float) -> float:
    """neat-python only accepts Python functions as activations, not builtins like math.tanh."""
    return math.tanh(z)


def load_config(params: dict[str, Any] | None = None) -> neat.Config:
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        str(CONFIG_PATH),
    )
    # Plain tanh instead of neat-python's tanh(2.5 z); see the note in neat_config.ini.
    config.genome_config.add_activation("tanh", plain_tanh)
    params = params or {}
    if "pop_size" in params:
        config.pop_size = int(params["pop_size"])
    if "compatibility_threshold" in params:
        config.species_set_config.compatibility_threshold = float(params["compatibility_threshold"])
    return config


def genome_size(genome: Any) -> dict[str, int]:
    """Hidden nodes and enabled connections: the size of the network that actually runs."""
    enabled = [cg for cg in genome.connections.values() if cg.enabled]
    hidden = {k for k in genome.nodes if k >= 3}  # outputs are 0..2, hidden ids start at 3
    return {"hidden_nodes": len(hidden), "connections": len(enabled)}


def run(
    params: dict[str, Any],
    eval_cfg: EvalConfig,
    budget_steps: int,
    seed: int,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> RunResult:
    config = load_config(params)
    pop = neat.Population(config, seed=seed)
    rng = np.random.default_rng(seed)
    rec = Recorder("neat", seed)
    best: dict[str, Any] = {"fitness": float("-inf"), "genome": None}

    def evaluate(genomes: list[tuple[int, Any]], cfg: neat.Config) -> None:
        seeds = draw_seeds(rng, eval_cfg.k_seeds)
        for _, genome in genomes:
            net = neat.nn.FeedForwardNetwork.create(genome, cfg)

            def act(obs: list[float], net: Any = net) -> int:
                out = net.activate(obs)
                return out.index(max(out))

            genome.fitness, steps = fitness(act, seeds, eval_cfg)
            rec.add_steps(steps)
        top = max(genomes, key=lambda item: item[1].fitness)[1]
        if top.fitness > best["fitness"]:
            best["fitness"] = top.fitness
            best["genome"] = copy.deepcopy(top)
        species = pop.species.species
        row = rec.generation(
            [g.fitness for _, g in genomes],
            seeds=seeds,
            species=len(species),
            species_sizes=sorted((len(s.members) for s in species.values()), reverse=True),
            **{f"best_{k}": v for k, v in genome_size(top).items()},
        )
        if progress:
            progress(row)

    while rec.result.env_steps < budget_steps:
        try:
            pop.run(evaluate, 1)
        except (RuntimeError, neat.CompleteExtinctionException) as err:
            # neat-python can fail to apportion offspring when there are very many species.
            # Keep what was learned and say so in the result rather than losing the run.
            rec.result.extras["stopped_early"] = f"{type(err).__name__}: {err}"
            break

    rec.result.best_fitness = float(best["fitness"])
    rec.result.best_genome = neat_payload(
        best["genome"],
        config,
        {"algorithm": "neat", "seed": seed, "train_fitness": best["fitness"]},
    )
    return rec.finish()
