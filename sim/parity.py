"""Golden traces that pin the simulation down for the TypeScript port.

The Python game is the reference. This module plays it with fixed action sequences and
stores the state every few ticks. The port replays the same actions and has to land on
the same states (the test uses a 1e-6 tolerance; in practice the values are identical).

Action sequences are stored explicitly as digit strings, so the port does not need to
re-implement any policy or random generator to replay them.

Regenerate after an intentional change to the game::

    uv run python -m sim.parity write

``python -m sim.parity check`` replays the stored actions and verifies the traces.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from sim.algorithms.rule_based import RuleParams, rule_action
from sim.game import DUCK, JUMP, NOOP, DinoGame, Mulberry32
from sim.network import GENOME_LENGTH, INPUTS, FixedNetwork

FIXTURE_PATH = Path(__file__).resolve().parent / "tests" / "golden" / "parity_traces.json"
FIELDS = [
    "tick",
    "dead",
    "distance",
    "speed",
    "y",
    "vy",
    "ducking",
    "hit_ticks",
    "rng_state",
    "obstacles",
    "observation",
]
RNG_SEEDS = [0, 1, 42, 123456789, 0xFFFFFFFF]
RNG_DRAWS = 16


def _sample(game: DinoGame) -> list[Any]:
    s = game.snapshot()
    s["observation"] = game.observe()
    return [s[f] for f in FIELDS]


def replay(seed: int, collisions: bool, max_ticks: int, actions: str, every: int) -> dict[str, Any]:
    """Play ``actions`` and return sampled states plus the final outcome."""
    game = DinoGame(seed, max_ticks, collisions)
    samples = [_sample(game)]
    for i, a in enumerate(actions):
        alive = game.step(int(a))
        if (i + 1) % every == 0:
            samples.append(_sample(game))
        if not alive:
            break
    samples.append(_sample(game))
    return {
        "samples": samples,
        "ticks": game.tick,
        "death_tick": game.death_tick,
        "score": game.score,
    }


def _record_policy(seed: int, collisions: bool, max_ticks: int, pick: Any) -> str:
    """Run ``pick(game, i) -> action`` and return the action string actually played."""
    game = DinoGame(seed, max_ticks, collisions)
    out: list[str] = []
    i = 0
    while not game.done:
        a = pick(game, i)
        out.append(str(a))
        game.step(a)
        i += 1
    return "".join(out)


def _noisy(noise_seed: int) -> Any:
    """Random jumps and ducks from an independent generator: idle 70%, jump 15%, duck 15%."""
    rng = Mulberry32(noise_seed)

    def pick(_game: DinoGame, _i: int) -> int:
        r = rng.random()
        return NOOP if r < 0.7 else (JUMP if r < 0.85 else DUCK)

    return pick


def _rule(params: RuleParams, noise_seed: int | None = None, epsilon: float = 0.0) -> Any:
    """Rule-based play. With ``epsilon > 0`` a fraction of ticks use a random action
    instead, so that deaths land at varied ticks and on varied obstacles."""
    rng = Mulberry32(noise_seed or 0)

    def pick(game: DinoGame, _i: int) -> int:
        if epsilon and rng.random() < epsilon:
            return rng.randint(0, 2)
        return rule_action(game, params)

    return pick


def network_vectors() -> list[dict[str, Any]]:
    """Random genomes and inputs with the network's hidden activations, outputs and action.

    The web test recomputes them with its own loop-based forward pass. Numpy and the loop sum
    in different orders, so they may differ in the last bit; the test allows 1e-12.
    """
    rng = np.random.default_rng(2024)
    out = []
    for i in range(12):
        genome = rng.normal(0.0, 1.0 if i % 2 else 0.4, GENOME_LENGTH)
        obs = rng.uniform(-0.2, 1.1, INPUTS)
        obs[-1] = 1.0
        hidden, outputs = FixedNetwork(genome).forward(obs)
        out.append(
            {
                "genome": genome.tolist(),
                "inputs": obs.tolist(),
                "hidden": hidden.tolist(),
                "outputs": outputs.tolist(),
                "action": FixedNetwork(genome).act(obs),
            }
        )
    return out


def policy_vectors() -> list[dict[str, Any]]:
    """Exported NEAT and MLP policies with the action each takes on a set of inputs.

    The web test loads the payloads with its own readers and must pick the same actions.
    ``verify`` replays the stored payloads rather than regenerating them, so a new neat-python
    version cannot make the fixture look stale.
    """
    import random

    import neat

    from sim.algorithms import neat_runner
    from sim.genome_io import load_policy, mlp_payload, neat_payload

    rng = np.random.default_rng(77)
    inputs = rng.uniform(-0.2, 1.1, size=(60, INPUTS))
    inputs[:, -1] = 1.0
    out: list[dict[str, Any]] = []

    config = neat_runner.load_config({"pop_size": 40})
    rnd = random.Random(5)
    pop = neat.Population(config, seed=11)
    for _ in range(10):  # let structural mutations accumulate
        for g in pop.population.values():
            g.fitness = rnd.random()
        pop.population = pop.reproduction.reproduce(config, pop.species, 40, pop.generation)
        pop.species.speciate(config, pop.population, pop.generation)
        pop.generation += 1
    ranked = sorted(pop.population.values(), key=lambda g: -len(g.connections))
    for genome in ranked[:4]:
        payload = neat_payload(genome, config)
        # Keep the evolved structure but redraw every weight and bias, so the networks take a
        # mix of actions on the test inputs instead of all jumping.
        for node in payload["nodes"]:
            node["bias"] = float(rng.normal(0, 1.0))
            for link in node["inputs"]:
                link[1] = float(rng.normal(0, 1.5))
        act = load_policy(payload)
        out.append({"payload": payload, "inputs": inputs.tolist(),
                    "actions": [act(list(x)) for x in inputs]})  # fmt: skip

    for sizes in ((INPUTS, 16, 16, 3), (INPUTS, 5, 3)):
        layers = [(rng.normal(0, 0.9, (o, i)), rng.normal(0, 0.3, o))
                  for i, o in itertools.pairwise(sizes)]  # fmt: skip
        payload = mlp_payload(layers)
        act = load_policy(payload)
        out.append({"payload": payload, "inputs": inputs.tolist(),
                    "actions": [act(list(x)) for x in inputs]})  # fmt: skip
    return out


def build_fixture() -> dict[str, Any]:
    fixed = RuleParams(100.0)
    scaled = RuleParams(40.0, 12.0)
    # (name, seed, collisions, max_ticks, sample_every, action source). Episodes driven by
    # a noise-free rule also record its parameters so the port can regenerate the actions.
    policies = {"rule-fixed": fixed, "rule-scaled": scaled, "rule-ghost": fixed}
    plan = [
        ("rule-fixed", 1, True, 6000, 10, _rule(fixed)),
        ("rule-scaled", 2, True, 6000, 10, _rule(scaled)),
        ("rule-eps-a", 3, True, 6000, 5, _rule(scaled, 1001, 0.01)),
        ("rule-eps-b", 4, True, 6000, 5, _rule(scaled, 1002, 0.01)),
        ("rule-eps-c", 5, True, 6000, 5, _rule(scaled, 1003, 0.02)),
        ("rule-eps-d", 6, True, 6000, 5, _rule(scaled, 1004, 0.02)),
        ("random-early", 10, True, 6000, 5, _noisy(1005)),
        ("rule-ghost", 9, False, 9000, 25, _rule(fixed)),
        ("noisy-ghost", 7, False, 9000, 25, _noisy(2001)),
        ("noisy-ghost-2", 8, False, 9000, 25, _noisy(2002)),
    ]
    episodes = []
    for name, seed, collisions, max_ticks, every, pick in plan:
        actions = _record_policy(seed, collisions, max_ticks, pick)
        result = replay(seed, collisions, max_ticks, actions, every)
        episodes.append(
            {
                "name": name,
                "seed": seed,
                "collisions": collisions,
                "max_ticks": max_ticks,
                "sample_every": every,
                "actions": actions,
                **(
                    {"rule": {"d": policies[name].d, "gain": policies[name].gain}}
                    if name in policies
                    else {}
                ),
                **result,
            }
        )
    rng = []
    for seed in RNG_SEEDS:
        gen = Mulberry32(seed)
        rng.append({"seed": seed, "u32": [gen.next_u32() for _ in range(RNG_DRAWS)]})
    return {
        "version": 1,
        "fields": FIELDS,
        "rng": rng,
        "episodes": episodes,
        "network": network_vectors(),
        "policies": policy_vectors(),
    }


def verify(fixture: dict[str, Any]) -> list[str]:
    """Replay the stored actions with the current game and list every mismatch.

    Replaying stored actions, rather than rebuilding them from the policies, keeps this
    check about the game alone: retuning the rule-based baseline cannot break it.
    """
    errors: list[str] = []
    for entry in fixture["rng"]:
        gen = Mulberry32(entry["seed"])
        if [gen.next_u32() for _ in entry["u32"]] != entry["u32"]:
            errors.append(f"rng seed {entry['seed']}: stream differs")
    if network_vectors() != fixture["network"]:
        errors.append("network vectors differ")
    from sim.genome_io import load_policy

    for i, entry in enumerate(fixture.get("policies", [])):
        act = load_policy(entry["payload"])
        if [act(list(x)) for x in entry["inputs"]] != entry["actions"]:
            errors.append(f"policy vector {i} ({entry['payload']['format']}): actions differ")
    for ep in fixture["episodes"]:
        got = replay(
            ep["seed"], ep["collisions"], ep["max_ticks"], ep["actions"], ep["sample_every"]
        )
        errors.extend(
            f"episode {ep['name']}: {key} differs"
            for key in ("samples", "ticks", "death_tick", "score")
            if got[key] != ep[key]
        )
    return errors


def main(argv: list[str]) -> int:
    mode = argv[1] if len(argv) > 1 else "check"
    if mode == "write":
        text = json.dumps(build_fixture(), separators=(",", ":")) + "\n"
        FIXTURE_PATH.parent.mkdir(parents=True, exist_ok=True)
        FIXTURE_PATH.write_text(text)
        print(f"wrote {FIXTURE_PATH} ({len(text) / 1024:.0f} KiB)")
        return 0
    if mode == "check":
        errors = verify(json.loads(FIXTURE_PATH.read_text()))
        for e in errors:
            print(e)
        if errors:
            print("parity fixture does not match the game; run `python -m sim.parity write`")
            return 1
        print("parity fixture matches the game")
        return 0
    print("usage: python -m sim.parity [write|check]")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv))
