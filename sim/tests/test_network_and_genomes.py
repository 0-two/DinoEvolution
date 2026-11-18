"""The network, and every exported genome format reproducing the policy it came from."""

from __future__ import annotations

import random

import neat
import numpy as np
import pytest

from sim.algorithms import neat_runner
from sim.algorithms.rule_based import RuleParams, rule_action
from sim.game import DinoGame
from sim.genome_io import (
    fixed_genome_payload,
    load_game_policy,
    load_policy,
    mlp_payload,
    neat_payload,
    rule_payload,
)
from sim.network import GENOME_LENGTH, HIDDEN, INPUTS, OUTPUTS, FixedNetwork


def observations(seed: int, n: int = 300) -> list[list[float]]:
    """Real observations from a game played by a jumping agent, so inputs look like training."""
    game = DinoGame(seed, 4000, collisions=False)
    out = []
    for i in range(n):
        out.append(game.observe())
        game.step(1 if i % 37 == 0 else 0)
    return out


class TestFixedNetwork:
    def test_sizes(self) -> None:
        assert GENOME_LENGTH == 96

    def test_matches_a_plain_loop(self) -> None:
        g = np.random.default_rng(1).normal(size=GENOME_LENGTH)
        net = FixedNetwork(g)
        obs = np.random.default_rng(2).uniform(0, 1, INPUTS)
        hidden = [
            np.tanh(sum(g[h * INPUTS + i] * obs[i] for i in range(INPUTS))) for h in range(HIDDEN)
        ]
        outputs = [
            sum(g[HIDDEN * INPUTS + o * HIDDEN + h] * hidden[h] for h in range(HIDDEN))
            for o in range(OUTPUTS)
        ]
        got_hidden, got_out = net.forward(obs)
        assert got_hidden == pytest.approx(hidden, abs=1e-12)
        assert got_out == pytest.approx(outputs, abs=1e-12)
        assert net.act(obs) == int(np.argmax(outputs))

    def test_bias_input_is_the_hidden_bias(self) -> None:
        g = np.zeros(GENOME_LENGTH)
        g[0 * INPUTS + 8] = 1.0  # hidden 0 reads only the bias input
        g[HIDDEN * INPUTS + 1 * HIDDEN + 0] = 2.0  # output 1 = 2 * hidden 0
        hidden, out = FixedNetwork(g).forward([0.0] * 8 + [1.0])
        assert hidden[0] == pytest.approx(np.tanh(1.0))
        assert out[1] == pytest.approx(2 * np.tanh(1.0))

    def test_ties_go_to_nothing(self) -> None:
        assert FixedNetwork(np.zeros(GENOME_LENGTH)).act([0.5] * INPUTS) == 0

    def test_rejects_a_wrong_size(self) -> None:
        with pytest.raises(ValueError):
            FixedNetwork(np.zeros(95))


class TestFixedPayload:
    def test_round_trip_gives_the_same_actions(self) -> None:
        g = np.random.default_rng(3).normal(size=GENOME_LENGTH)
        act = load_policy(fixed_genome_payload(g))
        net = FixedNetwork(g)
        assert all(act(o) == net.act(o) for o in observations(1))

    def test_rejects_a_different_topology(self) -> None:
        payload = fixed_genome_payload(np.zeros(GENOME_LENGTH))
        payload["topology"]["hidden"] = 9
        with pytest.raises(ValueError):
            load_policy(payload)

    def test_format_matches_what_the_web_app_reads(self) -> None:
        p = fixed_genome_payload(np.zeros(GENOME_LENGTH))
        assert p["format"] == "dino-genome/1"
        assert p["topology"] == {"inputs": 9, "hidden": 8, "outputs": 3, "activation": "tanh"}
        assert len(p["weights"]) == 96


class TestNeatPayload:
    def test_export_reproduces_neat_pythons_own_network(self) -> None:
        """Genomes after several rounds of structural mutation, compared decision by decision."""
        config = neat_runner.load_config({"pop_size": 60})
        rnd = random.Random(0)
        pop = neat.Population(config, seed=7)
        checked = 0
        for _ in range(6):
            for g in pop.population.values():
                g.fitness = rnd.random()
            pop.population = pop.reproduction.reproduce(
                config, pop.species, config.pop_size, pop.generation
            )
            pop.species.speciate(config, pop.population, pop.generation)
            pop.generation += 1
            for genome in list(pop.population.values())[:10]:
                net = neat.nn.FeedForwardNetwork.create(genome, config)
                act = load_policy(neat_payload(genome, config))
                for obs in observations(pop.generation, 120):
                    out = net.activate(obs)
                    assert act(obs) == out.index(max(out))
                    checked += 1
        assert checked > 5000

    def test_hidden_nodes_do_appear(self) -> None:
        config = neat_runner.load_config({"pop_size": 60})
        rnd = random.Random(1)
        pop = neat.Population(config, seed=3)
        for _ in range(12):
            for g in pop.population.values():
                g.fitness = rnd.random()
            pop.population = pop.reproduction.reproduce(
                config, pop.species, config.pop_size, pop.generation
            )
            pop.species.speciate(config, pop.population, pop.generation)
            pop.generation += 1
        sizes = [neat_runner.genome_size(g)["hidden_nodes"] for g in pop.population.values()]
        assert max(sizes) > 0, "structural mutation should add hidden nodes within 12 generations"


class TestMlpPayload:
    def test_matches_torch(self) -> None:
        torch = pytest.importorskip("torch")
        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(9, 16), torch.nn.Tanh(), torch.nn.Linear(16, 16),
            torch.nn.Tanh(), torch.nn.Linear(16, 3),
        )  # fmt: skip
        layers = [
            (m.weight.detach().numpy(), m.bias.detach().numpy())
            for m in model
            if isinstance(m, torch.nn.Linear)
        ]
        act = load_policy(mlp_payload(layers))
        for obs in observations(2, 80):
            expected = int(model(torch.tensor(obs, dtype=torch.float64).float()).argmax())
            logits = model(torch.tensor(obs, dtype=torch.float32)).detach().numpy()
            if abs(np.sort(logits)[-1] - np.sort(logits)[-2]) > 1e-4:  # skip float32 near-ties
                assert act(obs) == expected


class TestRulePayload:
    def test_matches_rule_action(self) -> None:
        params = RuleParams(60.0, 24.0)
        act = load_game_policy(rule_payload(params.d, params.gain))
        game = DinoGame(4, 5000, collisions=False)
        for _ in range(1500):
            assert act(game) == rule_action(game, params)
            game.step(act(game))

    def test_has_no_observation_policy(self) -> None:
        with pytest.raises(ValueError):
            load_policy(rule_payload(60.0, 0.0))
