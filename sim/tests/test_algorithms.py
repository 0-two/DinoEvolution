"""Each algorithm at a tiny budget: it runs, respects the budget, is reproducible, and exports a
genome that plays."""

from __future__ import annotations

import numpy as np
import pytest

from sim.algorithms import cmaes, ga, neat_runner, ppo_baseline, rule_based
from sim.evaluation import EvalConfig, draw_seeds, fitness
from sim.genome_io import load_game_policy, load_policy
from sim.network import GENOME_LENGTH
from sim.seeds import TRAIN_SEED_LIMIT

CFG = EvalConfig(k_seeds=2, max_ticks=600)
BUDGET = 120_000


def check_result(result, budget: int = BUDGET) -> None:
    assert result.env_steps >= budget
    assert result.history, "at least one generation"
    steps = [h["env_steps"] for h in result.history]
    assert steps == sorted(steps)
    for h in result.history:
        assert h["best"] >= h["median"] or h["best"] >= h["mean"]
        assert h["best"] >= h["mean"] - 1e-9
    assert result.wall_clock_s > 0
    assert result.best_genome is not None
    assert result.best_genome["meta"]["seed"] == result.seed


def curve(result) -> list[tuple]:
    return [(h["generation"], h["env_steps"], h["best"], h["mean"]) for h in result.history]


class TestEvaluation:
    def test_fitness_is_the_mean_over_seeds_and_counts_ticks(self) -> None:
        idle = lambda _obs: 0  # noqa: E731
        f, steps = fitness(idle, [1, 2, 3], EvalConfig(3, 600))
        singles = [fitness(idle, [s], EvalConfig(1, 600)) for s in (1, 2, 3)]
        assert f == pytest.approx(sum(s[0] for s in singles) / 3)
        assert steps == sum(s[1] for s in singles)

    def test_jump_penalty_applies_only_when_asked(self) -> None:
        always_jump = lambda _obs: 1  # noqa: E731
        plain, _ = fitness(always_jump, [1], EvalConfig(1, 600, 0.0))
        shaped, _ = fitness(always_jump, [1], EvalConfig(1, 600, 5.0))
        assert shaped < plain

    def test_training_seeds_stay_below_the_limit(self) -> None:
        seeds = draw_seeds(np.random.default_rng(0), 5000)
        assert max(seeds) < TRAIN_SEED_LIMIT and min(seeds) >= 0


class TestGA:
    def test_next_population_keeps_elites_and_size(self) -> None:
        rng = np.random.default_rng(0)
        ranked = rng.normal(size=(20, GENOME_LENGTH))
        p = {**ga.DEFAULTS, "pop_size": 20, "elite_count": 3}
        nxt = ga.next_population(ranked, p, 0.3, rng)
        assert nxt.shape == ranked.shape
        assert np.array_equal(nxt[:3], ranked[:3])

    def test_without_mutation_or_crossover_children_are_copies(self) -> None:
        rng = np.random.default_rng(1)
        ranked = rng.normal(size=(20, GENOME_LENGTH))
        p = {**ga.DEFAULTS, "pop_size": 20, "elite_count": 0, "mutation_rate": 0.0,
             "crossover_rate": 0.0}  # fmt: skip
        nxt = ga.next_population(ranked, p, 0.3, rng)
        parents = {tuple(r) for r in ranked}
        assert all(tuple(c) in parents for c in nxt)

    def test_tournament_favours_the_fittest(self) -> None:
        rng = np.random.default_rng(2)
        ranked = np.arange(100, dtype=float)[:, None] * np.ones((1, GENOME_LENGTH))
        p = {**ga.DEFAULTS, "pop_size": 100, "elite_count": 0, "mutation_rate": 0.0,
             "crossover_rate": 0.0, "tournament_size": 5}  # fmt: skip
        nxt = ga.next_population(ranked, p, 0.0, rng)
        # Lower value = better rank. A size-5 tournament makes the mean parent index about 16.
        assert nxt[:, 0].mean() < 25

    def test_run_is_valid_and_reproducible(self) -> None:
        a = ga.run({"pop_size": 20}, CFG, BUDGET, seed=5)
        b = ga.run({"pop_size": 20}, CFG, BUDGET, seed=5)
        c = ga.run({"pop_size": 20}, CFG, BUDGET, seed=6)
        check_result(a)
        assert curve(a) == curve(b)
        assert curve(a) != curve(c)

    def test_exported_genome_scores_what_training_said(self) -> None:
        r = ga.run({"pop_size": 20}, CFG, BUDGET, seed=2)
        # Training fitness used shared seeds drawn in the best generation.
        row = max(r.history, key=lambda h: h["best"])
        score, _ = fitness(load_policy(r.best_genome), row["seeds"], CFG)
        assert score == pytest.approx(row["best"])

    def test_elitism_off_is_allowed(self) -> None:
        check_result(ga.run({"pop_size": 20, "elite_count": 0}, CFG, BUDGET, seed=1))


class TestCMAES:
    def test_run(self) -> None:
        r = cmaes.run({"pop_size": 12}, CFG, BUDGET, seed=3)
        check_result(r)
        assert "sigma" in r.history[0]


class TestNEAT:
    def test_run_logs_species_and_topology(self) -> None:
        r = neat_runner.run({"pop_size": 30}, CFG, BUDGET, seed=4)
        check_result(r)
        row = r.history[0]
        assert row["species"] >= 1
        assert sum(row["species_sizes"]) == 30
        assert "best_hidden_nodes" in row and "best_connections" in row

    def test_is_reproducible(self) -> None:
        a = neat_runner.run({"pop_size": 30}, CFG, BUDGET, seed=9)
        b = neat_runner.run({"pop_size": 30}, CFG, BUDGET, seed=9)
        assert [h["best"] for h in a.history] == [h["best"] for h in b.history]

    def test_checked_in_config_matches_the_game(self) -> None:
        config = neat_runner.load_config()
        assert config.genome_config.num_inputs == 9
        assert config.genome_config.num_outputs == 3
        assert config.genome_config.feed_forward is True


class TestPPO:
    def test_env_follows_the_gymnasium_api(self) -> None:
        pytest.importorskip("stable_baselines3")
        from stable_baselines3.common.env_checker import check_env

        from sim.env import DinoEnv

        check_env(DinoEnv(max_ticks=300), warn=True)

    def test_episode_return_equals_score(self) -> None:
        from sim.env import DinoEnv

        env = DinoEnv(max_ticks=400)
        env.reset(options={"game_seed": 11})
        total, info = 0.0, {}
        done = False
        while not done:
            _, reward, term, trunc, info = env.step(0)
            total += reward
            done = term or trunc
        assert total == pytest.approx(info["episode_score"])
        assert info["episode_score"] == pytest.approx(env.game.score)

    def test_truncation_is_not_termination(self) -> None:
        from sim.env import DinoEnv

        env = DinoEnv(max_ticks=50)
        env.reset(options={"game_seed": 3})
        for _ in range(49):
            env.step(0)
        _, _, term, trunc, _ = env.step(0)
        assert trunc and not term

    def test_reset_draws_training_seeds(self) -> None:
        from sim.env import DinoEnv

        env = DinoEnv()
        env.reset(seed=0)
        seeds = [env.reset()[1]["game_seed"] for _ in range(200)]
        assert max(seeds) < TRAIN_SEED_LIMIT

    def test_run_trains_and_exports(self) -> None:
        params = {"n_envs": 2, "n_steps": 64, "batch_size": 64, "n_epochs": 1, "log_every": 500}
        r = ppo_baseline.run(params, EvalConfig(1, 300), 3000, seed=1)
        assert r.env_steps >= 3000
        assert r.best_genome is not None
        act = load_policy(r.best_genome)
        assert act([0.5] * 9) in (0, 1, 2)


class TestRuleBased:
    def test_run_tunes_and_counts_ticks(self) -> None:
        r = rule_based.run({}, EvalConfig(1, 2000), 0, seed=0, variant="rule_fixed")
        assert r.env_steps > 0 and r.best_genome is not None
        assert r.best_genome["format"] == "dino-rule/1"
        act = load_game_policy(r.best_genome)
        assert act(__import__("sim.game", fromlist=["DinoGame"]).DinoGame(1)) in (0, 1, 2)

    def test_repeats_use_different_tuning_seeds(self) -> None:
        a = rule_based.run({}, EvalConfig(1, 800), 0, seed=0, variant="rule_fixed")
        b = rule_based.run({}, EvalConfig(1, 800), 0, seed=1, variant="rule_fixed")
        assert a.best_genome["meta"]["tune_seeds"] != b.best_genome["meta"]["tune_seeds"]
