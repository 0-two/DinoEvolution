"""The rule-based baseline has to beat doing nothing, and the tuning has to be repeatable."""

from __future__ import annotations

import statistics

from sim.algorithms.rule_based import RuleParams, evaluate, rule_action, tune
from sim.game import DUCK, JUMP, NOOP, DinoGame

SEEDS = tuple(range(3_000_000, 3_000_010))


def _idle_score(seed: int) -> float:
    g = DinoGame(seed)
    while g.step(NOOP):
        pass
    return g.score


def test_beats_doing_nothing() -> None:
    idle = statistics.fmean(_idle_score(s) for s in SEEDS)
    rule = statistics.fmean(evaluate(RuleParams(100.0), SEEDS))
    assert rule > 5 * idle


def test_tuning_is_deterministic_and_prefers_a_sensible_distance() -> None:
    a = tune(SEEDS, range(40, 181, 20))
    b = tune(SEEDS, range(40, 181, 20))
    assert a == b
    assert 40 <= a[0].d <= 140
    assert a[1] > statistics.fmean(evaluate(RuleParams(180.0), SEEDS))


def test_action_choice_by_obstacle_type() -> None:
    def action_for(bottom: float) -> int:
        g = DinoGame(0)
        g.obstacles.append([140.0, bottom, 46.0, 30.0])
        return rule_action(g, RuleParams(100.0))

    assert action_for(0.0) == JUMP
    assert action_for(30.0) == DUCK
    assert action_for(60.0) == NOOP


def test_waits_until_obstacle_is_inside_trigger_distance() -> None:
    g = DinoGame(0)
    g.obstacles.append([400.0, 0.0, 17.0, 35.0])
    assert rule_action(g, RuleParams(100.0)) == NOOP
    g.obstacles[0][0] = 150.0
    assert rule_action(g, RuleParams(100.0)) == JUMP
