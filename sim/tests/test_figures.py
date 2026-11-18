"""Analysis helpers and figure builders, on small synthetic runs."""

from __future__ import annotations

import itertools
import json
import math
import xml.dom.minidom
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from sim.analysis import curves, probe
from sim.figures import build
from sim.figures.svgkit import axis_top, esc, nice_ceil, nice_ticks, spread_labels
from sim.genome_io import fixed_genome_payload, rule_payload
from sim.network import GENOME_LENGTH


def fake_history(steps: list[int], values: list[float]) -> list[dict[str, Any]]:
    return [
        {"generation": i + 1, "env_steps": s, "best": v, "mean": v / 2, "median": v / 2,
         "species": 2 + i % 3, "best_hidden_nodes": i // 2, "best_connections": 20 - i}
        for i, (s, v) in enumerate(zip(steps, values, strict=True))
    ]  # fmt: skip


def fake_runs(tmp_path: Path) -> list[dict[str, Any]]:
    rng = np.random.default_rng(0)
    runs = []
    for algo in ("ga", "cmaes", "neat", "ppo", "rule_scaled"):
        for seed in range(3):
            genome_path = tmp_path / f"{algo}_{seed}.json"
            if algo == "rule_scaled":
                payload = rule_payload(60.0, 24.0)
            else:
                payload = fixed_genome_payload(rng.normal(0, 0.8, GENOME_LENGTH))
            genome_path.write_text(json.dumps(payload), encoding="utf-8")
            steps = [100_000 * (i + 1) for i in range(12)]
            runs.append({
                "run_id": algo, "algorithm": algo, "seed": seed,
                "eval": {"k_seeds": 3, "max_ticks": 12000, "jump_penalty": 0.0},
                "env_steps": steps[-1], "wall_clock_s": 10.0 + seed,
                "best_train_fitness": 900.0 + 50 * seed,
                "history": fake_history(steps, [100.0 * (i + 1) * (1 + 0.1 * seed) for i in range(12)]),
                "test": {"mean": 500.0 + 100 * seed, "max_ticks": 36000, "seeds": [1, 100, 100]},
                "genome_file": str(genome_path),
            })  # fmt: skip
    return runs


class TestCurves:
    def test_resample_is_a_step_function(self) -> None:
        run = {"history": [{"env_steps": 100, "best": 1.0}, {"env_steps": 200, "best": 5.0}]}
        assert curves.resample(run, "best", [50, 100, 150, 200, 999]) == [None, 1.0, 1.0, 5.0, 5.0]

    def test_curve_stats_stops_at_the_shortest_run_and_averages(self) -> None:
        a = {"history": [{"env_steps": 1e6 * i, "best": 10.0 * i} for i in range(1, 6)]}
        b = {"history": [{"env_steps": 1e6 * i, "best": 20.0 * i} for i in range(1, 4)]}
        s = curves.curve_stats([a, b], "best", n_points=5)
        assert s["steps_m"][-1] == pytest.approx(3.0)
        assert s["mean"][-1] == pytest.approx((30.0 + 60.0) / 2)
        assert all(lo <= m <= hi for lo, m, hi in zip(s["lo"], s["mean"], s["hi"], strict=True))

    def test_single_run_has_no_band(self) -> None:
        a = {"history": [{"env_steps": 1e6 * i, "best": 7.0} for i in range(1, 4)]}
        s = curves.curve_stats([a], "best", n_points=4)
        assert s["lo"] == s["mean"] == s["hi"]

    def test_by_generation(self) -> None:
        runs = [
            {"history": [{"species": 2}, {"species": 4}]},
            {"history": [{"species": 4}, {"species": 6}, {"species": 9}]},
        ]
        s = curves.by_generation(runs, "species")
        assert s["generation"] == [1, 2] and s["mean"] == [3.0, 5.0]


class TestSvgKit:
    def test_nice_ticks_cover_the_range_with_round_steps(self) -> None:
        ticks = nice_ticks(0, 3287, 4)
        assert ticks[0] == 0 and ticks[-1] <= 3287 and len(ticks) <= 8
        steps = {round(b - a, 6) for a, b in itertools.pairwise(ticks)}
        assert len(steps) == 1

    def test_axis_top_is_tight_and_covers_the_value(self) -> None:
        assert axis_top(11_000, 4) == 15_000  # tick step 5,000
        assert axis_top(11_000, 4) < nice_ceil(11_000)
        for v in (3, 87, 1234, 99_999):
            assert axis_top(v, 4) >= v

    def test_nice_ceil(self) -> None:
        assert nice_ceil(3287) == 5000 and nice_ceil(1) == 1 and nice_ceil(0) == 1

    def test_spread_labels_keeps_the_gap_and_the_bounds(self) -> None:
        out = spread_labels([100, 101, 102, 103], 15, 0, 200)
        assert all(b - a >= 15 - 1e-9 for a, b in zip(sorted(out), sorted(out)[1:], strict=False))
        assert min(out) >= 0 and max(out) <= 200
        crowded_low = spread_labels([196, 198, 199], 15, 0, 200)
        assert max(crowded_low) <= 200

    def test_escape(self) -> None:
        assert esc('<a href="x">&') == "&lt;a href=&quot;x&quot;&gt;&amp;"


class TestProbe:
    def test_rule_policy_map_matches_its_definition(self) -> None:
        grid = probe.decision_map(rule_payload(60.0, 24.0), "small cactus")
        first = probe.first_jump_distance(grid)
        assert all(f is not None for f in first)
        # trigger = d + gain * (speed - 6); jump when distance < trigger, columns every 5 px
        for speed, f in zip(probe.SPEEDS, first, strict=True):
            trigger = 60.0 + 24.0 * (speed - 6.0)
            assert trigger - 5 <= f < trigger + 0.001

    def test_faster_speed_means_an_earlier_jump_for_the_scaled_rule(self) -> None:
        first = probe.first_jump_distance(
            probe.decision_map(rule_payload(60.0, 24.0), "small cactus")
        )
        assert first == sorted(first)

    def test_mid_bird_is_ducked_by_the_rule(self) -> None:
        grid = probe.decision_map(rule_payload(60.0, 24.0), "mid bird")
        assert {a for row in grid for a in row} == {0, 2}

    def test_network_map_has_the_right_shape_and_actions(self) -> None:
        grid = probe.decision_map(
            fixed_genome_payload(np.random.default_rng(1).normal(size=GENOME_LENGTH)),
            "small cactus",
        )
        assert len(grid) == len(probe.SPEEDS) and len(grid[0]) == len(probe.DISTANCES)
        assert {a for row in grid for a in row} <= {0, 1, 2}


class TestFigures:
    @pytest.fixture
    def runs(self, tmp_path: Path) -> list[dict[str, Any]]:
        return fake_runs(tmp_path)

    def check(self, svg: str | None) -> str:
        assert svg is not None
        xml.dom.minidom.parseString(svg)  # raises if malformed
        assert "NaN" not in svg and "Infinity" not in svg
        assert "<title" in svg and 'role="img"' in svg
        return svg

    def test_learning_curves(self, runs: list[dict[str, Any]]) -> None:
        svg = self.check(build.learning_curves(runs))
        for name in ("GA", "CMA-ES", "NEAT", "PPO"):
            assert name in svg
        assert svg.count("data-series=") >= 8

    def test_final_scores(self, runs: list[dict[str, Any]]) -> None:
        svg = self.check(build.final_scores(runs))
        assert "rule, scaled" in svg and svg.count("data-tip=") == 15

    def test_sample_efficiency(self, runs: list[dict[str, Any]]) -> None:
        svg = self.check(build.sample_efficiency(runs))
        assert "3 of 3" in svg  # every fake run reaches 1,000

    def test_generalisation(self, runs: list[dict[str, Any]]) -> None:
        rows = [{"run_id": r["run_id"], "algorithm": r["algorithm"], "seed": r["seed"],
                 "train_fitness": 1500.0, "test_at_train_cap": 900.0 + 50 * r["seed"]}
                for r in runs]  # fmt: skip
        self.check(build.generalisation(rows))

    def test_ablations(self, runs: list[dict[str, Any]]) -> None:
        abl = []
        for rid in (
            "ga-default",
            "ga-pop50",
            "ga-pop300",
            "ga-noelite",
            "ga-k1",
            "ga-k5",
            "ga-sigma0_1",
            "ga-sigma0_6",
        ):
            for r in (r for r in runs if r["run_id"] == "ga"):
                abl.append({**r, "run_id": rid})
        svg = self.check(build.ablations(abl))
        assert "Population size" in svg and "Elitism" in svg

    def test_decision_maps(self, runs: list[dict[str, Any]]) -> None:
        svg = self.check(build.decision_maps(runs, Path(".")))
        assert "small cactus" in svg and "mid bird" in svg

    def test_neat_growth(self, runs: list[dict[str, Any]]) -> None:
        svg = self.check(build.neat_growth(runs))
        assert "Hidden nodes" in svg and "Species" in svg

    def test_figures_without_data_are_skipped(self) -> None:
        assert build.learning_curves([]) is None
        assert build.ablations([]) is None


class TestTables:
    def test_headline_table_numbers_come_from_the_runs(self, tmp_path: Path) -> None:
        runs = fake_runs(tmp_path)
        table, rows = build.headline_table(runs)
        by_id = {r["run_id"]: r for r in rows}
        assert by_id["ga"]["mean"] == pytest.approx(600.0)  # mean of 500, 600, 700
        assert by_id["ga"]["best_run"] == 700.0
        assert by_id["ga"]["reached"] == 3
        assert table.count("\n") == 2 + 5  # header, rule, five algorithms
        assert "GA" in table and "rule, scaled" in table
        assert not any(math.isnan(v) for r in rows for v in (r["mean"],))

    def test_ablation_table_marks_the_default(self, tmp_path: Path) -> None:
        base = [r for r in fake_runs(tmp_path) if r["run_id"] == "ga"]
        abl = [{**r, "run_id": rid} for rid in ("ga-default", "ga-pop50") for r in base]
        table = build.ablation_table(abl)
        assert "50" in table and "150" in table and "(default)" not in table.split("\n")[2]

    def test_pm_formatting(self) -> None:
        assert build.pm(1234.4, 56.2) == "1,234 ± 56"
        assert build.pm(1234.4, float("nan")) == "1,234"
