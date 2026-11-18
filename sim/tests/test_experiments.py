"""Suite configs, a full job round trip, aggregation and the statistics helper."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from sim.evaluation import EvalConfig
from sim.experiments import aggregate
from sim.experiments.config import SUITE_DIR, load_suite
from sim.experiments.runner import Job, execute, finite, score_on_test_seeds
from sim.stats import mean_ci95, permutation_p, t_critical

SUITE_YAML = """
suite: tiny
seeds: [0, 1]
budget_steps: 40000
eval: {k_seeds: 2, max_ticks: 500}
test: {max_ticks: 800, n_seeds: 4}
runs:
  - {id: ga-small, algorithm: ga, params: {pop_size: 12}}
  - {id: ga-k1, algorithm: ga, params: {pop_size: 12}, eval: {k_seeds: 1}, budget_steps: 20000}
"""


def write_suite(tmp_path: Path, text: str = SUITE_YAML) -> Path:
    path = tmp_path / "tiny.yaml"
    path.write_text(text, encoding="utf-8")
    return path


class TestConfig:
    def test_loads_and_applies_overrides(self, tmp_path: Path) -> None:
        suite = load_suite(write_suite(tmp_path))
        assert suite.name == "tiny" and suite.seeds == (0, 1)
        assert (suite.test_max_ticks, suite.test_n_seeds) == (800, 4)
        small, k1 = suite.runs
        assert small.eval == EvalConfig(k_seeds=2, max_ticks=500) and small.budget_steps == 40000
        assert k1.eval.k_seeds == 1 and k1.eval.max_ticks == 500 and k1.budget_steps == 20000

    def test_rejects_unknown_algorithm_bad_id_and_duplicates(self, tmp_path: Path) -> None:
        bad_algo = SUITE_YAML.replace(
            "algorithm: ga, params: {pop_size: 12}}\n  - {id: ga-k1",
            "algorithm: sgd}\n  - {id: ga-k1",
            1,
        )
        with pytest.raises(ValueError, match="unknown algorithm"):
            load_suite(write_suite(tmp_path, bad_algo))
        with pytest.raises(ValueError, match="letters, digits"):
            load_suite(write_suite(tmp_path, SUITE_YAML.replace("ga-small", "ga small")))
        with pytest.raises(ValueError, match="duplicate"):
            load_suite(write_suite(tmp_path, SUITE_YAML.replace("ga-k1", "ga-small")))

    def test_checked_in_suites_load(self) -> None:
        for path in sorted(SUITE_DIR.glob("*.yaml")):
            suite = load_suite(path)
            assert suite.runs and suite.seeds, path.name


class TestJob:
    def run_job(self, tmp_path: Path) -> tuple[Job, dict]:
        suite = load_suite(write_suite(tmp_path))
        job = Job(suite.name, suite.runs[0], 0, 800, 4, str(tmp_path / "results"))
        summary = execute(job)
        assert summary["run_id"] == "ga-small"
        return job, json.loads(job.run_path.read_text(encoding="utf-8"))

    def test_writes_a_complete_run_record(self, tmp_path: Path) -> None:
        _, rec = self.run_job(tmp_path)
        required = {"params", "eval", "budget_steps", "commit", "seed", "history", "wall_clock_s"}
        assert required <= rec.keys()
        assert rec["commit"] and rec["libraries"]["numpy"]
        assert rec["history"][0].keys() >= {
            "generation",
            "env_steps",
            "wall_s",
            "best",
            "mean",
            "median",
        }
        assert rec["test"]["seeds"][2] == 4 and len(rec["test"]["scores"]) == 4
        assert rec["env_steps"] >= rec["budget_steps"]

    def test_genome_file_is_loadable_and_scores_as_recorded(self, tmp_path: Path) -> None:
        job, rec = self.run_job(tmp_path)
        payload = json.loads(job.genome_path.read_text(encoding="utf-8"))
        assert payload["format"] == "dino-genome/1"
        assert score_on_test_seeds(payload, 4, 800) == rec["test"]["scores"]

    def test_files_are_strict_json(self, tmp_path: Path) -> None:
        job, _ = self.run_job(tmp_path)
        text = job.run_path.read_text(encoding="utf-8")
        for bad in ("NaN", "Infinity"):
            assert bad not in text

    def test_finite_replaces_non_finite_numbers(self) -> None:
        out = finite({"a": float("nan"), "b": [1.0, float("inf")], "c": {"d": -math.inf}})
        assert out == {"a": None, "b": [1.0, None], "c": {"d": None}}


def fake_run(run_id: str, seed: int, test_mean: float, hit_steps: int | None) -> dict:
    history = [{"generation": 1, "env_steps": 1000, "best": 10.0}]
    if hit_steps is not None:
        history.append({"generation": 2, "env_steps": hit_steps, "best": 1500.0})
    return {
        "run_id": run_id,
        "algorithm": "ga",
        "seed": seed,
        "test": {"mean": test_mean},
        "history": history,
        "env_steps": 5000,
        "wall_clock_s": 2.0,
    }


class TestAggregate:
    def test_summary_numbers(self) -> None:
        runs = [
            fake_run("a", 0, 100.0, 4000),
            fake_run("a", 1, 300.0, 6000),
            fake_run("a", 2, 200.0, None),
            fake_run("b", 0, 50.0, None),
        ]
        rows = {r["run_id"]: r for r in aggregate.summarise(runs)}
        a = rows["a"]
        assert a["n_runs"] == 3 and a["test_mean"] == pytest.approx(200.0)
        assert a["test_ci95"] == pytest.approx(4.303 * 100.0 / math.sqrt(3))
        assert a["best_run"] == 300.0
        assert a["reached_1000"] == 2 and a["steps_to_1000"] == 5000
        assert rows["b"]["steps_to_1000"] is None and math.isnan(rows["b"]["test_ci95"])

    def test_steps_to_threshold(self) -> None:
        assert aggregate.steps_to_threshold(fake_run("a", 0, 1.0, 7000)) == 7000
        assert aggregate.steps_to_threshold(fake_run("a", 0, 1.0, None)) is None

    def test_write_summary_reads_run_files(self, tmp_path: Path) -> None:
        suite = load_suite(write_suite(tmp_path))
        for seed in suite.seeds:
            execute(Job(suite.name, suite.runs[0], seed, 800, 3, str(tmp_path / "results")))
        rows = aggregate.write_summary("tiny", tmp_path / "results")
        assert rows[0]["n_runs"] == 2
        csv_text = (tmp_path / "results" / "tiny" / "summary.csv").read_text(encoding="utf-8")
        assert csv_text.splitlines()[0].startswith("run_id,algorithm,n_runs")


class TestStats:
    def test_permutation_p_separates_clear_differences_and_not_equal_samples(self) -> None:
        low, high = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [20.0, 21.0, 22.0, 23.0, 24.0, 25.0]
        assert permutation_p(low, high, n=4000) < 0.01
        same = [1.0, 5.0, 2.0, 8.0, 3.0, 9.0]
        assert permutation_p(same, list(reversed(same)), n=2000) > 0.9

    def test_permutation_p_is_reproducible_and_never_zero(self) -> None:
        a, b = [1.0, 2.0, 3.0], [50.0, 60.0, 70.0]
        assert permutation_p(a, b, n=500, seed=3) == permutation_p(a, b, n=500, seed=3)
        assert permutation_p(a, b, n=500) > 0

    def test_ci_of_two_values(self) -> None:
        mean, half = mean_ci95([1.0, 3.0])
        assert mean == 2.0 and half == pytest.approx(12.706 * 1.0)

    def test_ci_of_one_value_is_nan(self) -> None:
        assert math.isnan(mean_ci95([5.0])[1])

    def test_t_critical_known_values_and_monotone(self) -> None:
        assert t_critical(9) == 2.262 and t_critical(1000) == 1.96
        values = [t_critical(d) for d in range(1, 200)]
        assert values == sorted(values, reverse=True)
