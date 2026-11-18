"""Audit the numbers in the README and the report against results/.

Two checks, both run by ``python -m sim.tools.audit_numbers``:

1. Hand-typed numbers. The templates should carry results only through placeholders. This lists
   every numeric token in the template prose (outside placeholders, code blocks and tables of
   placeholders) so each can be classified as a design constant or flagged as a result.
2. Independent recomputation. The headline table, the comparison table and the ablation table are
   recomputed here from the raw run files, with code that shares nothing with the figure and
   document builders except the Student-t critical values, and compared with the numbers in the
   committed README and report.

Exit status is 1 if any recomputed number disagrees with the documents.
"""

from __future__ import annotations

import json
import math
import re
import statistics
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
NUMBER = re.compile(r"(?<![\w.{])(\d[\d,]*\.?\d*)(?![\w}])")

# Two-sided 95% t critical values, copied from standard tables (not imported from sim.stats).
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262}


def template_numbers() -> list[tuple[str, int, str, str]]:
    """(file, line, token, line text) for numbers typed into template prose."""
    found = []
    for name in ("README.template.md", "REPORT.template.md"):
        in_code = False
        for i, line in enumerate(
            (REPO / "report" / "templates" / name).read_text(encoding="utf-8").splitlines(), 1
        ):
            if line.startswith("```"):
                in_code = not in_code
                continue
            if in_code or line.startswith(
                ("#", "|", "![", "- Stanley", "- Salimans", "- Hansen", "- Schulman", "- Raffin")
            ):
                continue
            stripped = re.sub(r"\{\{[^}]*\}\}", " ", line)
            stripped = re.sub(r"`[^`]*`", " ", stripped)
            for m in NUMBER.finditer(stripped):
                found.append((name, i, m.group(1), line.strip()))
    return found


def load_runs(suite: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = {}
    for p in sorted((REPO / "results" / suite / "runs").glob("*/seed_*.json")):
        r = json.loads(p.read_text(encoding="utf-8"))
        out.setdefault(r["run_id"], []).append(r)
    return out


def ci_half(values: list[float]) -> float:
    n = len(values)
    return T975[n - 1] * statistics.stdev(values) / math.sqrt(n)


def first_hit(run: dict, threshold: float) -> int | None:
    for row in run["history"]:
        if row["best"] >= threshold:
            return row["env_steps"]
    return None


def check_headline(failures: list[str]) -> int:
    runs = load_runs("main")
    readme = (REPO / "README.md").read_text(encoding="utf-8")
    report = (REPO / "report" / "REPORT.md").read_text(encoding="utf-8")
    checked = 0
    for rid, rs in runs.items():
        means = [statistics.fmean(r["test"]["scores"]) for r in rs]
        # Also verify the stored per-run mean matches the mean of the stored per-seed scores.
        for r, m in zip(rs, means, strict=True):
            if abs(r["test"]["mean"] - m) > 1e-9:
                failures.append(
                    f"{rid} seed {r['seed']}: stored test mean differs from mean of its scores"
                )
        mean = statistics.fmean(means)
        half = ci_half(means)
        best = max(means)
        hits = [h for r in rs if (h := first_hit(r, 1000.0)) is not None]
        wall = statistics.fmean(r["wall_clock_s"] for r in rs)
        cells = [f"{best:,.0f}", f"{mean:,.0f} ± {half:,.0f}", f"{wall:,.0f}"]
        if hits and not rid.startswith("rule"):
            cells.append(f"{statistics.median(hits) / 1e6:.2f} M ({len(hits)} of {len(rs)} runs)")
        for text, name in ((readme, "README"), (report, "REPORT")):
            row = next(
                (ln for ln in _headline_rows(text) if ln.split("|")[1].strip() == LABELS[rid]),
                None,
            )
            if row is None:
                failures.append(f"{name}: no headline row for {rid}")
                continue
            for c in cells:
                checked += 1
                if c not in row:
                    failures.append(f"{name} row {rid}: expected {c!r} in {row!r}")
    return checked


def _headline_rows(text: str) -> list[str]:
    """Rows of the table whose header starts with Algorithm and Best run."""
    lines = text.splitlines()
    start = next(i for i, ln in enumerate(lines) if ln.startswith("| Algorithm | Best run"))
    rows = []
    for ln in lines[start + 2 :]:
        if not ln.startswith("|"):
            break
        rows.append(ln)
    return rows


LABELS = {
    "ga": "GA",
    "cmaes": "CMA-ES",
    "neat": "NEAT",
    "ppo": "PPO",
    "rule_fixed": "rule, fixed",
    "rule_scaled": "rule, scaled",
}


def check_ablations(failures: list[str]) -> int:
    runs = load_runs("ablations")
    report = (REPO / "report" / "REPORT.md").read_text(encoding="utf-8")
    checked = 0
    for rid, rs in runs.items():
        means = [statistics.fmean(r["test"]["scores"]) for r in rs]
        cell = f"{statistics.fmean(means):,.0f} ± {ci_half(means):,.0f}"
        checked += 1
        if cell not in report:
            failures.append(f"REPORT ablation {rid}: expected {cell!r}")
    return checked


def check_run_integrity(failures: list[str]) -> int:
    """Every run file: monotone steps, finite scores, genome file present, budget respected."""
    checked = 0
    for suite in ("main", "ablations"):
        for rid, rs in load_runs(suite).items():
            for r in rs:
                checked += 1
                steps = [row["env_steps"] for row in r["history"]]
                if steps != sorted(steps):
                    failures.append(f"{suite}/{rid}/{r['seed']}: env_steps not monotone")
                if r["env_steps"] < r["budget_steps"] and not rid.startswith("rule"):
                    failures.append(f"{suite}/{rid}/{r['seed']}: stopped before its budget")
                if not (REPO / r["genome_file"]).exists():
                    failures.append(f"{suite}/{rid}/{r['seed']}: genome file missing")
                if len(r["test"]["scores"]) != r["test"]["seeds"][2]:
                    failures.append(f"{suite}/{rid}/{r['seed']}: wrong number of test scores")
                if not all(math.isfinite(s) and s >= 0 for s in r["test"]["scores"]):
                    failures.append(f"{suite}/{rid}/{r['seed']}: bad test score")
    return checked


def check_permutation_p(failures: list[str]) -> int:
    """Recompute some permutation p-values with numpy and ten times the resamples."""
    import numpy as np

    main_runs, abl_runs = load_runs("main"), load_runs("ablations")
    pairs = [
        ("NEAT against GA", main_runs["neat"], main_runs["ga"]),
        ("NEAT against PPO", main_runs["neat"], main_runs["ppo"]),
        ("GA against PPO", main_runs["ga"], main_runs["ppo"]),
        ("GA noelite against default", abl_runs["ga-noelite"], abl_runs["ga-default"]),
        ("GA k1 against default", abl_runs["ga-k1"], abl_runs["ga-default"]),
    ]
    report = (REPO / "report" / "REPORT.md").read_text(encoding="utf-8")
    rng = np.random.default_rng(12345)
    checked = 0
    for label, ra, rb in pairs:
        a = np.array([statistics.fmean(r["test"]["scores"]) for r in ra])
        b = np.array([statistics.fmean(r["test"]["scores"]) for r in rb])
        pooled = np.concatenate([a, b])
        observed = abs(a.mean() - b.mean())
        n = 200_000
        idx = np.argsort(rng.random((n, len(pooled))), axis=1)
        shuffled = pooled[idx]
        diff = np.abs(shuffled[:, : len(a)].mean(axis=1) - shuffled[:, len(a) :].mean(axis=1))
        p = (1 + np.sum(diff >= observed - 1e-9)) / (n + 1)
        row = next((ln for ln in report.splitlines() if ln.startswith(f"| {label} |")), None)
        checked += 1
        if row is None:
            failures.append(f"no comparison row for {label}")
            continue
        text = row.strip("| ").split("|")[-1].strip()
        reported = 0.0005 if text.startswith("<") else float(text)
        if abs(reported - p) > 0.01 and not (text.startswith("<") and p < 0.002):
            failures.append(f"{label}: report p {text} but independent estimate {p:.4f}")
    return checked


def check_generalisation_sample(failures: list[str]) -> int:
    """Replay one genome per algorithm at the training cap and compare with generalisation.json."""
    sys.path.insert(0, str(REPO))
    from sim.game import DinoGame
    from sim.genome_io import load_game_policy
    from sim.seeds import TEST_SEEDS

    rows = json.loads(
        (REPO / "results" / "main" / "generalisation.json").read_text(encoding="utf-8")
    )
    runs = load_runs("main")
    checked = 0
    for rid in ("ga", "cmaes", "neat", "ppo"):
        row = next(r for r in rows if r["run_id"] == rid and r["seed"] == 3)
        run = next(r for r in runs[rid] if r["seed"] == 3)
        act = load_game_policy(json.loads((REPO / run["genome_file"]).read_text(encoding="utf-8")))
        scores = []
        for seed in TEST_SEEDS:
            g = DinoGame(seed, run["eval"]["max_ticks"])
            while g.step(act(g)):
                pass
            scores.append(g.score)
        checked += 1
        if abs(statistics.fmean(scores) - row["test_at_train_cap"]) > 1e-9:
            failures.append(
                f"{rid} seed 3: replay {statistics.fmean(scores):.3f} "
                f"vs stored {row['test_at_train_cap']:.3f}"
            )
        if abs(row["train_fitness"] - run["best_train_fitness"]) > 1e-9:
            failures.append(f"{rid} seed 3: train fitness differs from the run file")
    return checked


def main() -> int:
    typed = template_numbers()
    print(f"numbers typed into template prose: {len(typed)}")
    for name, line, token, text in typed:
        print(f"  {name}:{line}  {token:>8s}  {text[:110]}")
    failures: list[str] = []
    n1 = check_headline(failures)
    n2 = check_ablations(failures)
    n3 = check_run_integrity(failures)
    n4 = check_permutation_p(failures)
    n5 = check_generalisation_sample(failures)
    print(
        f"\nrecomputed and compared: {n1} headline cells, {n2} ablation cells, {n3} run files, "
        f"{n4} permutation p-values, {n5} generalisation replays"
    )
    for f in failures:
        print("FAIL", f)
    print("all recomputed numbers agree" if not failures else f"{len(failures)} disagreement(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
