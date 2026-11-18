"""Write README.md and report/REPORT.md from templates, with every number taken from results/.

    uv run python -m sim.figures.docs

The templates (``report/templates/*.md``) contain prose and placeholders of the form
``{{name}}``. Each placeholder is replaced by a string computed here from the run files, the
generated tables, the benchmark files or the code's own parameter tables. A placeholder that has
no value is an error, so the documents cannot quietly keep a stale or invented figure. Where a
result has not been produced yet (for example the notes on what did not work) the value is the
text ``TBD``, as the project brief asks.

Placeholder families:

``{{ga.mean}}``, ``{{neat.reached}}``       per-algorithm results (algorithms: ga, cmaes, neat, ppo,
                                            rule_fixed, rule_scaled)
``{{abl.ga_noelite.mean}}``                 GA ablation runs (run id with dashes as underscores)
``{{neat.hidden_final}}``, ``{{dm.ga.jump13}}``   analysis values
``{{setup.*}}``, ``{{bench.*}}``, ``{{web.*}}``   configuration, speed and performance
``{{de.*}}``                                the notes on settings that did not work
``{{table:headline}}``                      a Markdown table
"""

from __future__ import annotations

import configparser
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from sim import game
from sim.algorithms import cmaes, ga, neat_runner, ppo_baseline, rule_based
from sim.analysis import curves, probe
from sim.experiments.aggregate import load_runs, summarise
from sim.experiments.config import load_suite
from sim.figures.build import ALGO_LABEL, ablation_table, group, md_table
from sim.provenance import library_versions
from sim.seeds import TEST_SEEDS, TRAIN_SEED_LIMIT, TUNE_SEEDS
from sim.stats import mean_ci95, permutation_p

REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results"
REPORT = REPO / "report"
TBD = "TBD"
ALGOS = ["ga", "cmaes", "neat", "ppo", "rule_fixed", "rule_scaled"]
LEARNED = ["ga", "cmaes", "neat", "ppo"]


# --- formatting --------------------------------------------------------------------------------
def n0(x: float) -> str:
    return f"{x:,.0f}"


def millions(x: float, digits: int = 1) -> str:
    return f"{x / 1e6:.{digits}f} million"


def pm(mean: float, half: float) -> str:
    return n0(mean) if math.isnan(half) else f"{n0(mean)} ± {n0(half)}"


def read_json(path: Path) -> Any | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def fp(p: float) -> str:
    """A p-value as text. Small values are bounded, since the test resolves only to 1 in n+1."""
    return "< 0.001" if p < 0.001 else f"{p:.3f}"


def fpo(p: float) -> str:
    """The same with its operator, for prose: ``= 0.036`` or ``< 0.001``."""
    return fp(p) if p < 0.001 else f"= {p:.3f}"


def interval_relation(a: list[float], b: list[float]) -> str:
    """Whether the 95% intervals of the means of ``a`` and ``b`` overlap."""
    ma, ha = mean_ci95(a)
    mb, hb = mean_ci95(b)
    return "overlap" if abs(ma - mb) <= ha + hb else "do not overlap"


# --- the context -------------------------------------------------------------------------------
def build_values() -> dict[str, str]:
    v: dict[str, str] = {}
    main = load_runs("main")
    abl = load_runs("ablations")
    if not main:
        raise SystemExit("results/main is empty: run `make experiments` first")
    g_main = group(main)
    summary = {r["run_id"]: r for r in summarise(main)}
    suite = load_suite("main")
    tables = read_json(REPORT / "tables" / "headline.json") or []
    headline = {r["run_id"]: r for r in tables}
    gen_rows = read_json(RESULTS / "main" / "generalisation.json") or []

    # Setup
    first = suite.runs[0]
    evo_budget = max(r.budget_steps for r in suite.runs if r.algorithm in ("ga", "cmaes", "neat"))
    ppo_budget = next(r.budget_steps for r in suite.runs if r.algorithm == "ppo")
    ceiling_train = _ceiling(first.eval.max_ticks)
    ceiling_full = _ceiling(suite.test_max_ticks)
    v.update(
        {
            "setup.n_runs": str(len(suite.seeds)),
            "setup.n_test": str(suite.test_n_seeds),
            "setup.train_cap": n0(first.eval.max_ticks),
            "setup.test_cap": n0(suite.test_max_ticks),
            "setup.ceiling_train": n0(ceiling_train),
            "setup.ceiling_full": n0(ceiling_full),
            "setup.budget_evo": millions(evo_budget, 0),
            "setup.budget_ppo": millions(ppo_budget, 0),
            "setup.k": str(first.eval.k_seeds),
            "setup.train_seeds": f"0 to {n0(TRAIN_SEED_LIMIT - 1)}",
            "setup.tune_seeds": f"{n0(TUNE_SEEDS[0])} to {n0(TUNE_SEEDS[-1])}",
            "setup.test_seeds": f"{n0(TEST_SEEDS[0])} to {n0(TEST_SEEDS[-1])}",
            "setup.ticks_per_s": str(game.TICKS_PER_SECOND),
            "setup.cpu_hours": f"{sum(r['wall_clock_s'] for r in main + abl) / 3600:.1f}",
            "setup.n_jobs": str(len(main) + len(abl)),
        }
    )
    v["setup.steps_per_run_note"] = f"{v['setup.budget_evo']} environment steps"

    # Per algorithm
    for a in ALGOS:
        s, h = summary[a], headline.get(a, {})
        v[f"label.{a}"] = ALGO_LABEL[a]
        v[f"{a}.mean"] = n0(s["test_mean"])
        v[f"{a}.ci"] = n0(s["test_ci95"]) if not math.isnan(s["test_ci95"]) else TBD
        v[f"{a}.pm"] = pm(s["test_mean"], s["test_ci95"])
        v[f"{a}.best"] = n0(s["best_run"])
        v[f"{a}.reached"] = f"{s['reached_1000']} of {s['n_runs']}"
        v[f"{a}.steps1000"] = (
            millions(s["steps_to_1000"], 1) if s["steps_to_1000"] else "not reached"
        )
        v[f"{a}.gens1000"] = (
            f"{h['generations_to_1000']:.0f}" if h.get("generations_to_1000") else "-"
        )
        v[f"{a}.wall"] = n0(s["wall_s"])
        v[f"{a}.cpu_min"] = f"{s['wall_s'] / 60:.1f}"
    v["learned.best_mean_label"] = ALGO_LABEL[max(LEARNED, key=lambda a: summary[a]["test_mean"])]
    v["ceiling_runs"] = ", ".join(
        f"{ALGO_LABEL[a]} {sum(r['test']['mean'] >= ceiling_full - 1 for r in g_main[a])}"
        for a in LEARNED
    )
    for a in LEARNED:
        v[f"{a}.at_ceiling"] = str(sum(r["test"]["mean"] >= ceiling_full - 1 for r in g_main[a]))
    v["neat.vs_ga"] = interval_relation(
        [r["test"]["mean"] for r in g_main["neat"]], [r["test"]["mean"] for r in g_main["ga"]]
    )
    v["ga.vs_ppo"] = interval_relation(
        [r["test"]["mean"] for r in g_main["ga"]], [r["test"]["mean"] for r in g_main["ppo"]]
    )
    v["ga.vs_rule"] = interval_relation(
        [r["test"]["mean"] for r in g_main["ga"]],
        [r["test"]["mean"] for r in g_main["rule_scaled"]],
    )
    v["ppo.vs_rule"] = interval_relation(
        [r["test"]["mean"] for r in g_main["ppo"]],
        [r["test"]["mean"] for r in g_main["rule_scaled"]],
    )
    v["ppo.step_ratio"] = (
        f"{summary['ga']['steps_to_1000'] / summary['ppo']['steps_to_1000']:.0f}"
        if summary["ppo"]["steps_to_1000"] and summary["ga"]["steps_to_1000"]
        else TBD
    )

    # Medians, the 3,000-point milestone, and curves read at matched steps
    from sim.experiments.aggregate import steps_to_threshold

    for algo in LEARNED:
        runs = g_main[algo]
        means_ = [r["test"]["mean"] for r in runs]
        v[f"{algo}.median"] = n0(statistics.median(means_))
        v[f"{algo}.below2000"] = str(sum(1 for m in means_ if m < 2000))
        hits3 = [h for r in runs if (h := steps_to_threshold(r, 3000.0)) is not None]
        v[f"{algo}.reached3000"] = f"{len(hits3)} of {len(runs)}"
        v[f"{algo}.steps3000"] = millions(statistics.median(hits3)) if hits3 else "not reached"
        end_best = curves.curve_stats(runs, "best")
        end_mean = curves.curve_stats(runs, "mean")
        v[f"curve.{algo}.best_end"] = n0(end_best["mean"][-1])
        v[f"curve.{algo}.mean_end"] = n0(end_mean["mean"][-1])
        v[f"curve.{algo}.end_steps"] = f"{end_best['steps_m'][-1]:.0f}"
        at = [x for r in runs if (x := curves.resample(r, "best", [ppo_budget])[0]) is not None]
        v[f"curve.{algo}.best_at_ppo"] = n0(statistics.fmean(at))
    v["setup.ppo_steps_m"] = f"{ppo_budget / 1e6:.0f}"

    def _mean(algo: str, key: str) -> float:
        return statistics.fmean(r[key] for r in gen_rows if r["algorithm"] == algo)

    gaps = [
        100
        * (_mean(x, "train_fitness") - _mean(x, "test_at_train_cap"))
        / _mean(x, "train_fitness")
        for x in ("ga", "cmaes", "neat")
    ]
    v["gap.min_pct"], v["gap.max_pct"] = f"{min(gaps):.0f}", f"{max(gaps):.0f}"
    v["cmaes.pop"] = str(cmaes.DEFAULTS["pop_size"])
    v["env.clear_ticks"] = str(game.START_CLEAR_TICKS)

    # Game and GA constants quoted in prose
    pop = ga.DEFAULTS["pop_size"]
    v.update({
        "env.start_speed": f"{game.START_SPEED:g}",
        "env.max_speed": f"{game.MAX_SPEED:g}",
        "env.ramp_ticks": n0((game.MAX_SPEED - game.START_SPEED) / game.ACCEL),
        "ga.pop": str(pop),
        "ga.elites": str(ga.DEFAULTS["elite_count"]),
        "ga.tournament": str(ga.DEFAULTS["tournament_size"]),
        "ga.n_weights": "96",
        "ppo.hidden": str(ppo_baseline.DEFAULTS["hidden"]),
    })  # fmt: skip

    # Rule baselines: tuned parameters
    for a in ("rule_fixed", "rule_scaled"):
        params = {
            (json.loads((REPO / r["genome_file"]).read_text())["d"],
             json.loads((REPO / r["genome_file"]).read_text())["gain"])
            for r in g_main[a]
        }  # fmt: skip
        v[f"{a}.params"] = "; ".join(f"d {d:g}, gain {gn:g}" for d, gn in sorted(params))
        v[f"{a}.n_params"] = str(len(params))

    # Generalisation
    for a in LEARNED:
        rows = [r for r in gen_rows if r["algorithm"] == a]
        if not rows:
            continue
        train = statistics.fmean(r["train_fitness"] for r in rows)
        test = statistics.fmean(r["test_at_train_cap"] for r in rows)
        v[f"{a}.train"] = n0(train)
        v[f"{a}.testcap"] = n0(test)
        v[f"{a}.gap"] = n0(train - test)
        v[f"{a}.gap_pct"] = f"{100 * (train - test) / train:.0f}"
        v[f"{a}.capshare"] = (
            f"{100 * statistics.fmean(r['reached_cap_fraction'] for r in rows):.0f}"
        )

    # Ablations
    g_abl = group(abl)
    default = g_abl.get("ga-default", [])
    for rid, runs in g_abl.items():
        k = rid.replace("-", "_")
        means = [r["test"]["mean"] for r in runs]
        m, h = mean_ci95(means)
        hits = [
            next((row["env_steps"] for row in r["history"] if row["best"] >= 1000), None)
            for r in runs
        ]
        hits = [x for x in hits if x is not None]
        v[f"abl.{k}.mean"] = n0(m)
        v[f"abl.{k}.pm"] = pm(m, h)
        v[f"abl.{k}.reached"] = f"{len(hits)} of {len(runs)}"
        v[f"abl.{k}.steps"] = millions(statistics.median(hits)) if hits else "not reached"
        if default and rid != "ga-default":
            v[f"abl.{k}.vs_default"] = interval_relation(
                means, [r["test"]["mean"] for r in default]
            )
    best_other = max(
        (r for r in g_abl if r != "ga-default"),
        key=lambda r: statistics.fmean(x["test"]["mean"] for x in g_abl[r]),
    )
    v["abl.best_other_id"] = best_other
    v["abl.best_other_mean"] = n0(statistics.fmean(x["test"]["mean"] for x in g_abl[best_other]))

    # NEAT growth and the genomes themselves
    neat_runs = g_main["neat"]
    hid = curves.by_generation(neat_runs, "best_hidden_nodes")
    con = curves.by_generation(neat_runs, "best_connections")
    spc = curves.by_generation(neat_runs, "species")
    v.update({
        "neat.common_gens": str(len(hid["generation"])),
        "neat.hidden_final": f"{hid['mean'][-1]:.1f}",
        "neat.conn_start": f"{con['mean'][0]:.0f}",
        "neat.conn_end": f"{con['mean'][-1]:.0f}",
        "neat.species_peak": f"{max(spc['mean']):.0f}",
        "neat.species_peak_gen": str(spc["generation"][spc["mean"].index(max(spc["mean"]))]),
        "neat.species_final": f"{spc['mean'][-1]:.0f}",
    })  # fmt: skip
    drop = 1 - float(v["neat.conn_end"]) / float(v["neat.conn_start"])
    v["neat.conn_drop_pct"] = f"{100 * drop:.0f}"
    hidden, conns = [], []
    for r in neat_runs:
        p = json.loads((REPO / r["genome_file"]).read_text(encoding="utf-8"))
        hidden.append(sum(1 for n in p["nodes"] if n["id"] not in p["outputs"]))
        conns.append(sum(len(n["inputs"]) for n in p["nodes"]))
    v["neat.best_hidden_min"] = str(min(hidden))
    v["neat.best_hidden_max"] = str(max(hidden))
    v["neat.best_zero_hidden"] = str(sum(1 for h in hidden if h == 0))
    v["neat.best_conn_median"] = f"{statistics.median(conns):.0f}"
    top = max(neat_runs, key=lambda r: r["test"]["mean"])
    tp = json.loads((REPO / top["genome_file"]).read_text(encoding="utf-8"))
    v["neat.top_hidden"] = str(sum(1 for n in tp["nodes"] if n["id"] not in tp["outputs"]))
    v["neat.top_conn"] = str(sum(len(n["inputs"]) for n in tp["nodes"]))
    v["neat.top_seed"] = str(top["seed"])

    # Decision maps
    for a in [*LEARNED, "rule_scaled"]:
        best = max(g_main[a], key=lambda r: r["test"]["mean"])
        payload = json.loads((REPO / best["genome_file"]).read_text(encoding="utf-8"))
        cactus = probe.decision_map(payload, "small cactus")
        bird = probe.decision_map(payload, "mid bird")
        first = probe.first_jump_distance(cactus)

        def share(grid: list[list[int]], action: int) -> float:
            cells = [c for row in grid for c in row]
            return 100 * cells.count(action) / len(cells)

        v[f"dm.{a}.seed"] = str(best["seed"])
        v[f"dm.{a}.jump6"] = str(first[0]) if first[0] is not None else "none"
        v[f"dm.{a}.jump13"] = str(first[-1]) if first[-1] is not None else "none"
        v[f"dm.{a}.cactus_duck"] = f"{share(cactus, 2):.0f}"
        v[f"dm.{a}.bird_duck"] = f"{share(bird, 2):.0f}"
        v[f"dm.{a}.cactus_jump"] = f"{share(cactus, 1):.0f}"

    # Speed and performance
    bench = read_json(RESULTS / "benchmarks" / "sim_speed.json")
    if bench:
        r = bench["results"]
        v["bench.step"] = n0(r["step"]["ticks_per_s_median"])
        v["bench.observe"] = n0(r["step+observe"]["ticks_per_s_median"])
        v["bench.rule"] = n0(r["step+rule"]["ticks_per_s_median"])
        v["bench.target"] = n0(bench["target_ticks_per_s"])
        v["bench.python"] = bench["machine"]["python"]
        v["bench.cpu"] = bench["machine"]["processor"]
    web = read_json(RESULTS / "benchmarks" / "web_perf.json")
    if web:
        rows = {(x["population"], x["speed"]): x for x in web["results"]}
        v["web.browser"] = web["browser"]
        v["web.cpu"] = web["machine"]["cpu"].strip()
        v["web.cores"] = str(web["machine"]["cores"])
        v["web.os"] = web["machine"]["os"]
        v["web.fps300"] = f"{rows[(300, '1x')]['fps']:.1f}"
        v["web.p99_300"] = f"{rows[(300, '1x')]['frame_ms']['p99']:.1f}"
        v["web.fps300_16"] = f"{rows[(300, '16x')]['fps']:.1f}"
        v["web.rate_max"] = rows[(300, "max")]["simulation_rate"].split(" ")[0]
        v["web.errors"] = str(len(web["page_errors"]))

    # Parity fixture
    fx = read_json(REPO / "sim" / "tests" / "golden" / "parity_traces.json")
    if fx:
        v["parity.episodes"] = str(len(fx["episodes"]))
        v["parity.samples"] = n0(sum(len(e["samples"]) for e in fx["episodes"]))
        v["parity.ticks"] = n0(sum(len(e["actions"]) for e in fx["episodes"]))
        v["parity.network"] = str(len(fx["network"]))
        v["parity.policies"] = str(len(fx["policies"]))
        v["parity.rng"] = str(len(fx["rng"]))

    # Machine and libraries
    libs = library_versions()
    v.update({
        "lib.numpy": libs["numpy"], "lib.torch": libs["torch"].split("+")[0],
        "lib.sb3": libs["stable-baselines3"], "lib.neat": libs["neat-python"],
        "lib.cma": libs["cma"], "lib.gymnasium": libs["gymnasium"],
    })  # fmt: skip
    v["lib.python"] = main[0]["machine"]["python"]

    # PPO settings comparison
    sweep = read_json(RESULTS / "notes" / "ppo_sweep.json")
    sweep_keys = [
        "ppo_sweep.n",
        "ppo_sweep.steps",
        "ppo_sweep.chosen_score",
        "ppo_sweep.other_min",
        "ppo_sweep.other_max",
    ]
    if sweep:
        final = {x["name"]: x["window_mean"][-1] for x in sweep["variants"]}
        others = [score for name, score in final.items() if name != sweep["chosen"]]
        v["ppo_sweep.n"] = str(len(final))
        v["ppo_sweep.steps"] = millions(sweep["steps_per_variant"], 0)
        v["ppo_sweep.chosen_score"] = n0(final[sweep["chosen"]])
        v["ppo_sweep.other_min"], v["ppo_sweep.other_max"] = n0(min(others)), n0(max(others))
    else:
        v.update(dict.fromkeys(sweep_keys, TBD))
    v["ppo.log_every"] = n0(ppo_baseline.DEFAULTS["log_every"])
    v["rule.tune_seeds"] = str(rule_based.RUN_TUNE_SEEDS)

    # Notes on settings that did not work
    de = read_json(RESULTS / "notes" / "dead_ends.json")
    keys = ["de.gap06.mean", "de.gap06.cap", "de.gap06.d", "de.gap05.mean", "de.gap05.cap",
            "de.ppo_first.last", "de.ppo_first.first", "de.ppo_final.last", "de.neat30.max",
            "de.neat30.min", "de.neat10.max", "de.ppo_steps", "de.neat30.best", "de.neat10.best",
            "de.gap06.pct", "de.gap05.pct"]  # fmt: skip
    if de:
        g06, g05 = de["gap_coefficient"]["0.6"], de["gap_coefficient"]["0.5"]
        v["de.gap06.mean"], v["de.gap05.mean"] = n0(g06["test_mean"]), n0(g05["test_mean"])
        v["de.gap06.cap"] = f"{g06['seeds_at_cap']} of {g06['n_seeds']}"
        v["de.gap05.cap"] = f"{g05['seeds_at_cap']} of {g05['n_seeds']}"
        v["de.gap06.d"] = f"d {g06['tuned_d']:g}, gain {g06['tuned_gain']:g}"
        v["de.gap06.pct"] = f"{100 * g06['test_mean'] / g06['max_possible']:.0f}"
        v["de.gap05.pct"] = f"{100 * g05['test_mean'] / g05['max_possible']:.0f}"
        pf, pl = de["ppo_first_settings"], de["ppo_final_settings"]
        v["de.ppo_first.first"], v["de.ppo_first.last"] = (
            n0(pf["window_mean"][0]),
            n0(pf["window_mean"][-1]),
        )
        v["de.ppo_final.last"] = n0(pl["window_mean"][-1])
        v["de.ppo_steps"] = n0(pf["env_steps"][-1])
        n3, n1 = de["neat_threshold"]["3.0"], de["neat_threshold"]["1.0"]
        v["de.neat30.max"], v["de.neat30.min"] = str(n3["species_max"]), str(n3["species_min"])
        v["de.neat10.max"] = str(n1["species_max"])
        v["de.neat30.best"], v["de.neat10.best"] = n0(n3["best_fitness"]), n0(n1["best_fitness"])
    else:
        v.update(dict.fromkeys(keys, TBD))

    # Permutation tests on the per-run test means
    means = lambda rs: [r["test"]["mean"] for r in rs]  # noqa: E731
    pairs = [(a, b, g_main[a], g_main[b]) for a, b in
             [("neat", "ga"), ("neat", "ppo"), ("neat", "cmaes"), ("neat", "rule_scaled"),
              ("ga", "ppo"), ("ga", "cmaes"), ("ga", "rule_scaled"), ("ppo", "rule_scaled")]]  # fmt: skip
    rows = []
    for a, b, ra, rb in pairs:
        p = permutation_p(means(ra), means(rb))
        v[f"p.{a}_{b}"] = fpo(p)
        rows.append(
            [
                f"{ALGO_LABEL[a]} against {ALGO_LABEL[b]}",
                n0(statistics.fmean(means(ra))),
                n0(statistics.fmean(means(rb))),
                n0(statistics.fmean(means(ra)) - statistics.fmean(means(rb))),
                fp(p),
            ]
        )
    for rid in ("ga-noelite", "ga-sigma0_6", "ga-k1", "ga-k5"):
        if rid in g_abl and default:
            p = permutation_p(means(g_abl[rid]), means(default))
            v[f"p.{rid.replace('-', '_')}"] = fpo(p)
            rows.append(
                [
                    f"GA {rid.removeprefix('ga-')} against default",
                    n0(statistics.fmean(means(g_abl[rid]))),
                    n0(statistics.fmean(means(default))),
                    n0(statistics.fmean(means(g_abl[rid])) - statistics.fmean(means(default))),
                    fp(p),
                ]
            )
    v["table:comparisons"] = md_table(
        ["Comparison", "Mean A", "Mean B", "A minus B", "p"], rows, ["l", "r", "r", "r", "r"]
    ).strip()
    v["p.bonferroni"] = f"{0.05 / len(rows):.4f}"
    v["p.n_resamples"] = n0(20_000)
    v["p.n_tests"] = str(len(rows))
    v["setup.cost_full_gen"] = millions(
        ga.DEFAULTS["pop_size"] * suite.runs[0].eval.k_seeds * suite.test_max_ticks, 0
    )

    # Quick mode, if it has been run on this machine (its outputs are not kept in the repository)
    quick = load_runs("quick")
    if quick:
        cpu = sum(r["wall_clock_s"] + r["test"]["wall_clock_s"] for r in quick)
        v["quick.n_jobs"] = str(len(quick))
        v["quick.cpu_s"] = n0(cpu)
        v["quick.min_4"] = f"{cpu / 4 / 60:.1f}"
    else:
        v.update({"quick.n_jobs": TBD, "quick.cpu_s": TBD, "quick.min_4": TBD})

    v.update(tables_as_values(main, abl, suite))
    return v


def _ceiling(ticks: int) -> float:
    g = game.DinoGame(0, ticks, collisions=False)
    while not g.done:
        g.step(0)
    return g.score


# --- tables ------------------------------------------------------------------------------------
def tables_as_values(
    main: list[dict[str, Any]], abl: list[dict[str, Any]], suite: Any
) -> dict[str, str]:
    out: dict[str, str] = {}
    for name in ("headline", "ablations", "generalisation"):
        path = REPORT / "tables" / f"{name}.md"
        out[f"table:{name}"] = path.read_text(encoding="utf-8").strip() if path.exists() else TBD

    def params(defaults: dict[str, Any], notes: dict[str, str]) -> str:
        rows = [
            [
                f"`{k}`",
                f"{defaults[k]:g}" if isinstance(defaults[k], float) else str(defaults[k]),
                notes[k],
            ]
            for k in notes
        ]
        return md_table(["Parameter", "Value", "Meaning"], rows, ["l", "r", "l"]).strip()

    out["table:hyper_ga"] = params(ga.DEFAULTS, {
        "pop_size": "genomes per generation",
        "elite_count": "best genomes copied unchanged (0 turns elitism off)",
        "tournament_size": "genomes drawn per tournament; the fittest wins",
        "crossover_rate": "chance that two parents are crossed over",
        "mutation_rate": "chance that a weight is perturbed",
        "mutation_sigma": "standard deviation of the perturbation",
        "sigma_decay": "per-generation multiplier on that deviation (1 means off)",
        "init_sigma": "standard deviation of the initial weights",
    })  # fmt: skip
    out["table:hyper_cmaes"] = params(cmaes.DEFAULTS, {
        "pop_size": "samples per iteration", "sigma0": "initial step size",
    })  # fmt: skip
    out["table:hyper_ppo"] = params(ppo_baseline.DEFAULTS, {
        "n_envs": "parallel environments", "n_steps": "steps per environment per update",
        "batch_size": "minibatch size", "n_epochs": "passes over each batch",
        "learning_rate": "Adam step size", "gamma": "discount factor",
        "gae_lambda": "GAE lambda", "ent_coef": "entropy bonus",
        "hidden": "width of each of the two hidden layers", "log_every": "environment steps per logged window",
    })  # fmt: skip

    ini = configparser.ConfigParser(interpolation=None)
    ini.read(Path(neat_runner.CONFIG_PATH), encoding="utf-8")
    picks = [
        ("NEAT", "pop_size", "genomes per generation"),
        ("DefaultGenome", "initial_connection", "starting topology: every input wired to every output"),
        ("DefaultGenome", "conn_add_prob", "chance of adding a connection"),
        ("DefaultGenome", "conn_delete_prob", "chance of deleting a connection"),
        ("DefaultGenome", "node_add_prob", "chance of adding a hidden node"),
        ("DefaultGenome", "node_delete_prob", "chance of deleting a hidden node"),
        ("DefaultGenome", "weight_mutate_rate", "chance a weight is perturbed"),
        ("DefaultGenome", "weight_mutate_power", "size of that perturbation"),
        ("DefaultGenome", "bias_mutate_rate", "chance a bias is perturbed"),
        ("DefaultSpeciesSet", "compatibility_threshold", "genetic distance that separates species"),
        ("DefaultStagnation", "max_stagnation", "generations without improvement before a species is dropped"),
        ("DefaultReproduction", "elitism", "best genomes per species copied unchanged"),
        ("DefaultReproduction", "survival_threshold", "fraction of each species allowed to breed"),
    ]  # fmt: skip
    out["table:hyper_neat"] = md_table(
        ["Parameter", "Value", "Meaning"],
        [[f"`{k}`", ini[sec][k], note] for sec, k, note in picks],
        ["l", "r", "l"],
    ).strip()

    d_fixed, _ = rule_based.RUN_GRIDS["rule_fixed"]
    d_scaled, gains = rule_based.RUN_GRIDS["rule_scaled"]
    out["table:hyper_rule"] = md_table(
        ["Variant", "Searched", "Tuning seeds per run"],
        [
            ["fixed distance", f"d from {d_fixed[0]} to {d_fixed[-1]} in steps of {d_fixed.step}", str(rule_based.RUN_TUNE_SEEDS)],
            ["speed-scaled", f"d from {d_scaled[0]} to {d_scaled[-1]} in steps of {d_scaled.step}; gain {gains[0]:g} to {gains[-1]:g} in steps of {gains[1] - gains[0]:g}", str(rule_based.RUN_TUNE_SEEDS)],
        ],
        ["l", "l", "r"],
    ).strip()  # fmt: skip

    consts = [
        ("Tick rate", f"{game.TICKS_PER_SECOND} per second", "fixed timestep"),
        ("World width", f"{game.WORLD_WIDTH:g} px", "obstacles enter at the right edge"),
        ("Dino, standing", f"{game.DINO_W:g} by {game.DINO_H:g} px", "left edge fixed at x = " + f"{game.DINO_X:g}"),
        ("Dino, ducking", f"{game.DUCK_W:g} by {game.DUCK_H:g} px", "on the ground only"),
        ("Gravity", f"{game.GRAVITY:g} px per tick squared", ""),
        ("Jump speed", f"{game.JUMP_V:g} px per tick", "a full jump peaks near 92 px and lasts about 35 ticks"),
        ("Fast fall", f"gravity times {game.FAST_FALL:g}", "ducking in the air"),
        ("Hitbox margin", f"{game.DINO_INSET:g} px dino, {game.OBSTACLE_INSET:g} px obstacle", "on every side"),
        ("Speed", f"{game.START_SPEED:g} rising by {game.ACCEL:g} per tick to {game.MAX_SPEED:g}", "maximum reached after 7,000 ticks"),
        ("Small cactus", "17 by 35 px, groups of up to 3", "groups from speed 4"),
        ("Large cactus", "25 by 50 px, groups of up to 3", "groups from speed 7"),
        ("Bird", f"{game.BIRD_W:g} by {game.BIRD_H:g} px at heights {', '.join(f'{b:g}' for b in game.BIRD_BOTTOMS)}", f"from speed {game.BIRD_MIN_SPEED:g}"),
        ("Gap between obstacles", f"width x speed + {game.GAP_COEFF:g} x base gap", "then a random factor up to 1.5"),
        ("Score", f"distance x {game.SCORE_PER_PX:g}", ""),
    ]  # fmt: skip
    out["table:env"] = md_table(
        ["Quantity", "Value", "Note"], [list(c) for c in consts], ["l", "l", "l"]
    ).strip()

    budget_rows = []
    for run in suite.runs:
        budget = (
            "tuning grid" if run.algorithm.startswith("rule") else millions(run.budget_steps, 0)
        )
        budget_rows.append(
            [
                ALGO_LABEL[run.algorithm],
                budget,
                str(run.eval.k_seeds) if not run.algorithm.startswith("rule") else "-",
                n0(run.eval.max_ticks),
            ]
        )
    obs = [
        ("1", "distance to the next obstacle", f"divided by {game.OBS_DIST_NORM:g}, at most 1"),
        ("2", "width of the next obstacle", f"divided by {game.OBS_WIDTH_NORM:g}"),
        ("3", "height of the next obstacle", f"divided by {game.OBS_HEIGHT_NORM:g}"),
        ("4", "height of its bottom edge above the ground", f"divided by {game.OBS_Y_NORM:g}"),
        ("5", "distance to the obstacle after that", f"divided by {game.OBS_DIST_NORM:g}, at most 1"),
        ("6", "game speed", f"(speed - {game.START_SPEED:g}) / ({game.MAX_SPEED:g} - {game.START_SPEED:g})"),
        ("7", "dino height above the ground", f"divided by {game.OBS_DINO_Y_NORM:g}"),
        ("8", "dino vertical velocity", f"divided by {game.OBS_VY_NORM:g}"),
        ("9", "bias", "always 1"),
    ]  # fmt: skip
    out["table:obs"] = md_table(
        ["Input", "Meaning", "Scaling"], [list(o) for o in obs], ["l", "l", "l"]
    ).strip()

    seeds = [
        ["training", f"0 to {n0(TRAIN_SEED_LIMIT - 1)}", "drawn at random each generation or episode"],
        ["rule tuning", f"{n0(TUNE_SEEDS[0])} to {n0(TUNE_SEEDS[-1])}", f"{rule_based.RUN_TUNE_SEEDS} per run, offset by the run seed"],
        ["test", f"{n0(TEST_SEEDS[0])} to {n0(TEST_SEEDS[-1])}", "final genomes only; nothing is trained or tuned on them"],
    ]  # fmt: skip
    out["table:seeds"] = md_table(["Purpose", "Game seeds", "Use"], seeds, ["l", "l", "l"]).strip()

    out["table:budget"] = md_table(["Algorithm", "Budget (environment steps)", "Seeds per fitness", "Training cap (ticks)"], budget_rows, ["l", "r", "r", "r"]).strip()  # fmt: skip
    out["table:ablation_defs"] = ablation_table(abl).strip() if abl else TBD
    return out


# --- rendering ---------------------------------------------------------------------------------
PLACEHOLDER = re.compile(r"\{\{\s*([\w:.\-]+)\s*\}\}")


def render(template: str, values: dict[str, str]) -> str:
    """Replace every placeholder. Unknown names raise, so nothing is left unfilled."""
    missing: set[str] = set()

    def sub(m: re.Match[str]) -> str:
        key = m.group(1)
        if key not in values:
            missing.add(key)
            return m.group(0)
        return values[key]

    out = PLACEHOLDER.sub(sub, template)
    if missing:
        raise KeyError(f"template uses undefined placeholders: {sorted(missing)}")
    return out


def main() -> None:
    values = build_values()
    for template, target in (
        (REPORT / "templates" / "README.template.md", REPO / "README.md"),
        (REPORT / "templates" / "REPORT.template.md", REPORT / "REPORT.md"),
    ):
        text = render(template.read_text(encoding="utf-8"), values)
        target.write_text(text, encoding="utf-8", newline="\n")
        print(f"wrote {target.relative_to(REPO)} ({len(text.split())} words)")


if __name__ == "__main__":
    main()
