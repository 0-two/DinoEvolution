"""Command line for running a suite.

    uv run python -m sim.experiments.run --suite quick
    uv run python -m sim.experiments.run --suite main --workers 8
    uv run python -m sim.experiments.run --suite main --only ga --force

Jobs are one algorithm on one seed and run in separate processes. A job whose result file
already exists is skipped, so an interrupted suite resumes where it stopped; ``--force`` reruns
everything selected. Results land in ``results/<suite>/``. When all jobs are done the summary
table is written and printed.
"""

from __future__ import annotations

import argparse
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

from sim.experiments.aggregate import format_table, write_summary
from sim.experiments.config import load_suite
from sim.experiments.runner import Job, execute

# PPO is by far the slowest per step. Starting it first keeps the last worker from being the
# one that holds everything up.
SLOWEST_FIRST = ("ppo", "neat", "cmaes", "ga", "rule_scaled", "rule_fixed")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--suite", required=True, help="suite name or path to a YAML file")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--only", action="append", help="run only these run ids (repeatable)")
    ap.add_argument("--force", action="store_true", help="rerun jobs that already have results")
    args = ap.parse_args(argv)

    suite = load_suite(args.suite)
    jobs = [
        Job(suite.name, run, seed, suite.test_max_ticks, suite.test_n_seeds)
        for run in suite.runs
        if not args.only or run.id in args.only
        for seed in suite.seeds
    ]
    todo = [j for j in jobs if args.force or not j.run_path.exists()]
    todo.sort(key=lambda j: SLOWEST_FIRST.index(j.run.algorithm))
    print(
        f"suite {suite.name}: {len(jobs)} jobs, {len(jobs) - len(todo)} already done, "
        f"{len(todo)} to run on {args.workers} workers"
    )

    started = time.perf_counter()
    if todo:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(execute, job): job for job in todo}
            for n, future in enumerate(as_completed(futures), 1):
                job = futures[future]
                try:
                    r = future.result()
                except Exception as err:  # report the failure, keep the other jobs going
                    print(f"[{n}/{len(todo)}] {job.run.id} seed {job.seed} FAILED: {err!r}")
                    continue
                print(
                    f"[{n}/{len(todo)}] {r['run_id']:16s} seed {r['seed']:2d}  "
                    f"{r['env_steps'] / 1e6:6.2f}M steps  {r['wall_clock_s']:6.0f}s  "
                    f"test mean {r['test_mean']:7.0f}"
                )
    rows = write_summary(suite.name)
    print(f"\nfinished in {time.perf_counter() - started:.0f}s\n")
    print(format_table(rows))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
