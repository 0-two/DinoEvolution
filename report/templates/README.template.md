# Dino neuroevolution

Small neural networks learn to play a clone of the Chrome Dino game by neuroevolution. This repository holds a deterministic simulation in Python and TypeScript, four learning methods and a hand-written baseline compared over {{setup.n_runs}} seeds each, and a browser page that evolves a population live.

![The live evolution page: a population of dinos running one course, the best dino's network, and a fitness chart that grows as generations finish.](report/media/live-evolution.gif)

Live demo: TBD. `vercel.json` configures a Vercel deployment: import the repository in Vercel, leave the root directory at the repository root, and every push is built after the web tests (including the parity test) pass. I have not deployed it yet, so there is no address to give. The site has four pages: live evolution, watch an agent, results, and how it works.

## Results

{{table:headline}}

Each row is {{setup.n_runs}} independent runs. The score is the mean over {{setup.n_test}} seeds that no method trained or was tuned on, in the full {{setup.test_cap}}-tick game, where the highest possible score is {{setup.ceiling_full}}. "Best run" is the single best run's mean. "Steps to 1,000 points" counts environment steps (one per game tick) until a generation's best fitness first reached that score. Intervals are 95% t-intervals over runs and are wide. {{learned.best_mean_label}} has the highest mean score, and a permutation test against the GA gives p {{p.neat_ga}}, which is weak evidence once {{p.n_tests}} comparisons are made. The comparisons that stand out are elitism switched off in the GA (p {{p.ga_noelite}}) and NEAT against CMA-ES (p {{p.neat_cmaes}}). The report has the full table of tests and says which ones I would not rely on.

## How it works

The game is a fixed-timestep clone of the Dino game, written from scratch. A dino stands, jumps or ducks. Cacti come in two sizes and in groups of up to three, birds fly at three heights, and the speed rises from {{env.start_speed}} to {{env.max_speed}} pixels per tick over about {{env.ramp_ticks}} ticks. Each tick the network receives nine numbers, all scaled to roughly 0 to 1: the distance to the next obstacle and its width, height and height above ground, the distance to the one after it, the game speed, the dino's height and vertical velocity, and a constant 1. It picks the action with the largest output: nothing, jump or duck. Python and TypeScript run the same rules, and {{parity.samples}} sampled game states and {{parity.network}} network test vectors check that they agree.

The genetic algorithm keeps {{ga.pop}} genomes, each a vector of {{ga.n_weights}} weights for a network with 8 tanh hidden units. A generation plays every genome on {{setup.k}} random seeds and scores it by the mean, selects parents by tournament, mixes two parents weight by weight, and perturbs some weights with Gaussian noise. The best {{ga.elites}} genomes are copied across unchanged. Averaging over seeds matters because the obstacle sequence is random: a genome that happens to meet an easy sequence would otherwise outbreed a better one. Of the GA settings I changed, elitism is the only one whose effect stands out from run-to-run variation. With it off, the GA's mean test score was {{abl.ga_noelite.mean}} against {{abl.ga_default.mean}} with it on.

![The evolutionary loop: evaluate on several seeds, select, cross over, mutate, repeat.](report/figures/loop.svg)

Three other methods train the same task. CMA-ES adapts a Gaussian over the same 96 weights from the ranking of each sample. NEAT evolves the network's topology and weights together, starting from inputs wired straight to outputs. PPO, from Stable-Baselines3, learns a {{ppo.hidden}} by {{ppo.hidden}} policy by gradient steps on the same observations. They are compared by environment steps, the number of game ticks simulated during training, so that a method that is cheap per step is not credited for it. A rule that jumps when an obstacle is within a tuned distance is the floor: the fixed-distance version scored {{rule_fixed.mean}} and a version whose distance grows with speed scored {{rule_scaled.mean}}.

## How to reproduce

You need [uv](https://docs.astral.sh/uv/) (it installs Python 3.11), Node 22 or later, and `make`. On Windows, run the `make` recipes' commands directly if `make` is missing.

```
make setup          # Python environment (CPU-only PyTorch)
make lint test      # ruff, then the Python tests (which include the parity fixture check)
make web-setup      # npm packages for the web app
make web-check      # lint, typecheck, tests (with the parity test) and a production build
make web-dev        # the web app on a local server
```

Quick mode runs every algorithm and three of the GA ablations at small budgets on 3 seeds. It is {{quick.n_jobs}} jobs and {{quick.cpu_s}} CPU-seconds on my machine, so about {{quick.min_4}} minutes on four cores. It shows that the pipeline works and gives rough numbers, not final ones.

```
make quick
```

The full experiments are the main comparison ({{setup.n_runs}} seeds per algorithm, {{setup.budget_evo}} environment steps for the evolutionary methods and {{setup.budget_ppo}} for PPO) and the GA ablations. Together they were {{setup.n_jobs}} jobs and {{setup.cpu_hours}} CPU-hours on my machine. Interrupted suites resume where they stopped.

```
make experiments    # results/main and results/ablations
make figures        # report/figures and report/tables, from results/
make docs           # this README and report/REPORT.md, from results/
```

The animation above is captured from the production build by `web/scripts/capture-frames.mjs` and assembled by `make gif`.

Every result in this README and the report, and every figure, is generated from the files in `results/`. The prose around the numbers is written by hand; the numbers in it are placeholders that `make docs` fills in.

## Repository layout

```
sim/                  Python: the game, the algorithms and the experiment pipeline
  game.py             deterministic, seeded, fixed-timestep game (the reference)
  network.py          the 9-8-3 network; genome_io.py reads and writes genome files
  algorithms/         ga.py, cmaes.py, neat_runner.py (+ neat_config.ini), ppo_baseline.py, rule_based.py
  experiments/        suites/*.yaml, the runner, aggregation, notes on what did not work
  analysis/           learning curves, decision probes, generalisation replay
  figures/            SVG figure writer, figure and table builder, README and report generator
  tests/              pytest suite, golden traces for the parity test
web/                  TypeScript, Vite, Canvas 2D, no UI framework
  src/sim/            port of game.py, checked against the same golden traces
  src/evolve/         in-browser GA, trainer and genome readers
  src/render/         population canvas, network drawing, charts, figure interactions
  src/pages/          live evolution, watch an agent, results, how it works
results/              generated run files, best genomes and benchmarks
report/               REPORT.md, templates, figures (SVG) and tables
```

## What did not work

My first version of the game used the original Dino game's gap between obstacles (coefficient 0.6). A speed-aware rule tuned on {{rule.tune_seeds}} seeds then scored {{de.gap06.mean}} on the {{setup.n_test}} test seeds, {{de.gap06.pct}}% of the highest possible score, and survived to the cap on {{de.gap06.cap}}. In my first measurement, with a shorter episode cap, that rule reached the cap on most seeds, which would have left the learning methods nothing to improve on. I changed the coefficient to 0.5 and raised the cap to {{setup.test_cap}} ticks. The same rule now scores {{de.gap05.mean}} ({{de.gap05.pct}}%) and reaches the cap on {{de.gap05.cap}}. I could not show that every seed is winnable: a search for surviving action paths pruned too many states and failed even on the original setting, so I dropped it.

PPO with my first settings (learning rate 3e-4, discount 0.995) did not learn. The mean training episode score was {{de.ppo_first.first}} in the first window and {{de.ppo_first.last}} after {{de.ppo_steps}} steps. With learning rate 1e-3 and discount 0.99, the same measurement reached {{de.ppo_final.last}}. I chose those settings after comparing {{ppo_sweep.n}} settings on one seed for {{ppo_sweep.steps}} steps each; the training score at the end was {{ppo_sweep.chosen_score}} for the one I kept and {{ppo_sweep.other_min}} to {{ppo_sweep.other_max}} for the others. That is light tuning.

NEAT's speciation threshold of 3.0 produced {{de.neat30.max}} species in every generation, so there was nothing to log or protect. A threshold of 1.0 gave up to {{de.neat10.max}}. My first NEAT export disagreed with neat-python's own network on a small share of decisions, because neat-python {{lib.neat}} treats nodes with no incoming connections as constants. I now export neat-python's own evaluation plan, and the test suite compares the exported plan with neat-python's network decision by decision on evolved genomes.

Training at the game's full {{setup.test_cap}}-tick cap would have made one generation of {{ga.pop}} competent genomes cost {{setup.cost_full_gen}} environment steps, so training episodes stop at {{setup.train_cap}} ticks. The consequence is that the best training fitness is capped at {{setup.ceiling_train}} and that final genomes have to be tested separately at the full length.

## Limitations and future work

The comparison is one game on one machine, with {{setup.n_runs}} runs per method. The intervals are wide, and most differences between the GA, PPO, CMA-ES and the speed-aware rule are within them. Only PPO had its settings compared at all, apart from NEAT's speciation threshold. The GA, CMA-ES and NEAT otherwise use first-guess parameters. The "best" curves are not the same quantity across methods (best genome on three seeds for the evolutionary methods, best episode in a window for PPO), so the report compares levels only on unseen seeds. The best genome of a run is chosen by its score on three training seeds, and it loses {{ga.gap_pct}}% of that score on unseen seeds for the GA at the training cap. There is no git history yet, so run files record `commit: unknown`.

I measured the web page in one browser: {{web.browser}}, headless, with software rendering, on {{web.cpu}}. It held {{web.fps300}} frames per second with 300 dinos and a 99th-percentile frame of {{web.p99_300}} ms. I have not run it in Firefox or on a laptop-class machine.

Next steps I would take: choose each run's final genome on a separate validation set of seeds instead of its training seeds; give every method the same tuning effort; add a noise handler to CMA-ES; run more seeds so the intervals narrow; and add an evolution strategy in the style of Salimans et al. to the comparison.

## References

- Stanley, K. O. and Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. *Evolutionary Computation* 10(2), 99-127.
- Salimans, T., Ho, J., Chen, X., Sidor, S. and Sutskever, I. (2017). Evolution strategies as a scalable alternative to reinforcement learning. arXiv:1703.03864.
- Hansen, N. (2016). The CMA evolution strategy: a tutorial. arXiv:1604.00772.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.

## Licence

MIT. See `LICENSE`.
