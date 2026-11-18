# Neuroevolution for a Dino-style game

Technical report. Every number below is generated from the files in `results/` by `make docs`. A value that has not been produced is written TBD.

## Abstract

I trained small neural networks to play a clone of the Chrome Dino game with three evolutionary methods (a genetic algorithm, CMA-ES and NEAT), a reinforcement-learning baseline (PPO) and a hand-written rule. The game is a seeded, fixed-timestep simulation with a Python reference and a TypeScript port that agree on {{parity.samples}} sampled game states. Each method ran on {{setup.n_runs}} independent seeds, and each run's best genome was played on {{setup.n_test}} unseen seeds for the full {{setup.test_cap}}-tick game, where the highest possible score is {{setup.ceiling_full}}. NEAT had the highest mean test score ({{neat.pm}}), followed by the GA ({{ga.pm}}), PPO ({{ppo.pm}}), the speed-aware rule ({{rule_scaled.pm}}) and CMA-ES ({{cmaes.pm}}); the fixed-distance rule scored {{rule_fixed.pm}}. The 95% intervals of the learned methods overlap widely, and only NEAT against CMA-ES, against the rule and, weakly, against PPO and the GA separate in permutation tests. In the GA ablations, removing elitism was the only change with a clear effect ({{abl.ga_noelite.pm}} against {{abl.ga_default.pm}}). Best genomes of the evolutionary methods lose {{gap.min_pct}}% to {{gap.max_pct}}% of their training score on unseen seeds. The best NEAT genome reaches the maximum score with {{neat.top_hidden}} hidden nodes.

## 1. Introduction

The Chrome Dino game is a small control problem with a random obstacle sequence and a speed that rises over time. It suits neuroevolution: the score is a single number, an episode is cheap to simulate, and a network with fewer than a hundred weights can play it. I wanted to compare methods under conditions that make the comparison fair, and to write down the places where it is not.

The questions were these. How many environment steps does each method need to reach a given score? How does each method's best network perform on seeds it never saw? Which GA settings matter? What do the evolved networks do, and how large do NEAT's networks grow? The contributions are a deterministic simulation with a tested port to the browser, a pipeline that records configuration, seed and results for every run, and the comparison itself, with its limits stated.

## 2. Environment

### 2.1 Mechanics

I wrote the game from scratch and drew it with rectangles; it uses no Chrome source code or sprites. Time advances in fixed ticks of 1/{{setup.ticks_per_s}} second. A dino stands, jumps or ducks. Ducking on the ground shrinks the hitbox, and ducking in the air multiplies gravity, which shortens a jump. Collision uses axis-aligned rectangles shrunk by a few pixels so that near misses survive. The score is the distance run times a constant, so surviving longer and surviving at higher speed both raise it.

{{table:env}}

The first obstacle appears after {{env.clear_ticks}} ticks. A cactus group is one rectangle, wider for larger groups. Birds fly at three heights: the lowest must be jumped, the middle must be ducked or jumped, and the highest passes over a standing dino.

Episodes stop at a tick cap. At the full {{setup.test_cap}} ticks the highest possible score is {{setup.ceiling_full}}. Training episodes stop at {{setup.train_cap}} ticks (a ceiling of {{setup.ceiling_train}}), because one generation of {{ga.pop}} genomes that all survive {{setup.test_cap}} ticks on {{setup.k}} seeds would cost {{setup.cost_full_gen}} environment steps.

### 2.2 Observations and actions

The network sees nine numbers, each scaled to roughly 0 to 1, and chooses among three actions by taking the largest output: 0 for nothing, 1 for jump, 2 for duck. A tie goes to the lowest index. If there is no next obstacle, the distances read 1 and the other obstacle inputs read 0.

{{table:obs}}

### 2.3 Determinism

Everything that affects a game is a function of its seed and the actions taken. Obstacles come from Mulberry32, a 32-bit generator that needs only integer arithmetic, so Python and JavaScript produce the same stream. The only floating-point operations are addition, subtraction, multiplication, division, comparison, floor and min, which are exactly rounded in IEEE 754 double precision in both languages. Rounding is written as floor(x + 0.5) because Python rounds half to even and JavaScript rounds half up. The order of events inside a tick is fixed and documented at the top of `sim/game.py`. Obstacles do not depend on the dino's actions, which is why the browser can show a whole population on one course.

### 2.4 Parity test

Python is the reference. `sim/parity.py` plays {{parity.episodes}} episodes with fixed action strings ({{parity.ticks}} ticks in all) and stores the state every few ticks: {{parity.samples}} samples with position, speed, distance, hit count, generator state, every obstacle and the network inputs. The episodes include several deaths on different ticks, runs that reach top speed and birds, and runs with collisions switched off so that long games pass through every speed band. The TypeScript test replays the same actions and compares every field within a tolerance of 1e-6. A second test requires the two to be identical, not just close, and they are. A separate set of {{parity.network}} random genomes and inputs checks the browser's network against numpy within 1e-12, {{parity.rng}} generator streams are compared exactly, and {{parity.policies}} exported NEAT and PPO policies must choose the same action as Python on 60 inputs each. When I changed one constant in the port as a check, the parity tests failed. The continuous-integration workflow runs both sides.

### 2.5 Speed

The headless Python game runs at {{bench.step}} ticks per second on one core ({{web.cpu}}, Python {{bench.python}}), against a target of {{bench.target}}. Building the nine network inputs every tick brings that to {{bench.observe}}, and playing with the rule-based policy gives {{bench.rule}}. The network's forward pass adds to this, so the wall-clock times in the results are the end-to-end figures. PPO is slower per environment step than the evolutionary methods ({{ppo.cpu_min}} minutes per run for {{setup.budget_ppo}} steps, against {{ga.cpu_min}} for {{setup.budget_evo}} steps with the GA).

## 3. Methods

All learning methods except PPO act through the same 9-8-3 network: nine inputs, eight tanh hidden units, three outputs, no output bias. The ninth input is a constant 1, so its first-layer weights are the hidden biases. A genome is the 96 weights in a flat vector. NEAT evolves its own graph from the same inputs and outputs.

**Fitness.** A genome's fitness is its mean score over {{setup.k}} seeds, drawn at random and shared by every genome in the generation. Averaging keeps a lucky obstacle sequence from deciding who breeds. Reward shaping is behind a flag (`jump_penalty`) and is off in every reported run.

**Genetic algorithm.** Elitism, tournament selection, uniform crossover and Gaussian mutation, as in the loop below. Parameters:

{{table:hyper_ga}}

**CMA-ES.** The `cma` library on the same 96 weights, minimising the negative mean score. It uses ranks only. Every sample in an iteration plays the same seeds, and the seeds change between iterations, so the objective is noisy; I did not enable a noise handler.

{{table:hyper_cmaes}}

**NEAT.** neat-python {{lib.neat}} with the checked-in `sim/algorithms/neat_config.ini`. Networks start with every input connected to every output and no hidden nodes, and grow through mutation. Every node uses plain tanh (neat-python's default is tanh(2.5 x)). Selected parameters:

{{table:hyper_neat}}

**PPO.** Stable-Baselines3 {{lib.sb3}} on a Gymnasium wrapper of the same game. The reward each tick is the score gained, so an episode's return is its score. The policy is a two-layer network of width {{ppo.hidden}} with tanh, and its deterministic action is the largest logit.

{{table:hyper_ppo}}

**Rule-based baseline.** Jump when the next obstacle is within a trigger distance, duck under a bird whose bottom edge clears a ducking dino, ignore a bird that passes over a standing dino, and do nothing while airborne. The fixed variant uses one distance; the scaled variant adds a term proportional to the speed above the starting speed. Each run tunes on {{rule.tune_seeds}} seeds by grid search:

{{table:hyper_rule}}

Across the {{setup.n_runs}} runs the fixed rule tuned to {{rule_fixed.params}}, and the scaled rule to {{rule_scaled.params}}. Each run tunes on its own {{rule.tune_seeds}} seeds, yet most runs land on the same grid point, which is why these scores vary little.

![The evolutionary loop.](figures/loop.svg)

## 4. Experimental setup

**Seeds.** Game seeds come from three disjoint ranges, so nothing trained or tuned on a test seed:

{{table:seeds}}

Each algorithm ran on {{setup.n_runs}} independent seeds. A run seed fixes the algorithm's own random numbers.

**Budgets.** Methods are compared by environment steps, one per game tick, counted over every episode played during training. Evaluation on test seeds is not counted. The evolutionary methods stop at the first generation boundary after their budget, so a run overshoots a little. PPO gets a smaller budget because it simulates far fewer steps per second.

{{table:budget}}

**Compute.** Runs used one core each, in parallel processes, on {{web.cpu}} ({{web.cores}} hardware threads, {{web.os}}), Python {{lib.python}}, numpy {{lib.numpy}}, PyTorch {{lib.torch}} (CPU), Stable-Baselines3 {{lib.sb3}}, neat-python {{lib.neat}}, cma {{lib.cma}} and Gymnasium {{lib.gymnasium}}. The {{setup.n_jobs}} jobs of the main suite and the ablations took {{setup.cpu_hours}} CPU-hours.

**Statistics.** Intervals are 95% Student's t intervals over runs. Differences between methods use a two-sided permutation test on the per-run test means with {{p.n_resamples}} resamples. I report {{p.n_tests}} such tests without correcting for multiplicity. With a Bonferroni threshold of 0.05 divided by {{p.n_tests}}, which is {{p.bonferroni}}, only the clearest of them would count.

## 5. Results

### 5.1 Learning curves

![Best and mean fitness against environment steps.](figures/learning_curves.svg)

The curves are training fitness. For the GA, CMA-ES and NEAT, "best" is the best genome of each generation on its {{setup.k}} seeds, and "mean" is the population's mean. For PPO, "best" is the best episode in a window of about {{ppo.log_every}} steps and "mean" is the mean episode score in the window, under the stochastic policy. These are not the same quantity, so the shapes can be compared and the levels cannot. At {{setup.ppo_steps_m}} million steps, where PPO's budget ends, the mean best fitness was {{curve.ppo.best_at_ppo}} for PPO, {{curve.neat.best_at_ppo}} for NEAT, {{curve.ga.best_at_ppo}} for the GA and {{curve.cmaes.best_at_ppo}} for CMA-ES. At {{curve.ga.end_steps}} million steps it was {{curve.neat.best_end}} for NEAT, {{curve.ga.best_end}} for the GA and {{curve.cmaes.best_end}} for CMA-ES, against a training ceiling of {{setup.ceiling_train}}. The population means stay low for every evolutionary method ({{curve.neat.mean_end}} for NEAT, {{curve.cmaes.mean_end}} for CMA-ES, {{curve.ga.mean_end}} for the GA at the end), which is expected when most of a population is mutated children.

### 5.2 Final scores on unseen seeds

![Test score of each run's best genome.](figures/final_scores.svg)

{{table:headline}}

The distributions are far from normal. The median run of each learned method scored {{neat.median}} for NEAT, {{ga.median}} for the GA, {{ppo.median}} for PPO and {{cmaes.median}} for CMA-ES. The number of runs that scored under 2,000 was {{neat.below2000}} for NEAT, {{ga.below2000}} for the GA, {{ppo.below2000}} for PPO and {{cmaes.below2000}} for CMA-ES, out of {{setup.n_runs}}. The number of runs whose mean equals the highest possible score within one point was {{neat.at_ceiling}} for NEAT, {{ga.at_ceiling}} for the GA, and {{ppo.at_ceiling}} for PPO and {{cmaes.at_ceiling}} for CMA-ES. The speed-aware rule scored {{rule_scaled.pm}} with almost no spread, because its tuned parameters hardly change between runs.

{{table:comparisons}}

NEAT's lead over CMA-ES and over the rule is clear in these tests. Its lead over the GA (p {{p.neat_ga}}) and over PPO (p {{p.neat_ppo}}) is weaker and would not survive a Bonferroni correction for {{p.n_tests}} comparisons (threshold {{p.bonferroni}}). The GA, PPO and the speed-aware rule are not separated from one another: GA against PPO gives p {{p.ga_ppo}}, and PPO's mean ({{ppo.mean}}) is within a few points of the rule's ({{rule_scaled.mean}}).

### 5.3 Sample efficiency

![Environment steps to reach 1,000 and 3,000 points.](figures/sample_efficiency.svg)

To reach a generation whose best fitness was 1,000, the median run needed {{ppo.steps1000}} environment steps for PPO ({{ppo.reached}} runs got there), {{neat.steps1000}} for NEAT ({{neat.reached}}), {{ga.steps1000}} for the GA ({{ga.reached}}) and {{cmaes.steps1000}} for CMA-ES ({{cmaes.reached}}). For 3,000 points, near the training ceiling, the counts were {{ppo.reached3000}} for PPO (median {{ppo.steps3000}}), {{neat.reached3000}} for NEAT ({{neat.steps3000}}), {{ga.reached3000}} for the GA ({{ga.steps3000}}) and {{cmaes.reached3000}} for CMA-ES ({{cmaes.steps3000}}). PPO's first score of 1,000 comes about {{ppo.step_ratio}} times sooner than the GA's. This is the least like-for-like comparison in the report, because PPO's "best" is the best single episode in a window and the others' is the best of a population on three seeds. PPO's mean episode score in a window was {{curve.ppo.mean_end}} at the end of its budget. Its final test score ({{ppo.pm}}) says more about its quality than its first crossing does.

### 5.4 Generalisation to unseen seeds

![Best training fitness against score on unseen seeds, at the training cap.](figures/generalisation.svg)

{{table:generalisation}}

Both scores in this table use the {{setup.train_cap}}-tick training cap, so they can be compared. The best genome of every evolutionary method does worse on unseen seeds than on the three seeds that selected it: by {{ga.gap}} points ({{ga.gap_pct}}%) for the GA, {{cmaes.gap}} ({{cmaes.gap_pct}}%) for CMA-ES and {{neat.gap}} ({{neat.gap_pct}}%) for NEAT. Part of that is selection: the genome with the best score on three seeds is partly the genome that was lucky on them. PPO's gap ({{ppo.gap}}) is not comparable. It has no selection step, and its training value is the best window mean under a stochastic policy, while the test plays the deterministic policy. At the training cap, {{neat.capshare}}% of the test seeds were played to the end by NEAT's best genomes, {{ga.capshare}}% by the GA's, {{ppo.capshare}}% by PPO's and {{cmaes.capshare}}% by CMA-ES's. The full-length scores in the last column are higher than the capped ones because the game continues past the training cap.

### 5.5 GA ablations

![GA test score under changes to population size, mutation size, elitism and evaluation seeds.](figures/ablations.svg)

{{table:ablations}}

Elitism off scored {{abl.ga_noelite.pm}} against {{abl.ga_default.pm}} for the default; the intervals {{abl.ga_noelite.vs_default}}, the permutation test gives p {{p.ga_noelite}}, and {{abl.ga_noelite.reached}} runs reached 1,000 points. Without elitism the best genome of a generation can be replaced by a mutated child. I did not measure how often that happens. No other change is separated from the default. A mutation size of 0.6 scored the highest mean ({{abl.ga_sigma0_6.pm}}, p {{p.ga_sigma0_6}}), five seeds per fitness {{abl.ga_k5.pm}} (p {{p.ga_k5}}), and one seed per fitness {{abl.ga_k1.pm}} (p {{p.ga_k1}}). One seed per fitness reached 1,000 points soonest ({{abl.ga_k1.steps}}, {{abl.ga_k1.reached}} runs) but finished with the lower mean. With five seeds only {{abl.ga_k5.reached}} runs reached 1,000 within the budget, probably because each fitness evaluation costs more steps; I did not isolate that. With ten runs per setting I cannot rank the other settings.

## 6. Analysis

### 6.1 What the evolved networks do

![Decision maps of each algorithm's best genome.](figures/decision_maps.svg)

The maps are produced by placing one obstacle at a set distance from a grounded dino, setting the game speed, and asking the policy for an action. The best genome of each method (the run with the highest test mean) is shown, with the speed-aware rule for reference.

The rule jumps within a distance that grows with speed: at speed {{env.start_speed}} it starts jumping at a distance of {{dm.rule_scaled.jump6}} px, at speed {{env.max_speed}} at {{dm.rule_scaled.jump13}} px. The GA's best genome (seed {{dm.ga.seed}}) starts at {{dm.ga.jump6}} px at the slowest speed and {{dm.ga.jump13}} px at the fastest, CMA-ES's (seed {{dm.cmaes.seed}}) at {{dm.cmaes.jump6}} and {{dm.cmaes.jump13}}, and NEAT's (seed {{dm.neat.seed}}) at {{dm.neat.jump6}} and {{dm.neat.jump13}}. So the evolved networks jump later than the rule at high speed and shift less with speed. They compensate in other ways. In front of a small cactus the GA's best genome ducks in {{dm.ga.cactus_duck}}% of the grid, NEAT's in {{dm.neat.cactus_duck}}%, CMA-ES's in {{dm.cmaes.cactus_duck}}% and PPO's in {{dm.ppo.cactus_duck}}%, while the rule never does. Ducking on the ground is harmless while an obstacle is far away, and the fitness never penalises it. I read this as evolution settling on ducking as a default posture from which a late jump begins; I did not test that reading. For a mid-height bird, which has to be ducked, the share of duck actions is {{dm.ga.bird_duck}}% for the GA, {{dm.neat.bird_duck}}% for NEAT, {{dm.cmaes.bird_duck}}% for CMA-ES and {{dm.ppo.bird_duck}}% for PPO.

PPO's map is different in kind. Its best policy chooses jump in {{dm.ppo.cactus_jump}}% of the cactus grid, against {{dm.ga.cactus_jump}}% for the GA, including when the obstacle is far away, and its decision boundary follows an irregular band across distance and speed. I read this as a policy that jumps frequently and relies on the world being forgiving, but I did not test that reading.

### 6.2 How NEAT's networks grew

![Hidden nodes, connections and species by generation for NEAT.](figures/neat_growth.svg)

Over the first {{neat.common_gens}} generations (the shortest run's length), the best genome's mean count of hidden nodes rose from none to {{neat.hidden_final}}, while its mean count of enabled connections fell from {{neat.conn_start}} to {{neat.conn_end}}. NEAT started from {{neat.conn_start}} direct connections (nine inputs times three outputs) and removed about {{neat.conn_drop_pct}}% of them on average. In these runs the useful change was mostly removing connections, with a small number of hidden nodes added. The number of species rose to a mean of {{neat.species_peak}} at generation {{neat.species_peak_gen}} and was {{neat.species_final}} at the end of the shared range. Among the {{setup.n_runs}} final best genomes, hidden nodes ranged from {{neat.best_hidden_min}} to {{neat.best_hidden_max}}, and {{neat.best_zero_hidden}} had none; the median connection count was {{neat.best_conn_median}}. The best genome by test score (seed {{neat.top_seed}}) has {{neat.top_hidden}} hidden nodes and {{neat.top_conn}} connections and reaches the highest possible score of {{setup.ceiling_full}}. A single layer of weights between the inputs and the actions was enough to reach the maximum score on the {{setup.n_test}} test seeds, so the hidden layer of the fixed network is not needed to solve this game. A smaller search space is one possible reason for NEAT's advantage.

## 7. Discussion and limitations

NEAT's results are the best and the most consistent: {{neat.reached}} runs reached 1,000 points and {{neat.reached3000}} reached 3,000. A smaller search space may explain some of that. Starting from {{neat.conn_start}} weights and ending with about {{neat.best_conn_median}} connections is a smaller problem than 96 free weights, and the best genomes show that the problem does not need the hidden layer. I did not test this directly (for example by running the GA on a network with no hidden layer), so it is a reading of the results, not a result.

The GA is slower and more variable, and elitism is essential to it. CMA-ES, on the same network, scored lower than the GA on average, though the difference is not separated (p {{p.ga_cmaes}}), and that is not what I expected. I used {{cmaes.pop}} samples per iteration without tuning that number or using a noise handler, and fitness here is noisy because each generation draws new seeds, so this result says little about CMA-ES in general. PPO's first generation-best score of 1,000 came after about {{ppo.steps1000}} steps, and its final test mean was close to the tuned rule's, with a wide interval.

The main limits are these. The comparison covers one game, one machine and {{setup.n_runs}} runs per method, and the intervals are wide enough that most pairwise differences cannot be separated. The tuning effort was uneven: PPO's settings were compared on one seed ({{ppo_sweep.n}} settings, {{ppo_sweep.steps}} steps each), NEAT's speciation threshold was compared on one seed, and everything else is a first guess. The "best" curves are not the same quantity across methods. Final genomes were chosen by their score on three training seeds, which inflates their training scores and lowers their test scores a little for every evolutionary method. The game is easy to saturate: {{neat.at_ceiling}} NEAT runs reach the maximum score, so differences at the top cannot show. Training at {{setup.train_cap}} ticks and testing at {{setup.test_cap}} ticks mixes two horizons. There is no git history, so run files record `commit: unknown`. The web page's frame rate was measured in one browser (headless {{web.browser}}), not Firefox.

## 8. Reproducibility checklist

- Code: this repository. No git history exists yet, so the commit hash in run files is `unknown`; TBD until the repository is initialised.
- Environment: Python from `uv.lock`, Node packages from `web/package-lock.json`. CPU-only PyTorch.
- Seeds: ranges in section 4. Each run's seed is in its run file. The tests check that GA and NEAT runs repeat exactly from a seed.
- Configuration: every run file stores its parameters, evaluation settings and budget; suites are `sim/experiments/suites/*.yaml`; NEAT uses `sim/algorithms/neat_config.ini`.
- Hardware and libraries: section 4; each run file records Python and library versions.
- Commands: `make experiments`, `make figures`, `make docs`; quick mode `make quick`.
- Data: run files in `results/main/runs` and `results/ablations/runs`, best genomes in `results/*/genomes`, benchmarks in `results/benchmarks`, notes on settings that did not work in `results/notes`.
- Parity: `python -m sim.parity check` and `make web-parity`.
- Tests: `make lint test web-check`.
- Statistics: t intervals and permutation tests in `sim/stats.py`, with {{p.n_resamples}} resamples and a fixed seed.
- Known gaps: Firefox not tested; no correction for multiple comparisons; PPO, GA, CMA-ES and NEAT not tuned to the same extent.

## 9. References

- Stanley, K. O. and Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. *Evolutionary Computation* 10(2), 99-127.
- Salimans, T., Ho, J., Chen, X., Sidor, S. and Sutskever, I. (2017). Evolution strategies as a scalable alternative to reinforcement learning. arXiv:1703.03864.
- Hansen, N. (2016). The CMA evolution strategy: a tutorial. arXiv:1604.00772.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
- Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M. and Dormann, N. (2021). Stable-Baselines3: reliable reinforcement learning implementations. *Journal of Machine Learning Research* 22(268), 1-8.
