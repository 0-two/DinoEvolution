# Neuroevolution for a Dino-style game

Technical report. Every number below is generated from the files in `results/` by `make docs`. A value that has not been produced is written TBD.

## Abstract

I trained small neural networks to play a clone of the Chrome Dino game with three evolutionary methods (a genetic algorithm, CMA-ES and NEAT), a reinforcement-learning baseline (PPO) and a hand-written rule. The game is a seeded, fixed-timestep simulation with a Python reference and a TypeScript port that agree on 2,557 sampled game states. Each method ran on 10 independent seeds, and each run's best genome was played on 100 unseen seeds for the full 36,000-tick game, where the highest possible score is 11,087. NEAT had the highest mean test score (8,033 ± 2,049), followed by the GA (4,558 ± 2,722), PPO (3,182 ± 2,773), the speed-aware rule (3,178 ± 55) and CMA-ES (2,252 ± 1,600); the fixed-distance rule scored 684 ± 42. The 95% intervals of the learned methods overlap widely, and only NEAT against CMA-ES, against the rule and, weakly, against PPO and the GA separate in permutation tests. In the GA ablations, removing elitism was the only change with a clear effect (402 ± 132 against 4,558 ± 2,722). Best genomes of the evolutionary methods lose 14% to 25% of their training score on unseen seeds. The best NEAT genome reaches the maximum score with 0 hidden nodes.

## 1. Introduction

The Chrome Dino game is a small control problem with a random obstacle sequence and a speed that rises over time. It suits neuroevolution: the score is a single number, an episode is cheap to simulate, and a network with fewer than a hundred weights can play it. I wanted to compare methods under conditions that make the comparison fair, and to write down the places where it is not.

The questions were these. How many environment steps does each method need to reach a given score? How does each method's best network perform on seeds it never saw? Which GA settings matter? What do the evolved networks do, and how large do NEAT's networks grow? The contributions are a deterministic simulation with a tested port to the browser, a pipeline that records configuration, seed and results for every run, and the comparison itself, with its limits stated.

## 2. Environment

### 2.1 Mechanics

I wrote the game from scratch and drew it with rectangles; it uses no Chrome source code or sprites. Time advances in fixed ticks of 1/60 second. A dino stands, jumps or ducks. Ducking on the ground shrinks the hitbox, and ducking in the air multiplies gravity, which shortens a jump. Collision uses axis-aligned rectangles shrunk by a few pixels so that near misses survive. The score is the distance run times a constant, so surviving longer and surviving at higher speed both raise it.

| Quantity | Value | Note |
| :--- | :--- | :--- |
| Tick rate | 60 per second | fixed timestep |
| World width | 600 px | obstacles enter at the right edge |
| Dino, standing | 44 by 47 px | left edge fixed at x = 50 |
| Dino, ducking | 59 by 26 px | on the ground only |
| Gravity | 0.6 px per tick squared |  |
| Jump speed | 10.5 px per tick | a full jump peaks near 92 px and lasts about 35 ticks |
| Fast fall | gravity times 3 | ducking in the air |
| Hitbox margin | 3 px dino, 2 px obstacle | on every side |
| Speed | 6 rising by 0.001 per tick to 13 | maximum reached after 7,000 ticks |
| Small cactus | 17 by 35 px, groups of up to 3 | groups from speed 4 |
| Large cactus | 25 by 50 px, groups of up to 3 | groups from speed 7 |
| Bird | 46 by 30 px at heights 5, 30, 60 | from speed 8.5 |
| Gap between obstacles | width x speed + 0.5 x base gap | then a random factor up to 1.5 |
| Score | distance x 0.025 |  |

The first obstacle appears after 90 ticks. A cactus group is one rectangle, wider for larger groups. Birds fly at three heights: the lowest must be jumped, the middle must be ducked or jumped, and the highest passes over a standing dino.

Episodes stop at a tick cap. At the full 36,000 ticks the highest possible score is 11,087. Training episodes stop at 12,000 ticks (a ceiling of 3,287), because one generation of 150 genomes that all survive 36,000 ticks on 3 seeds would cost 16 million environment steps.

### 2.2 Observations and actions

The network sees nine numbers, each scaled to roughly 0 to 1, and chooses among three actions by taking the largest output: 0 for nothing, 1 for jump, 2 for duck. A tie goes to the lowest index. If there is no next obstacle, the distances read 1 and the other obstacle inputs read 0.

| Input | Meaning | Scaling |
| :--- | :--- | :--- |
| 1 | distance to the next obstacle | divided by 600, at most 1 |
| 2 | width of the next obstacle | divided by 100 |
| 3 | height of the next obstacle | divided by 60 |
| 4 | height of its bottom edge above the ground | divided by 80 |
| 5 | distance to the obstacle after that | divided by 600, at most 1 |
| 6 | game speed | (speed - 6) / (13 - 6) |
| 7 | dino height above the ground | divided by 100 |
| 8 | dino vertical velocity | divided by 15 |
| 9 | bias | always 1 |

### 2.3 Determinism

Everything that affects a game is a function of its seed and the actions taken. Obstacles come from Mulberry32, a 32-bit generator that needs only integer arithmetic, so Python and JavaScript produce the same stream. The only floating-point operations are addition, subtraction, multiplication, division, comparison, floor and min, which are exactly rounded in IEEE 754 double precision in both languages. Rounding is written as floor(x + 0.5) because Python rounds half to even and JavaScript rounds half up. The order of events inside a tick is fixed and documented at the top of `sim/game.py`. Obstacles do not depend on the dino's actions, which is why the browser can show a whole population on one course.

### 2.4 Parity test

Python is the reference. `sim/parity.py` plays 10 episodes with fixed action strings (36,347 ticks in all) and stores the state every few ticks: 2,557 samples with position, speed, distance, hit count, generator state, every obstacle and the network inputs. The episodes include several deaths on different ticks, runs that reach top speed and birds, and runs with collisions switched off so that long games pass through every speed band. The TypeScript test replays the same actions and compares every field within a tolerance of 1e-6. A second test requires the two to be identical, not just close, and they are. A separate set of 12 random genomes and inputs checks the browser's network against numpy within 1e-12, 5 generator streams are compared exactly, and 6 exported NEAT and PPO policies must choose the same action as Python on 60 inputs each. When I changed one constant in the port as a check, the parity tests failed. The continuous-integration workflow runs both sides.

### 2.5 Speed

The headless Python game runs at 1,383,661 ticks per second on one core (AMD Ryzen 7 7735HS with Radeon Graphics, Python 3.11.16), against a target of 50,000. Building the nine network inputs every tick brings that to 767,133, and playing with the rule-based policy gives 1,088,861. The network's forward pass adds to this, so the wall-clock times in the results are the end-to-end figures. PPO is slower per environment step than the evolutionary methods (32.9 minutes per run for 5 million steps, against 5.3 for 20 million steps with the GA).

## 3. Methods

All learning methods except PPO act through the same 9-8-3 network: nine inputs, eight tanh hidden units, three outputs, no output bias. The ninth input is a constant 1, so its first-layer weights are the hidden biases. A genome is the 96 weights in a flat vector. NEAT evolves its own graph from the same inputs and outputs.

**Fitness.** A genome's fitness is its mean score over 3 seeds, drawn at random and shared by every genome in the generation. Averaging keeps a lucky obstacle sequence from deciding who breeds. Reward shaping is behind a flag (`jump_penalty`) and is off in every reported run.

**Genetic algorithm.** Elitism, tournament selection, uniform crossover and Gaussian mutation, as in the loop below. Parameters:

| Parameter | Value | Meaning |
| :--- | ---: | :--- |
| `pop_size` | 150 | genomes per generation |
| `elite_count` | 8 | best genomes copied unchanged (0 turns elitism off) |
| `tournament_size` | 3 | genomes drawn per tournament; the fittest wins |
| `crossover_rate` | 0.8 | chance that two parents are crossed over |
| `mutation_rate` | 0.1 | chance that a weight is perturbed |
| `mutation_sigma` | 0.3 | standard deviation of the perturbation |
| `sigma_decay` | 1 | per-generation multiplier on that deviation (1 means off) |
| `init_sigma` | 0.5 | standard deviation of the initial weights |

**CMA-ES.** The `cma` library on the same 96 weights, minimising the negative mean score. It uses ranks only. Every sample in an iteration plays the same seeds, and the seeds change between iterations, so the objective is noisy; I did not enable a noise handler.

| Parameter | Value | Meaning |
| :--- | ---: | :--- |
| `pop_size` | 50 | samples per iteration |
| `sigma0` | 0.5 | initial step size |

**NEAT.** neat-python 2.0.0 with the checked-in `sim/algorithms/neat_config.ini`. Networks start with every input connected to every output and no hidden nodes, and grow through mutation. Every node uses plain tanh (neat-python's default is tanh(2.5 x)). Selected parameters:

| Parameter | Value | Meaning |
| :--- | ---: | :--- |
| `pop_size` | 150 | genomes per generation |
| `initial_connection` | full_direct | starting topology: every input wired to every output |
| `conn_add_prob` | 0.3 | chance of adding a connection |
| `conn_delete_prob` | 0.15 | chance of deleting a connection |
| `node_add_prob` | 0.1 | chance of adding a hidden node |
| `node_delete_prob` | 0.05 | chance of deleting a hidden node |
| `weight_mutate_rate` | 0.4 | chance a weight is perturbed |
| `weight_mutate_power` | 0.3 | size of that perturbation |
| `bias_mutate_rate` | 0.2 | chance a bias is perturbed |
| `compatibility_threshold` | 1.0 | genetic distance that separates species |
| `max_stagnation` | 20 | generations without improvement before a species is dropped |
| `elitism` | 2 | best genomes per species copied unchanged |
| `survival_threshold` | 0.2 | fraction of each species allowed to breed |

**PPO.** Stable-Baselines3 2.9.2 on a Gymnasium wrapper of the same game. The reward each tick is the score gained, so an episode's return is its score. The policy is a two-layer network of width 64 with tanh, and its deterministic action is the largest logit.

| Parameter | Value | Meaning |
| :--- | ---: | :--- |
| `n_envs` | 8 | parallel environments |
| `n_steps` | 256 | steps per environment per update |
| `batch_size` | 256 | minibatch size |
| `n_epochs` | 10 | passes over each batch |
| `learning_rate` | 0.001 | Adam step size |
| `gamma` | 0.99 | discount factor |
| `gae_lambda` | 0.95 | GAE lambda |
| `ent_coef` | 0.01 | entropy bonus |
| `hidden` | 64 | width of each of the two hidden layers |
| `log_every` | 50000 | environment steps per logged window |

**Rule-based baseline.** Jump when the next obstacle is within a trigger distance, duck under a bird whose bottom edge clears a ducking dino, ignore a bird that passes over a standing dino, and do nothing while airborne. The fixed variant uses one distance; the scaled variant adds a term proportional to the speed above the starting speed. Each run tunes on 20 seeds by grid search:

| Variant | Searched | Tuning seeds per run |
| :--- | :--- | ---: |
| fixed distance | d from 20 to 300 in steps of 10 | 20 |
| speed-scaled | d from 20 to 200 in steps of 10; gain 0 to 40 in steps of 8 | 20 |

Across the 10 runs the fixed rule tuned to d 90, gain 0; d 100, gain 0, and the scaled rule to d 40, gain 32; d 60, gain 24. Each run tunes on its own 20 seeds, yet most runs land on the same grid point, which is why these scores vary little.

![The evolutionary loop.](figures/loop.svg)

## 4. Experimental setup

**Seeds.** Game seeds come from three disjoint ranges, so nothing trained or tuned on a test seed:

| Purpose | Game seeds | Use |
| :--- | :--- | :--- |
| training | 0 to 899,999 | drawn at random each generation or episode |
| rule tuning | 2,000,000 to 2,000,049 | 20 per run, offset by the run seed |
| test | 1,000,000 to 1,000,099 | final genomes only; nothing is trained or tuned on them |

Each algorithm ran on 10 independent seeds. A run seed fixes the algorithm's own random numbers.

**Budgets.** Methods are compared by environment steps, one per game tick, counted over every episode played during training. Evaluation on test seeds is not counted. The evolutionary methods stop at the first generation boundary after their budget, so a run overshoots a little. PPO gets a smaller budget because it simulates far fewer steps per second.

| Algorithm | Budget (environment steps) | Seeds per fitness | Training cap (ticks) |
| :--- | ---: | ---: | ---: |
| GA | 20 million | 3 | 12,000 |
| CMA-ES | 20 million | 3 | 12,000 |
| NEAT | 20 million | 3 | 12,000 |
| PPO | 5 million | 3 | 12,000 |
| rule, fixed | tuning grid | - | 12,000 |
| rule, scaled | tuning grid | - | 12,000 |

**Compute.** Runs used one core each, in parallel processes, on AMD Ryzen 7 7735HS with Radeon Graphics (16 hardware threads, win32 10.0.26200), Python 3.11.16, numpy 2.4.6, PyTorch 2.15.0 (CPU), Stable-Baselines3 2.9.2, neat-python 2.0.0, cma 4.6.0 and Gymnasium 1.4.0. The 140 jobs of the main suite and the ablations took 15.1 CPU-hours.

**Statistics.** Intervals are 95% Student's t intervals over runs. Differences between methods use a two-sided permutation test on the per-run test means with 20,000 resamples. I report 12 such tests without correcting for multiplicity. With a Bonferroni threshold of 0.05 divided by 12, which is 0.0042, only the clearest of them would count.

## 5. Results

### 5.1 Learning curves

![Best and mean fitness against environment steps.](figures/learning_curves.svg)

The curves are training fitness. For the GA, CMA-ES and NEAT, "best" is the best genome of each generation on its 3 seeds, and "mean" is the population's mean. For PPO, "best" is the best episode in a window of about 50,000 steps and "mean" is the mean episode score in the window, under the stochastic policy. These are not the same quantity, so the shapes can be compared and the levels cannot. At 5 million steps, where PPO's budget ends, the mean best fitness was 1,783 for PPO, 1,527 for NEAT, 907 for the GA and 484 for CMA-ES. At 20 million steps it was 3,184 for NEAT, 2,405 for the GA and 1,349 for CMA-ES, against a training ceiling of 3,287. The population means stay low for every evolutionary method (468 for NEAT, 365 for CMA-ES, 284 for the GA at the end), which is expected when most of a population is mutated children.

### 5.2 Final scores on unseen seeds

![Test score of each run's best genome.](figures/final_scores.svg)

| Algorithm | Best run | Mean ± 95% CI | Steps to 1,000 points | Generations | Wall-clock per run (s) |
| :--- | ---: | ---: | ---: | ---: | ---: |
| rule, fixed | 712 | 684 ± 42 | - | - | 1 |
| rule, scaled | 3,202 | 3,178 ± 55 | - | - | 7 |
| PPO | 10,869 | 3,182 ± 2,773 | 1.02 M (10 of 10 runs) | - | 1,974 |
| CMA-ES | 6,757 | 2,252 ± 1,600 | 9.28 M (6 of 10 runs) | 153 | 359 |
| GA | 11,087 | 4,558 ± 2,722 | 9.43 M (10 of 10 runs) | 48 | 321 |
| NEAT | 11,087 | 8,033 ± 2,049 | 4.34 M (10 of 10 runs) | 26 | 304 |

The distributions are far from normal. The median run of each learned method scored 7,509 for NEAT, 3,966 for the GA, 1,467 for PPO and 1,080 for CMA-ES. The number of runs that scored under 2,000 was 0 for NEAT, 4 for the GA, 5 for PPO and 6 for CMA-ES, out of 10. The number of runs whose mean equals the highest possible score within one point was 4 for NEAT, 1 for the GA, and 0 for PPO and 0 for CMA-ES. The speed-aware rule scored 3,178 ± 55 with almost no spread, because its tuned parameters hardly change between runs.

| Comparison | Mean A | Mean B | A minus B | p |
| :--- | ---: | ---: | ---: | ---: |
| NEAT against GA | 8,033 | 4,558 | 3,476 | 0.036 |
| NEAT against PPO | 8,033 | 3,182 | 4,852 | 0.008 |
| NEAT against CMA-ES | 8,033 | 2,252 | 5,781 | < 0.001 |
| NEAT against rule, scaled | 8,033 | 3,178 | 4,855 | < 0.001 |
| GA against PPO | 4,558 | 3,182 | 1,376 | 0.437 |
| GA against CMA-ES | 4,558 | 2,252 | 2,305 | 0.119 |
| GA against rule, scaled | 4,558 | 3,178 | 1,380 | 0.306 |
| PPO against rule, scaled | 3,182 | 3,178 | 4 | 1.000 |
| GA noelite against default | 402 | 4,558 | -4,156 | < 0.001 |
| GA sigma0_6 against default | 5,599 | 4,558 | 1,041 | 0.538 |
| GA k1 against default | 2,420 | 4,558 | -2,138 | 0.146 |
| GA k5 against default | 5,215 | 4,558 | 657 | 0.725 |

NEAT's lead over CMA-ES and over the rule is clear in these tests. Its lead over the GA (p = 0.036) and over PPO (p = 0.008) is weaker and would not survive a Bonferroni correction for 12 comparisons (threshold 0.0042). The GA, PPO and the speed-aware rule are not separated from one another: GA against PPO gives p = 0.437, and PPO's mean (3,182) is within a few points of the rule's (3,178).

### 5.3 Sample efficiency

![Environment steps to reach 1,000 and 3,000 points.](figures/sample_efficiency.svg)

To reach a generation whose best fitness was 1,000, the median run needed 1.0 million environment steps for PPO (10 of 10 runs got there), 4.3 million for NEAT (10 of 10), 9.4 million for the GA (10 of 10) and 9.3 million for CMA-ES (6 of 10). For 3,000 points, near the training ceiling, the counts were 5 of 10 for PPO (median 2.3 million), 10 of 10 for NEAT (7.7 million), 6 of 10 for the GA (12.9 million) and 3 of 10 for CMA-ES (16.0 million). PPO's first score of 1,000 comes about 9 times sooner than the GA's. This is the least like-for-like comparison in the report, because PPO's "best" is the best single episode in a window and the others' is the best of a population on three seeds. PPO's mean episode score in a window was 801 at the end of its budget. Its final test score (3,182 ± 2,773) says more about its quality than its first crossing does.

### 5.4 Generalisation to unseen seeds

![Best training fitness against score on unseen seeds, at the training cap.](figures/generalisation.svg)

| Algorithm | Best training fitness | Test at training cap | Gap | Test seeds played to the cap | Test at 36,000 ticks |
| :--- | ---: | ---: | ---: | ---: | ---: |
| GA | 2,644 | 1,990 ± 668 | 654 | 40% | 4,558 ± 2,722 |
| CMA-ES | 1,872 | 1,484 ± 709 | 388 | 20% | 2,252 ± 1,600 |
| NEAT | 3,287 | 2,825 ± 316 | 462 | 76% | 8,033 ± 2,049 |
| PPO | 1,443 | 1,514 ± 752 | -70 | 28% | 3,182 ± 2,773 |

Both scores in this table use the 12,000-tick training cap, so they can be compared. The best genome of every evolutionary method does worse on unseen seeds than on the three seeds that selected it: by 654 points (25%) for the GA, 388 (21%) for CMA-ES and 462 (14%) for NEAT. Part of that is selection: the genome with the best score on three seeds is partly the genome that was lucky on them. PPO's gap (-70) is not comparable. It has no selection step, and its training value is the best window mean under a stochastic policy, while the test plays the deterministic policy. At the training cap, 76% of the test seeds were played to the end by NEAT's best genomes, 40% by the GA's, 28% by PPO's and 20% by CMA-ES's. The full-length scores in the last column are higher than the capped ones because the game continues past the training cap.

### 5.5 GA ablations

![GA test score under changes to population size, mutation size, elitism and evaluation seeds.](figures/ablations.svg)

| Factor | Setting | Test score, mean ± 95% CI | Runs reaching 1,000 | Median steps to 1,000 |
| :--- | :--- | ---: | ---: | ---: |
| Population size | 50 | 3,859 ± 2,404 | 8 of 10 | 4.66 M |
| Population size | 150 (default) | 4,558 ± 2,722 | 10 of 10 | 9.43 M |
| Population size | 300 | 4,240 ± 2,726 | 9 of 10 | 10.19 M |
| Mutation size | 0.1 | 3,517 ± 2,655 | 8 of 10 | 8.85 M |
| Mutation size | 0.3 (default) | 4,558 ± 2,722 | 10 of 10 | 9.43 M |
| Mutation size | 0.6 | 5,599 ± 2,640 | 8 of 10 | 5.83 M |
| Elitism | off | 402 ± 132 | 0 of 10 | - |
| Elitism | on (default) | 4,558 ± 2,722 | 10 of 10 | 9.43 M |
| Seeds per fitness, K | 1 | 2,420 ± 1,576 | 10 of 10 | 2.30 M |
| Seeds per fitness, K | 3 (default) | 4,558 ± 2,722 | 10 of 10 | 9.43 M |
| Seeds per fitness, K | 5 | 5,215 ± 3,207 | 6 of 10 | 7.29 M |

Elitism off scored 402 ± 132 against 4,558 ± 2,722 for the default; the intervals do not overlap, the permutation test gives p < 0.001, and 0 of 10 runs reached 1,000 points. Without elitism the best genome of a generation can be replaced by a mutated child. I did not measure how often that happens. No other change is separated from the default. A mutation size of 0.6 scored the highest mean (5,599 ± 2,640, p = 0.538), five seeds per fitness 5,215 ± 3,207 (p = 0.725), and one seed per fitness 2,420 ± 1,576 (p = 0.146). One seed per fitness reached 1,000 points soonest (2.3 million, 10 of 10 runs) but finished with the lower mean. With five seeds only 6 of 10 runs reached 1,000 within the budget, probably because each fitness evaluation costs more steps; I did not isolate that. With ten runs per setting I cannot rank the other settings.

## 6. Analysis

### 6.1 What the evolved networks do

![Decision maps of each algorithm's best genome.](figures/decision_maps.svg)

The maps are produced by placing one obstacle at a set distance from a grounded dino, setting the game speed, and asking the policy for an action. The best genome of each method (the run with the highest test mean) is shown, with the speed-aware rule for reference.

The rule jumps within a distance that grows with speed: at speed 6 it starts jumping at a distance of 55 px, at speed 13 at 225 px. The GA's best genome (seed 9) starts at 40 px at the slowest speed and 105 px at the fastest, CMA-ES's (seed 4) at 25 and 100, and NEAT's (seed 0) at 85 and 120. So the evolved networks jump later than the rule at high speed and shift less with speed. They compensate in other ways. In front of a small cactus the GA's best genome ducks in 78% of the grid, NEAT's in 44%, CMA-ES's in 39% and PPO's in 37%, while the rule never does. Ducking on the ground is harmless while an obstacle is far away, and the fitness never penalises it. I read this as evolution settling on ducking as a default posture from which a late jump begins; I did not test that reading. For a mid-height bird, which has to be ducked, the share of duck actions is 77% for the GA, 70% for NEAT, 59% for CMA-ES and 21% for PPO.

PPO's map is different in kind. Its best policy chooses jump in 54% of the cactus grid, against 22% for the GA, including when the obstacle is far away, and its decision boundary follows an irregular band across distance and speed. I read this as a policy that jumps frequently and relies on the world being forgiving, but I did not test that reading.

### 6.2 How NEAT's networks grew

![Hidden nodes, connections and species by generation for NEAT.](figures/neat_growth.svg)

Over the first 46 generations (the shortest run's length), the best genome's mean count of hidden nodes rose from none to 1.3, while its mean count of enabled connections fell from 27 to 15. NEAT started from 27 direct connections (nine inputs times three outputs) and removed about 44% of them on average. In these runs the useful change was mostly removing connections, with a small number of hidden nodes added. The number of species rose to a mean of 15 at generation 24 and was 13 at the end of the shared range. Among the 10 final best genomes, hidden nodes ranged from 0 to 4, and 3 had none; the median connection count was 16. The best genome by test score (seed 0) has 0 hidden nodes and 8 connections and reaches the highest possible score of 11,087. A single layer of weights between the inputs and the actions was enough to reach the maximum score on the 100 test seeds, so the hidden layer of the fixed network is not needed to solve this game. A smaller search space is one possible reason for NEAT's advantage.

## 7. Discussion and limitations

NEAT's results are the best and the most consistent: 10 of 10 runs reached 1,000 points and 10 of 10 reached 3,000. A smaller search space may explain some of that. Starting from 27 weights and ending with about 16 connections is a smaller problem than 96 free weights, and the best genomes show that the problem does not need the hidden layer. I did not test this directly (for example by running the GA on a network with no hidden layer), so it is a reading of the results, not a result.

The GA is slower and more variable, and elitism is essential to it. CMA-ES, on the same network, scored lower than the GA on average, though the difference is not separated (p = 0.119), and that is not what I expected. I used 50 samples per iteration without tuning that number or using a noise handler, and fitness here is noisy because each generation draws new seeds, so this result says little about CMA-ES in general. PPO's first generation-best score of 1,000 came after about 1.0 million steps, and its final test mean was close to the tuned rule's, with a wide interval.

The main limits are these. The comparison covers one game, one machine and 10 runs per method, and the intervals are wide enough that most pairwise differences cannot be separated. The tuning effort was uneven: PPO's settings were compared on one seed (5 settings, 2 million steps each), NEAT's speciation threshold was compared on one seed, and everything else is a first guess. The "best" curves are not the same quantity across methods. Final genomes were chosen by their score on three training seeds, which inflates their training scores and lowers their test scores a little for every evolutionary method. The game is easy to saturate: 4 NEAT runs reach the maximum score, so differences at the top cannot show. Training at 12,000 ticks and testing at 36,000 ticks mixes two horizons. There is no git history, so run files record `commit: unknown`. The web page's frame rate was measured in one browser (headless Edg/157.0.4341.60), not Firefox.

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
- Statistics: t intervals and permutation tests in `sim/stats.py`, with 20,000 resamples and a fixed seed.
- Known gaps: Firefox not tested; no correction for multiple comparisons; PPO, GA, CMA-ES and NEAT not tuned to the same extent.

## 9. References

- Stanley, K. O. and Miikkulainen, R. (2002). Evolving neural networks through augmenting topologies. *Evolutionary Computation* 10(2), 99-127.
- Salimans, T., Ho, J., Chen, X., Sidor, S. and Sutskever, I. (2017). Evolution strategies as a scalable alternative to reinforcement learning. arXiv:1703.03864.
- Hansen, N. (2016). The CMA evolution strategy: a tutorial. arXiv:1604.00772.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A. and Klimov, O. (2017). Proximal policy optimization algorithms. arXiv:1707.06347.
- Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M. and Dormann, N. (2021). Stable-Baselines3: reliable reinforcement learning implementations. *Journal of Machine Learning Research* 22(268), 1-8.
