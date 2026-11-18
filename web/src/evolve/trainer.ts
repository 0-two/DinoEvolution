/**
 * Runs the GA against the game and keeps everything the live page needs to draw.
 *
 * One generation is `rounds` episodes. In each round every genome plays the same seed at the
 * same time, so the whole population can be shown on one canvas. A genome's fitness is the
 * mean of its scores over the rounds, which stops one lucky obstacle sequence from deciding
 * who breeds. All seeds for a generation are drawn up front from the GA's random stream.
 *
 * `step()` advances every living agent by one tick and never touches the DOM.
 */
import { DinoGame, DEFAULT_MAX_TICKS } from "../sim/game";
import { DEFAULT_GA, GeneticAlgorithm, type GAConfig } from "./ga";
import { GENOME_LENGTH, forward, makeActivations, type Activations } from "./network";

export interface TrainerConfig {
  /** Evaluation seeds per genome, called K in the report. */
  rounds: number;
  maxTicks: number;
}

export interface GenerationStat {
  generation: number;
  best: number;
  mean: number;
  median: number;
  seeds: number[];
}

export interface Agent {
  genome: Float64Array;
  game: DinoGame;
  act: Activations;
  /** Action chosen on the latest tick: 0 nothing, 1 jump, 2 duck. */
  action: number;
  /** Sum of final scores over the finished rounds. */
  total: number;
}

export const DEFAULT_TRAINER: TrainerConfig = { rounds: 3, maxTicks: DEFAULT_MAX_TICKS };

export class Trainer {
  readonly ga: GeneticAlgorithm;
  readonly config: TrainerConfig;
  agents: Agent[] = [];
  /** 0-based index of the round in progress. */
  round = 0;
  seeds: number[] = [];
  aliveCount = 0;
  /** Index into `agents` of the dino the page follows. */
  leader = 0;
  history: GenerationStat[] = [];
  bestEver: { fitness: number; genome: Float64Array; generation: number } | null = null;
  /** Total agent-ticks simulated since the last reset, for throughput measurements. */
  agentTicks = 0;

  constructor(
    gaOverrides: Partial<GAConfig> = {},
    config: Partial<TrainerConfig> = {},
    seed = 1,
  ) {
    this.config = { ...DEFAULT_TRAINER, ...config };
    this.ga = new GeneticAlgorithm({
      ...DEFAULT_GA,
      genomeLength: GENOME_LENGTH,
      seed,
      ...gaOverrides,
    });
    this.beginGeneration();
  }

  reset(seed: number): void {
    this.ga.reset(seed);
    this.history = [];
    this.bestEver = null;
    this.agentTicks = 0;
    this.beginGeneration();
  }

  get generation(): number {
    return this.ga.generation;
  }

  get popSize(): number {
    return this.agents.length;
  }

  /** Highest score reached in the current round, dead or alive. */
  get roundBest(): number {
    let best = 0;
    for (const a of this.agents) if (a.game.score > best) best = a.game.score;
    return best;
  }

  /** The game of the followed dino. Every living dino shares its obstacles and speed. */
  get leaderGame(): DinoGame {
    return this.agents[this.leader].game;
  }

  /** Advance every living agent one tick. Starts the next round or generation when needed. */
  step(): void {
    let alive = 0;
    for (const a of this.agents) {
      const game = a.game;
      if (game.done) continue;
      game.observeInto(a.act.inputs);
      a.action = forward(a.genome, a.act);
      game.step(a.action);
      if (!game.done) alive++;
    }
    this.agentTicks += this.aliveCount;
    this.aliveCount = alive;
    if (alive === 0) this.endRound();
    else if (this.agents[this.leader].game.done) this.chooseLeader();
  }

  private beginGeneration(): void {
    this.seeds = this.ga.drawSeeds(this.config.rounds);
    this.round = 0;
    const pop = this.ga.population;
    // Reuse game objects when the population size is unchanged; they are reset per round.
    this.agents = pop.map((genome, i) => ({
      genome,
      game: this.agents[i]?.game ?? new DinoGame(0, this.config.maxTicks),
      act: this.agents[i]?.act ?? makeActivations(),
      action: 0,
      total: 0,
    }));
    this.beginRound();
  }

  private beginRound(): void {
    for (const a of this.agents) a.game.reset(this.seeds[this.round]);
    this.aliveCount = this.agents.length;
    this.chooseLeader();
  }

  private endRound(): void {
    for (const a of this.agents) a.total += a.game.score;
    this.round += 1;
    if (this.round < this.config.rounds) this.beginRound();
    else this.finishGeneration();
  }

  private finishGeneration(): void {
    const k = this.config.rounds;
    const fitness = this.agents.map((a) => a.total / k);
    const sorted = [...fitness].sort((a, b) => a - b);
    const mid = sorted.length >> 1;
    const median = sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
    const best = Math.max(...fitness);
    // 1-based, so the first finished generation is generation 1 on the chart.
    this.history.push({
      generation: this.ga.generation + 1,
      best,
      mean: fitness.reduce((s, f) => s + f, 0) / fitness.length,
      median,
      seeds: this.seeds,
    });
    if (this.bestEver === null || best > this.bestEver.fitness) {
      const i = fitness.indexOf(best);
      this.bestEver = {
        fitness: best,
        genome: Float64Array.from(this.agents[i].genome),
        generation: this.ga.generation + 1,
      };
    }
    this.ga.evolve(fitness);
    this.beginGeneration();
  }

  /**
   * Pick the dino to follow: the living one with the best total so far, counting the round
   * in progress. Ties go to the lowest index, which after the first generation is the genome
   * that ranked highest last time (elites are placed first, best first).
   */
  private chooseLeader(): void {
    let best = -1;
    let bestScore = -Infinity;
    for (let i = 0; i < this.agents.length; i++) {
      const a = this.agents[i];
      if (a.game.done) continue;
      const score = a.total + a.game.score;
      if (score > bestScore) {
        bestScore = score;
        best = i;
      }
    }
    if (best >= 0) this.leader = best;
  }
}
