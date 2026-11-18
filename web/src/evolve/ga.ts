/**
 * Fixed-topology genetic algorithm: elitism, tournament selection, uniform crossover and
 * Gaussian mutation with optional decay.
 *
 * The class holds only the population and the random stream. Evaluation lives in the
 * trainer, so this file has no knowledge of the game and can be tested on a toy problem.
 *
 * Reproducibility: the same seed gives the same run in the same browser. Gaussian noise uses
 * Math.log and Math.cos, which different engines may round differently in the last digit, so
 * runs are not promised to match across browsers. Nothing here has to match the Python code.
 */
import { Mulberry32 } from "../sim/game";

/** Training episodes use seeds below this bound. Test seeds sit above it (see sim/seeds.py). */
export const TRAIN_SEED_LIMIT = 900_000;

export interface GAConfig {
  popSize: number;
  genomeLength: number;
  /** Best genomes copied unchanged into the next generation. */
  eliteCount: number;
  tournamentSize: number;
  /** Probability that two parents are crossed over; otherwise the child clones the first. */
  crossoverRate: number;
  /** Probability that a gene is perturbed. */
  mutationRate: number;
  /** Standard deviation of the perturbation at generation 0. */
  mutationSigma: number;
  /** Multiplier applied to the standard deviation every generation. 1 turns decay off. */
  sigmaDecay: number;
  /** Standard deviation of the initial weights. */
  initSigma: number;
  seed: number;
}

export const DEFAULT_GA: Omit<GAConfig, "genomeLength" | "seed"> = {
  popSize: 150,
  eliteCount: 8,
  tournamentSize: 3,
  crossoverRate: 0.8,
  mutationRate: 0.1,
  mutationSigma: 0.3,
  sigmaDecay: 1,
  initSigma: 0.5,
};

export class GeneticAlgorithm {
  readonly config: GAConfig;
  rng: Mulberry32;
  generation = 0;
  /** Sorted best-first after the first call to `evolve`; arbitrary order before that. */
  population: Float64Array[] = [];
  private spare: number | null = null;

  constructor(config: GAConfig) {
    this.config = config;
    this.rng = new Mulberry32(config.seed);
    this.population = this.initialPopulation();
  }

  /** Start over with a new seed. */
  reset(seed: number): void {
    this.config.seed = seed;
    this.rng = new Mulberry32(seed);
    this.spare = null;
    this.generation = 0;
    this.population = this.initialPopulation();
  }

  /** Standard normal sample by the Box-Muller transform. */
  gaussian(): number {
    if (this.spare !== null) {
      const s = this.spare;
      this.spare = null;
      return s;
    }
    // 1 - random() is in (0, 1], so the logarithm is finite.
    const u = 1 - this.rng.random();
    const v = this.rng.random();
    const r = Math.sqrt(-2 * Math.log(u));
    this.spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  }

  /** Training seeds for the next generation. Every genome plays the same ones. */
  drawSeeds(count: number): number[] {
    return Array.from({ length: count }, () => this.rng.randint(0, TRAIN_SEED_LIMIT - 1));
  }

  /** Mutation standard deviation for the generation that is about to be created. */
  currentSigma(): number {
    return this.config.mutationSigma * Math.pow(this.config.sigmaDecay, this.generation);
  }

  /**
   * Replace the population using `fitness[i]` for `population[i]`. Pop size, rates and the
   * decay are read from the config now, so they can be changed between generations.
   */
  evolve(fitness: readonly number[]): void {
    const c = this.config;
    const order = Array.from(fitness, (_, i) => i).sort((a, b) => fitness[b] - fitness[a] || a - b);
    const ranked = order.map((i) => this.population[i]);
    const rankedFitness = order.map((i) => fitness[i]);

    this.generation += 1;
    const sigma = this.currentSigma();
    const elites = Math.min(c.eliteCount, c.popSize, ranked.length);
    const next: Float64Array[] = ranked.slice(0, elites).map((g) => Float64Array.from(g));
    while (next.length < c.popSize) {
      const a = ranked[this.tournament(rankedFitness)];
      const b = ranked[this.tournament(rankedFitness)];
      next.push(this.mutate(this.crossover(a, b), sigma));
    }
    this.population = next;
  }

  /** Index of the winner among `tournamentSize` random picks. Lower index means fitter. */
  private tournament(rankedFitness: readonly number[]): number {
    let best = this.rng.randint(0, rankedFitness.length - 1);
    for (let i = 1; i < this.config.tournamentSize; i++) {
      const challenger = this.rng.randint(0, rankedFitness.length - 1);
      if (challenger < best) best = challenger; // ranked best-first, so smaller is fitter
    }
    return best;
  }

  /** Uniform crossover: each gene comes from either parent with equal probability. */
  private crossover(a: Float64Array, b: Float64Array): Float64Array {
    const child = Float64Array.from(a);
    if (this.rng.random() >= this.config.crossoverRate) return child;
    for (let i = 0; i < child.length; i++) {
      if (this.rng.random() < 0.5) child[i] = b[i];
    }
    return child;
  }

  private mutate(genome: Float64Array, sigma: number): Float64Array {
    const rate = this.config.mutationRate;
    for (let i = 0; i < genome.length; i++) {
      if (this.rng.random() < rate) genome[i] += sigma * this.gaussian();
    }
    return genome;
  }

  private initialPopulation(): Float64Array[] {
    const { popSize, genomeLength, initSigma } = this.config;
    return Array.from({ length: popSize }, () => {
      const g = new Float64Array(genomeLength);
      for (let i = 0; i < genomeLength; i++) g[i] = initSigma * this.gaussian();
      return g;
    });
  }
}
