import { describe, expect, it } from "vitest";
import { DinoGame } from "../sim/game";
import { GeneticAlgorithm, type GAConfig, DEFAULT_GA } from "./ga";
import {
  GENOME_LENGTH,
  HIDDEN,
  INPUTS,
  forward,
  fromGenomeFile,
  makeActivations,
  toGenomeFile,
  w1,
  w2,
} from "./network";
import { Trainer } from "./trainer";

describe("network", () => {
  it("has 96 weights for 9-8-3", () => {
    expect(GENOME_LENGTH).toBe(9 * 8 + 8 * 3);
  });

  it("computes tanh hidden units and argmax output by hand", () => {
    const g = new Float64Array(GENOME_LENGTH);
    g[0 * INPUTS + 8] = 1; // hidden 0 reads the bias input
    g[HIDDEN * INPUTS + 1 * HIDDEN + 0] = 2; // output 1 = 2 * hidden 0
    const act = makeActivations();
    act.inputs[8] = 1;
    const action = forward(g, act);
    expect(act.hidden[0]).toBeCloseTo(Math.tanh(1), 12);
    expect(act.outputs[1]).toBeCloseTo(2 * Math.tanh(1), 12);
    expect(action).toBe(1);
  });

  it("breaks ties toward 'nothing'", () => {
    expect(forward(new Float64Array(GENOME_LENGTH), makeActivations())).toBe(0);
  });

  it("indexes weights the way the layout comment says", () => {
    const g = Float64Array.from({ length: GENOME_LENGTH }, (_, i) => i);
    expect(w1(g, 2, 3)).toBe(2 * INPUTS + 3);
    expect(w2(g, 1, 4)).toBe(HIDDEN * INPUTS + 1 * HIDDEN + 4);
  });

  it("round-trips through the genome file and rejects a wrong size", () => {
    const g = Float64Array.from({ length: GENOME_LENGTH }, (_, i) => Math.sin(i));
    expect(fromGenomeFile(JSON.parse(JSON.stringify(toGenomeFile(g))))).toEqual(g);
    const bad = toGenomeFile(g);
    bad.weights.pop();
    expect(() => fromGenomeFile(bad)).toThrow();
  });
});

function toyConfig(over: Partial<GAConfig> = {}): GAConfig {
  return { ...DEFAULT_GA, genomeLength: 10, seed: 5, popSize: 60, ...over };
}

/** Maximise -sum(x^2): the optimum is the zero vector. */
function toyFitness(pop: Float64Array[]): number[] {
  return pop.map((g) => -g.reduce((s, x) => s + x * x, 0));
}

describe("genetic algorithm", () => {
  it("produces a population of the configured size", () => {
    const ga = new GeneticAlgorithm(toyConfig());
    expect(ga.population).toHaveLength(60);
    ga.evolve(toyFitness(ga.population));
    expect(ga.population).toHaveLength(60);
    expect(ga.generation).toBe(1);
  });

  it("keeps the elites unchanged and best first", () => {
    const ga = new GeneticAlgorithm(toyConfig({ eliteCount: 3 }));
    const fit = toyFitness(ga.population);
    const order = fit.map((_, i) => i).sort((a, b) => fit[b] - fit[a]);
    const expected = order.slice(0, 3).map((i) => Float64Array.from(ga.population[i]));
    ga.evolve(fit);
    expect(ga.population.slice(0, 3)).toEqual(expected);
  });

  it("never loses its best fitness when elitism is on", () => {
    const ga = new GeneticAlgorithm(toyConfig());
    let previous = -Infinity;
    for (let gen = 0; gen < 30; gen++) {
      const fit = toyFitness(ga.population);
      const best = Math.max(...fit);
      expect(best).toBeGreaterThanOrEqual(previous);
      previous = best;
      ga.evolve(fit);
    }
  });

  it("improves a toy problem substantially", () => {
    const ga = new GeneticAlgorithm(toyConfig({ mutationRate: 0.3, mutationSigma: 0.2 }));
    const start = Math.max(...toyFitness(ga.population));
    for (let gen = 0; gen < 60; gen++) ga.evolve(toyFitness(ga.population));
    const end = Math.max(...toyFitness(ga.population));
    expect(end).toBeGreaterThan(start / 5); // both negative: closer to zero is better
  });

  it("is reproducible from a seed and differs between seeds", () => {
    const run = (seed: number) => {
      const ga = new GeneticAlgorithm(toyConfig({ seed }));
      for (let i = 0; i < 5; i++) ga.evolve(toyFitness(ga.population));
      return Array.from(ga.population[0]);
    };
    expect(run(9)).toEqual(run(9));
    expect(run(9)).not.toEqual(run(10));
  });

  it("applies a new population size at the next generation", () => {
    const ga = new GeneticAlgorithm(toyConfig());
    ga.config.popSize = 25;
    ga.evolve(toyFitness(ga.population));
    expect(ga.population).toHaveLength(25);
  });

  it("decays the mutation size by the configured factor", () => {
    const ga = new GeneticAlgorithm(toyConfig({ mutationSigma: 0.4, sigmaDecay: 0.5 }));
    expect(ga.currentSigma()).toBeCloseTo(0.4);
    ga.evolve(toyFitness(ga.population));
    expect(ga.currentSigma()).toBeCloseTo(0.2);
  });

  it("makes no change to a clone when the mutation rate is zero and crossover is off", () => {
    const ga = new GeneticAlgorithm(toyConfig({ mutationRate: 0, crossoverRate: 0, eliteCount: 1 }));
    const parents = new Set(ga.population.map((g) => Array.from(g).join(",")));
    ga.evolve(toyFitness(ga.population));
    for (const child of ga.population) expect(parents.has(Array.from(child).join(","))).toBe(true);
  });

  it("draws gaussian noise with roughly zero mean and unit variance", () => {
    const ga = new GeneticAlgorithm(toyConfig());
    const xs = Array.from({ length: 20000 }, () => ga.gaussian());
    const mean = xs.reduce((s, x) => s + x, 0) / xs.length;
    const variance = xs.reduce((s, x) => s + (x - mean) ** 2, 0) / xs.length;
    expect(Math.abs(mean)).toBeLessThan(0.03);
    expect(Math.abs(variance - 1)).toBeLessThan(0.05);
  });

  it("draws training seeds below the test-seed range", () => {
    const ga = new GeneticAlgorithm(toyConfig());
    for (const s of ga.drawSeeds(200)) {
      expect(s).toBeGreaterThanOrEqual(0);
      expect(s).toBeLessThan(900_000);
    }
  });
});

function runGenerations(trainer: Trainer, count: number): void {
  const target = trainer.generation + count;
  let guard = 0;
  while (trainer.generation < target) {
    trainer.step();
    if (++guard > 5_000_000) throw new Error("trainer did not finish a generation");
  }
}

const SMALL = { popSize: 40 };
const SHORT = { rounds: 2, maxTicks: 1500 };

describe("trainer", () => {
  it("records one history entry per generation with best >= median", () => {
    const t = new Trainer(SMALL, SHORT, 3);
    runGenerations(t, 3);
    expect(t.history).toHaveLength(3);
    for (const h of t.history) {
      expect(h.best).toBeGreaterThanOrEqual(h.median);
      expect(h.best).toBeGreaterThanOrEqual(h.mean);
      expect(h.seeds).toHaveLength(2);
    }
  });

  it("gives every genome the same seed in a round", () => {
    const t = new Trainer(SMALL, SHORT, 3);
    for (const a of t.agents) expect(a.game.seed).toBe(t.seeds[0]);
  });

  it("is reproducible from its seed", () => {
    const a = new Trainer(SMALL, SHORT, 11);
    const b = new Trainer(SMALL, SHORT, 11);
    runGenerations(a, 3);
    runGenerations(b, 3);
    expect(a.history).toEqual(b.history);
  });

  it("plays each agent exactly like a standalone game driven by the same network", () => {
    const t = new Trainer(SMALL, { rounds: 1, maxTicks: 1500 }, 7);
    const genomes = t.agents.map((a) => a.genome);
    const seed = t.seeds[0];
    runGenerations(t, 1);
    const fitness = (g: Float64Array) => {
      const game = new DinoGame(seed, 1500);
      const act = makeActivations();
      while (!game.done) {
        act.inputs.set(game.observe());
        game.step(forward(g, act));
      }
      return game.score;
    };
    const expected = genomes.map(fitness);
    expect(t.history[0].best).toBeCloseTo(Math.max(...expected), 9);
    expect(t.history[0].mean).toBeCloseTo(expected.reduce((s, x) => s + x, 0) / expected.length, 9);
  });

  it("improves over generations", () => {
    // Seeded, so the result is the same every run. Fitness is noisy from generation to
    // generation (three random seeds per genome), hence the comparison against the best of
    // the last few generations.
    const t = new Trainer({ popSize: 150 }, { rounds: 3, maxTicks: 3000 }, 1);
    runGenerations(t, 18);
    const first = t.history[0].best;
    const last = Math.max(...t.history.slice(-4).map((h) => h.best));
    expect(last).toBeGreaterThan(first * 3);
    expect(t.bestEver?.fitness).toBeCloseTo(Math.max(...t.history.map((h) => h.best)));
  });

  it("follows a living dino and keeps the alive count consistent", () => {
    const t = new Trainer(SMALL, SHORT, 2);
    for (let i = 0; i < 400; i++) {
      t.step();
      const alive = t.agents.filter((a) => !a.game.done).length;
      expect(t.aliveCount).toBe(alive === 0 ? t.agents.length : alive);
      if (alive > 0) expect(t.agents[t.leader].game.done).toBe(false);
    }
  });

  it("applies a new population size from the next generation", () => {
    const t = new Trainer(SMALL, SHORT, 2);
    t.ga.config.popSize = 25;
    runGenerations(t, 1);
    expect(t.popSize).toBe(25);
  });

  it("resets to generation zero with a new seed", () => {
    const t = new Trainer(SMALL, SHORT, 2);
    runGenerations(t, 2);
    t.reset(99);
    expect(t.generation).toBe(0);
    expect(t.history).toHaveLength(0);
    expect(t.round).toBe(0);
  });
});
