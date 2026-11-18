/**
 * Small experiments for the "How it works" page. They use the real game and network, on a
 * small scale so that they finish in a fraction of a second in the browser.
 *
 *  - Seed averaging: how often does a worse agent look better than a better one when each is
 *    judged on K random seeds?
 *  - Mutation strength: how does a child's score depend on the size of the Gaussian noise
 *    added to its parent's weights?
 *
 * Everything here is deterministic for a given seed, and the pure functions are tested in
 * toy.test.ts.
 */
import { DinoGame, Mulberry32 } from "../sim/game";
import { ruleAction } from "../sim/rule";
import { forward, makeActivations } from "./network";

/** Score of the rule-based agent with trigger distance `d` on one seed. */
export function ruleScore(d: number, seed: number, maxTicks: number): number {
  const game = new DinoGame(seed, maxTicks);
  const params = { d, gain: 0 };
  while (game.step(ruleAction(game, params))) {
    /* run to the end */
  }
  return game.score;
}

/** Score of a 9-8-3 network on one seed. */
export function genomeScore(genome: Float64Array, seed: number, maxTicks: number): number {
  const game = new DinoGame(seed, maxTicks);
  const act = makeActivations();
  while (!game.done) {
    game.observeInto(act.inputs);
    game.step(forward(genome, act));
  }
  return game.score;
}

function mean(xs: readonly number[]): number {
  return xs.reduce((s, x) => s + x, 0) / xs.length;
}

/**
 * Probability that agent B's mean over K random seeds beats agent A's mean over K random seeds,
 * when both are judged on the same K seeds drawn with replacement from a common pool.
 * `a[i]` and `b[i]` are the two agents' scores on pool seed i.
 */
export function winProbability(
  a: readonly number[],
  b: readonly number[],
  k: number,
  draws: number,
  seed = 1,
): number {
  if (a.length !== b.length || a.length === 0) throw new Error("score lists must match");
  const rng = new Mulberry32(seed);
  let wins = 0;
  for (let t = 0; t < draws; t++) {
    let sa = 0;
    let sb = 0;
    for (let i = 0; i < k; i++) {
      const j = rng.randint(0, a.length - 1);
      sa += a[j];
      sb += b[j];
    }
    if (sb > sa) wins++;
    else if (sb === sa) wins += 0.5;
  }
  return wins / draws;
}

/** Standard-normal sampler on a Mulberry32 stream (Box-Muller). */
export function gaussianSampler(rng: Mulberry32): () => number {
  let spare: number | null = null;
  return () => {
    if (spare !== null) {
      const s = spare;
      spare = null;
      return s;
    }
    const u = 1 - rng.random();
    const v = rng.random();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
}

export interface Mutation {
  child: Float64Array;
  /** Number of weights that changed. */
  changed: number;
  /** Largest absolute change to any weight. */
  largest: number;
}

/** Copy `parent` and add N(0, sigma) noise to each weight with probability `rate`. */
export function mutate(parent: Float64Array, sigma: number, rate: number, rng: Mulberry32): Mutation {
  const gauss = gaussianSampler(rng);
  const child = Float64Array.from(parent);
  let changed = 0;
  let largest = 0;
  for (let i = 0; i < child.length; i++) {
    if (rng.random() < rate) {
      const delta = sigma * gauss();
      child[i] += delta;
      changed++;
      largest = Math.max(largest, Math.abs(delta));
    }
  }
  return { child, changed, largest };
}

/**
 * Mean score of `children` mutated children of `parent` on `seeds`, divided by the parent's own
 * mean score on the same seeds. Below 1 means the average child is worse than its parent.
 */
export function relativeChildScore(
  parent: Float64Array,
  sigma: number,
  rate: number,
  seeds: readonly number[],
  children: number,
  maxTicks: number,
  rngSeed = 7,
): number {
  const parentScore = mean(seeds.map((s) => genomeScore(parent, s, maxTicks)));
  const rng = new Mulberry32(rngSeed);
  const childScores: number[] = [];
  for (let c = 0; c < children; c++) {
    const { child } = mutate(parent, sigma, rate, rng);
    childScores.push(mean(seeds.map((s) => genomeScore(child, s, maxTicks))));
  }
  return mean(childScores) / parentScore;
}
