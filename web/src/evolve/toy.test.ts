import { describe, expect, it } from "vitest";
import { Mulberry32 } from "../sim/game";
import { GENOME_LENGTH } from "./network";
import {
  gaussianSampler,
  genomeScore,
  mutate,
  relativeChildScore,
  ruleScore,
  winProbability,
} from "./toy";

describe("winProbability", () => {
  const a = [1, 2, 3, 4, 5, 6, 7, 8];
  const better = a.map((x) => x + 1);

  it("is 1 when B is better on every seed and 0 when it is worse on every seed", () => {
    expect(winProbability(a, better, 1, 200)).toBe(1);
    expect(winProbability(better, a, 3, 200)).toBe(0);
  });

  it("is one half for identical agents", () => {
    expect(winProbability(a, a, 4, 200)).toBe(0.5);
  });

  it("falls toward zero as K grows when the worse agent is only slightly worse", () => {
    // A is the better agent on average, but B wins on single seeds fairly often.
    const noisyA = [10, 0, 10, 0, 10, 0, 10, 0, 10, 0];
    const noisyB = [0, 9, 0, 9, 0, 9, 0, 9, 0, 9];
    const p1 = winProbability(noisyA, noisyB, 1, 4000);
    const p20 = winProbability(noisyA, noisyB, 20, 4000);
    expect(p1).toBeGreaterThan(0.3);
    expect(p20).toBeLessThan(p1);
  });

  it("is reproducible for a seed and rejects mismatched lists", () => {
    expect(winProbability(a, better, 2, 300, 5)).toBe(winProbability(a, better, 2, 300, 5));
    expect(() => winProbability([1], [1, 2], 1, 10)).toThrow();
  });
});

describe("mutation", () => {
  const parent = Float64Array.from({ length: GENOME_LENGTH }, (_, i) => Math.sin(i));

  it("changes about `rate` of the weights and leaves the parent alone", () => {
    const copy = Float64Array.from(parent);
    const { child, changed } = mutate(parent, 0.3, 0.5, new Mulberry32(3));
    expect(parent).toEqual(copy);
    expect(changed).toBeGreaterThan(30);
    expect(changed).toBeLessThan(66);
    expect(child.filter((v, i) => v !== parent[i]).length).toBe(changed);
  });

  it("changes nothing at rate zero, and larger sigma gives larger changes", () => {
    expect(mutate(parent, 0.3, 0, new Mulberry32(3)).changed).toBe(0);
    const small = mutate(parent, 0.01, 1, new Mulberry32(9)).largest;
    const large = mutate(parent, 1.0, 1, new Mulberry32(9)).largest;
    expect(large).toBeGreaterThan(small * 10);
  });

  it("draws standard-normal noise", () => {
    const g = gaussianSampler(new Mulberry32(1));
    const xs = Array.from({ length: 20000 }, g);
    const m = xs.reduce((s, x) => s + x, 0) / xs.length;
    const v = xs.reduce((s, x) => s + (x - m) ** 2, 0) / xs.length;
    expect(Math.abs(m)).toBeLessThan(0.03);
    expect(Math.abs(v - 1)).toBeLessThan(0.05);
  });
});

describe("game-backed helpers", () => {
  it("score the rule agent: a sensible distance beats a far one", () => {
    const seeds = [1, 2, 3, 4, 5];
    const total = (d: number) => seeds.reduce((s, x) => s + ruleScore(d, x, 3000), 0);
    expect(total(90)).toBeGreaterThan(total(180));
  });

  it("score a network and compare a child with its parent", () => {
    const parent = Float64Array.from({ length: GENOME_LENGTH }, (_, i) => Math.cos(i * 1.7));
    expect(genomeScore(parent, 11, 500)).toBeGreaterThan(0);
    const same = relativeChildScore(parent, 0, 1, [1, 2], 3, 500);
    expect(same).toBeCloseTo(1, 12); // zero noise: the child is the parent
  });
});
