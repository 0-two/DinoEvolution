/**
 * Replays the golden traces written by `python -m sim.parity write`.
 *
 * The Python game is the reference. Each episode stores an explicit action string and the
 * state sampled every few ticks. This test plays the same actions through the TypeScript
 * port and compares every field. Integers, booleans and RNG states must match exactly;
 * floats must agree within 1e-6 (in practice they are identical).
 */
import { readFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { describe, expect, it } from "vitest";
import { DinoGame, Mulberry32 } from "./game";
import { forward, makeActivations } from "../evolve/network";
import { loadPolicy, type PolicyFile } from "../evolve/policies";
import { ruleAction } from "./rule";

const TOLERANCE = 1e-6;
const FIXTURE_URL = new URL("../../../sim/tests/golden/parity_traces.json", import.meta.url);

type Obstacle = [number, number, number, number];
type Sample = [
  number, // tick
  boolean, // dead
  number, // distance
  number, // speed
  number, // y
  number, // vy
  boolean, // ducking
  number, // hit_ticks
  number, // rng_state
  Obstacle[],
  number[], // observation
];

interface Episode {
  name: string;
  seed: number;
  collisions: boolean;
  max_ticks: number;
  sample_every: number;
  actions: string;
  rule?: { d: number; gain: number };
  samples: Sample[];
  ticks: number;
  death_tick: number | null;
  score: number;
}

interface Fixture {
  version: number;
  fields: string[];
  rng: { seed: number; u32: number[] }[];
  episodes: Episode[];
  policies: { payload: PolicyFile; inputs: number[][]; actions: number[] }[];
  network: { genome: number[]; inputs: number[]; hidden: number[]; outputs: number[]; action: number }[];
}

const fixture: Fixture = JSON.parse(readFileSync(fileURLToPath(FIXTURE_URL), "utf8"));

function sample(game: DinoGame): Sample {
  const s = game.snapshot();
  return [
    s.tick,
    s.dead,
    s.distance,
    s.speed,
    s.y,
    s.vy,
    s.ducking,
    s.hitTicks,
    s.rngState,
    s.obstacles,
    game.observe(),
  ];
}

function replay(ep: Episode) {
  const game = new DinoGame(ep.seed, ep.max_ticks, ep.collisions);
  const samples: Sample[] = [sample(game)];
  for (let i = 0; i < ep.actions.length; i++) {
    const alive = game.step(Number(ep.actions[i]));
    if ((i + 1) % ep.sample_every === 0) samples.push(sample(game));
    if (!alive) break;
  }
  samples.push(sample(game));
  return { samples, ticks: game.tick, deathTick: game.deathTick, score: game.score };
}

/** Largest absolute difference between two equal-shaped numeric structures. */
function maxDiff(a: unknown, b: unknown): number {
  if (typeof a === "number" && typeof b === "number") return Math.abs(a - b);
  if (Array.isArray(a) && Array.isArray(b)) {
    if (a.length !== b.length) return Infinity;
    return a.reduce((m: number, v, i) => Math.max(m, maxDiff(v, b[i])), 0);
  }
  return a === b ? 0 : Infinity;
}

describe("fixture", () => {
  it("is the expected version and has episodes", () => {
    expect(fixture.version).toBe(1);
    expect(fixture.episodes.length).toBeGreaterThanOrEqual(8);
  });
});

describe("Mulberry32", () => {
  for (const entry of fixture.rng) {
    it(`matches the Python stream for seed ${entry.seed}`, () => {
      const rng = new Mulberry32(entry.seed);
      expect(entry.u32.map(() => rng.nextU32())).toEqual(entry.u32);
    });
  }
});

describe("game parity with Python", () => {
  for (const ep of fixture.episodes) {
    it(`episode ${ep.name}`, () => {
      const got = replay(ep);
      expect(got.samples.length).toBe(ep.samples.length);
      expect(got.ticks).toBe(ep.ticks);
      expect(got.deathTick).toBe(ep.death_tick);
      expect(Math.abs(got.score - ep.score)).toBeLessThanOrEqual(TOLERANCE);
      for (let i = 0; i < ep.samples.length; i++) {
        const diff = maxDiff(got.samples[i], ep.samples[i]);
        if (diff > TOLERANCE) {
          // Report the first sample that disagrees, with both rows, before failing.
          expect({ index: i, got: got.samples[i] }).toEqual({ index: i, got: ep.samples[i] });
        }
        expect(diff).toBeLessThanOrEqual(TOLERANCE);
      }
    });
  }
});

describe("exact agreement with Python", () => {
  // The two games use only operations that are exactly rounded in both languages, so the
  // sampled states should match bit for bit, not only within the 1e-6 tolerance.
  for (const ep of fixture.episodes) {
    it(`episode ${ep.name} is identical`, () => {
      expect(replay(ep).samples).toEqual(ep.samples);
    });
  }
});

describe("rule-based agent parity with Python", () => {
  for (const ep of fixture.episodes.filter((e) => e.rule !== undefined)) {
    it(`reproduces the actions of ${ep.name}`, () => {
      const params = ep.rule!;
      const game = new DinoGame(ep.seed, ep.max_ticks, ep.collisions);
      let played = "";
      while (!game.done) {
        const action = ruleAction(game, params);
        played += String(action);
        game.step(action);
      }
      expect(played).toBe(ep.actions);
    });
  }
});

describe("network parity with Python", () => {
  fixture.network.forEach((v, i) => {
    it(`vector ${i}: hidden units, outputs and action`, () => {
      const act = makeActivations();
      act.inputs.set(v.inputs);
      const action = forward(Float64Array.from(v.genome), act);
      expect(maxDiff(Array.from(act.hidden), v.hidden)).toBeLessThanOrEqual(1e-12);
      expect(maxDiff(Array.from(act.outputs), v.outputs)).toBeLessThanOrEqual(1e-12);
      expect(action).toBe(v.action);
    });
  });
});

describe("exported policy parity with Python", () => {
  fixture.policies.forEach((v, i) => {
    it(`${v.payload.format} vector ${i}: same action on every input`, () => {
      const policy = loadPolicy(v.payload);
      const got = v.inputs.map((x) => policy.actFromObservation!(x));
      expect(got).toEqual(v.actions);
    });
  });

  it("covers every action across the vectors", () => {
    const all = new Set(fixture.policies.flatMap((p) => p.actions));
    expect(all).toEqual(new Set([0, 1, 2]));
  });
});
