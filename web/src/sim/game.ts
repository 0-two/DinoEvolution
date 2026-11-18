/**
 * Port of sim/game.py. The Python file is the reference: read its module docstring for the
 * rules this file has to follow. Keep the tick order, the order of RNG draws and the float
 * expressions identical, or the parity test (parity.test.ts) will fail.
 *
 * Only + - * /, comparisons, Math.floor and Math.min are applied to floats, because those
 * are exactly rounded in both languages. Math.round is avoided on purpose (it rounds half
 * up, Python rounds half to even), so rounding is spelled Math.floor(x + 0.5).
 */

export const TICKS_PER_SECOND = 60;
export const WORLD_WIDTH = 600.0;

export const DINO_X = 50.0;
export const DINO_W = 44.0;
export const DINO_H = 47.0;
export const DUCK_W = 59.0;
export const DUCK_H = 26.0;
export const GRAVITY = 0.6;
export const JUMP_V = 10.5;
export const FAST_FALL = 3.0;
export const DINO_INSET = 3.0;
export const OBSTACLE_INSET = 2.0;

export const START_SPEED = 6.0;
export const MAX_SPEED = 13.0;
export const ACCEL = 0.001;
export const SCORE_PER_PX = 0.025;

export const START_CLEAR_TICKS = 90;
export const GAP_COEFF = 0.5;
export const MAX_GAP_COEFF = 1.5;
export const MAX_GROUP = 3;

/** [unit width, height, base min gap, speed from which groups are allowed] */
type CactusKind = readonly [number, number, number, number];
const SMALL_CACTUS: CactusKind = [17.0, 35.0, 120.0, 4.0];
const LARGE_CACTUS: CactusKind = [25.0, 50.0, 120.0, 7.0];
const CACTI: readonly CactusKind[] = [SMALL_CACTUS, LARGE_CACTUS];
export const BIRD_W = 46.0;
export const BIRD_H = 30.0;
const BIRD_MIN_GAP = 150.0;
export const BIRD_MIN_SPEED = 8.5;
export const BIRD_BOTTOMS: readonly number[] = [5.0, 30.0, 60.0];

export const DEFAULT_MAX_TICKS = 36_000;

const OBS_DIST_NORM = WORLD_WIDTH;
const OBS_WIDTH_NORM = 100.0;
const OBS_HEIGHT_NORM = 60.0;
const OBS_Y_NORM = 80.0;
const OBS_DINO_Y_NORM = 100.0;
const OBS_VY_NORM = 15.0;
export const OBSERVATION_SIZE = 9;

export const NOOP = 0;
export const JUMP = 1;
export const DUCK = 2;

/** Obstacle as [x, y, w, h]; y is the bottom edge above the ground. */
export type Obstacle = [number, number, number, number];

/** Mulberry32: 32-bit state, integer arithmetic only, same stream as the Python class. */
export class Mulberry32 {
  state: number;

  constructor(seed: number) {
    this.state = seed >>> 0;
  }

  nextU32(): number {
    this.state = (this.state + 0x6d2b79f5) >>> 0;
    let t = this.state;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return (t ^ (t >>> 14)) >>> 0;
  }

  /** Uniform in [0, 1). Dividing by 2^32 is exact in double precision. */
  random(): number {
    return this.nextU32() / 4294967296.0;
  }

  /** Uniform integer in [lo, hi], both ends included. */
  randint(lo: number, hi: number): number {
    return lo + Math.floor(this.random() * (hi - lo + 1));
  }
}

export interface Snapshot {
  tick: number;
  dead: boolean;
  distance: number;
  speed: number;
  y: number;
  vy: number;
  ducking: boolean;
  hitTicks: number;
  rngState: number;
  obstacles: Obstacle[];
}

export class DinoGame {
  seed: number;
  maxTicks: number;
  collisions: boolean;

  rng!: Mulberry32;
  tick = 0;
  distance = 0.0;
  speed = START_SPEED;
  y = 0.0;
  vy = 0.0;
  airborne = false;
  ducking = false;
  dead = false;
  deathTick: number | null = null;
  hitTicks = 0;
  gap = 0.0;
  /** Ordered by x, nearest first. */
  obstacles: Obstacle[] = [];

  constructor(seed: number, maxTicks: number = DEFAULT_MAX_TICKS, collisions = true) {
    this.seed = seed;
    this.maxTicks = maxTicks;
    this.collisions = collisions;
    this.reset(seed);
  }

  reset(seed?: number): void {
    if (seed !== undefined) this.seed = seed;
    this.rng = new Mulberry32(this.seed);
    this.tick = 0;
    this.distance = 0.0;
    this.speed = START_SPEED;
    this.y = 0.0;
    this.vy = 0.0;
    this.airborne = false;
    this.ducking = false;
    this.dead = false;
    this.deathTick = null;
    this.hitTicks = 0;
    this.gap = 0.0;
    this.obstacles = [];
  }

  get done(): boolean {
    return this.dead || this.tick >= this.maxTicks;
  }

  get score(): number {
    return this.distance * SCORE_PER_PX;
  }

  /** Advance one tick. Returns true while the episode is still running. */
  step(action: number): boolean {
    if (this.dead || this.tick >= this.maxTicks) return false;

    if (!this.airborne) {
      if (action === JUMP) {
        this.vy = JUMP_V;
        this.airborne = true;
        this.ducking = false;
      } else {
        this.ducking = action === DUCK;
      }
    }
    if (this.airborne) {
      let vy = this.vy - (action === DUCK ? GRAVITY * FAST_FALL : GRAVITY);
      let y = this.y + vy;
      if (y <= 0.0) {
        y = 0.0;
        vy = 0.0;
        this.airborne = false;
      }
      this.y = y;
      this.vy = vy;
    }

    const speed = this.speed;
    const obs = this.obstacles;
    for (const o of obs) o[0] -= speed;
    while (obs.length > 0 && obs[0][0] + obs[0][2] < 0.0) obs.shift();

    if (this.tick >= START_CLEAR_TICKS) {
      if (obs.length > 0) {
        const last = obs[obs.length - 1];
        if (last[0] + last[2] + this.gap < WORLD_WIDTH) this.spawn();
      } else {
        this.spawn();
      }
    }

    this.distance += speed;
    this.speed = Math.min(MAX_SPEED, speed + ACCEL);
    this.tick += 1;

    if (this.obstacles.length > 0 && this.collides()) {
      if (this.collisions) {
        this.dead = true;
        this.deathTick = this.tick;
        return false;
      }
      this.hitTicks += 1;
    }
    return this.tick < this.maxTicks;
  }

  /** Draw order matters: kind, then group size or bird height, then the gap. */
  private spawn(): void {
    const rng = this.rng;
    const speed = this.speed;
    const kind = rng.randint(0, speed >= BIRD_MIN_SPEED ? 2 : 1);
    let w: number;
    let h: number;
    let minGap: number;
    let y: number;
    if (kind < 2) {
      const [unitW, height, baseGap, groupSpeed] = CACTI[kind];
      let size = rng.randint(1, MAX_GROUP);
      if (size > 1 && speed < groupSpeed) size = 1;
      w = unitW * size;
      h = height;
      minGap = baseGap;
      y = 0.0;
    } else {
      w = BIRD_W;
      h = BIRD_H;
      minGap = BIRD_MIN_GAP;
      y = BIRD_BOTTOMS[rng.randint(0, 2)];
    }
    const lo = Math.floor(w * speed + minGap * GAP_COEFF + 0.5);
    const hi = Math.floor(lo * MAX_GAP_COEFF + 0.5);
    this.gap = rng.randint(lo, hi);
    this.obstacles.push([WORLD_WIDTH, y, w, h]);
  }

  /** AABB test between the shrunken dino and each shrunken obstacle. */
  private collides(): boolean {
    const dw = this.ducking ? DUCK_W : DINO_W;
    const dh = this.ducking ? DUCK_H : DINO_H;
    const dLeft = DINO_X + DINO_INSET;
    const dRight = DINO_X + dw - DINO_INSET;
    const dBottom = this.y + DINO_INSET;
    const dTop = this.y + dh - DINO_INSET;
    for (const [ox, oy, ow, oh] of this.obstacles) {
      if (ox >= dRight) break;
      if (
        ox + ow - OBSTACLE_INSET > dLeft &&
        ox + OBSTACLE_INSET < dRight &&
        oy + oh - OBSTACLE_INSET > dBottom &&
        oy + OBSTACLE_INSET < dTop
      ) {
        return true;
      }
    }
    return false;
  }

  /** The next two obstacles the dino has not fully passed. */
  upcoming(): [Obstacle | null, Obstacle | null] {
    let first: Obstacle | null = null;
    for (const o of this.obstacles) {
      if (o[0] + o[2] > DINO_X) {
        if (first === null) first = o;
        else return [first, o];
      }
    }
    return [first, null];
  }

  /** Gap between the dino's standing front edge and the obstacle, floored at 0. */
  distanceTo(o: Obstacle): number {
    return Math.max(0.0, o[0] - (DINO_X + DINO_W));
  }

  /** Network inputs, each roughly in [0, 1]. Missing obstacles read as far away. */
  observe(): number[] {
    const out = new Array<number>(OBSERVATION_SIZE);
    this.observeInto(out);
    return out;
  }

  /**
   * Same values as `observe`, written into `out` without allocating. The trainer calls this
   * once per agent per tick. It scans the obstacle list directly instead of calling
   * `upcoming()`, which would build a tuple every time.
   */
  observeInto(out: number[] | Float64Array): void {
    const obs = this.obstacles;
    let first: Obstacle | null = null;
    let second: Obstacle | null = null;
    for (let i = 0; i < obs.length; i++) {
      const o = obs[i];
      if (o[0] + o[2] > DINO_X) {
        if (first === null) first = o;
        else {
          second = o;
          break;
        }
      }
    }
    if (first === null) {
      out[0] = 1.0;
      out[1] = 0.0;
      out[2] = 0.0;
      out[3] = 0.0;
    } else {
      out[0] = Math.min(1.0, this.distanceTo(first) / OBS_DIST_NORM);
      out[1] = first[2] / OBS_WIDTH_NORM;
      out[2] = first[3] / OBS_HEIGHT_NORM;
      out[3] = first[1] / OBS_Y_NORM;
    }
    out[4] = second === null ? 1.0 : Math.min(1.0, this.distanceTo(second) / OBS_DIST_NORM);
    out[5] = (this.speed - START_SPEED) / (MAX_SPEED - START_SPEED);
    out[6] = this.y / OBS_DINO_Y_NORM;
    out[7] = this.vy / OBS_VY_NORM;
    out[8] = 1.0;
  }

  snapshot(): Snapshot {
    return {
      tick: this.tick,
      dead: this.dead,
      distance: this.distance,
      speed: this.speed,
      y: this.y,
      vy: this.vy,
      ducking: this.ducking,
      hitTicks: this.hitTicks,
      rngState: this.rng.state,
      obstacles: this.obstacles.map((o) => [...o] as Obstacle),
    };
  }
}
