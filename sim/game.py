"""Deterministic, headless Dino simulation.

This file is the reference implementation. The TypeScript port in ``web/src/sim`` has to
reproduce it tick for tick, so everything here is written to be portable:

* Time is a fixed tick (60 per simulated second). Nothing depends on wall-clock time.
* Randomness comes from Mulberry32, a 32-bit generator that needs only integer
  arithmetic. Python's ``random`` and JavaScript's ``Math.random`` cannot be made to agree.
* Only ``+ - * /``, comparison, ``floor`` and ``min`` are applied to floats. These are
  exactly rounded in IEEE 754 double precision in both languages, so traces match
  bit for bit, not just within a tolerance. Do not add ``sqrt``, ``pow``, ``sin`` etc.
* ``round`` is spelled ``floor(x + 0.5)`` because Python rounds half to even and
  JavaScript rounds half up.

Coordinates: x grows to the right, y is height above the ground (y-up). A rectangle is
``(x, y, w, h)`` with ``(x, y)`` at its bottom-left corner.

Tick order inside :meth:`DinoGame.step` (the port must keep this order):

1. Apply the action (start a jump, or set the duck flag).
2. Integrate the dino: ``vy -= g; y += vy``, landing when ``y <= 0``.
3. Scroll obstacles left by ``speed`` and drop those fully off screen.
4. Spawn a new obstacle if the required gap has opened up.
5. ``distance += speed``, then ``speed = min(MAX_SPEED, speed + ACCEL)``.
6. ``tick += 1``, then test collisions.

Actions: 0 = nothing, 1 = jump, 2 = duck. Ducking on the ground shrinks the hitbox.
Ducking in the air multiplies gravity by ``FAST_FALL`` (the original game's "speed drop").
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

# --- Time and world -------------------------------------------------------------------
TICKS_PER_SECOND = 60
WORLD_WIDTH = 600.0

# --- Dino -----------------------------------------------------------------------------
DINO_X = 50.0  # left edge; fixed, the world scrolls instead
DINO_W = 44.0
DINO_H = 47.0
DUCK_W = 59.0
DUCK_H = 26.0
GRAVITY = 0.6  # px / tick^2
JUMP_V = 10.5  # px / tick, upward
FAST_FALL = 3.0  # gravity multiplier while ducking in the air
# Hitboxes are shrunk by these margins on every side so near misses survive.
DINO_INSET = 3.0
OBSTACLE_INSET = 2.0

# --- Speed and score ------------------------------------------------------------------
START_SPEED = 6.0  # px / tick
MAX_SPEED = 13.0
ACCEL = 0.001  # px / tick^2, as in the original game
SCORE_PER_PX = 0.025

# --- Obstacles ------------------------------------------------------------------------
START_CLEAR_TICKS = 90  # nothing spawns during the first 1.5 s
# Smaller than the original game's 0.6. With 0.6 a one-line speed-aware rule survived to the
# tick cap on most unseen seeds in my first measurement, which left learning algorithms little to
# improve on. results/notes/dead_ends.json compares the two settings at the final cap.
GAP_COEFF = 0.5
MAX_GAP_COEFF = 1.5
MAX_GROUP = 3

# (unit width, height, base min gap, speed from which groups are allowed)
SMALL_CACTUS = (17.0, 35.0, 120.0, 4.0)
LARGE_CACTUS = (25.0, 50.0, 120.0, 7.0)
CACTI = (SMALL_CACTUS, LARGE_CACTUS)
BIRD_W = 46.0
BIRD_H = 30.0
BIRD_MIN_GAP = 150.0
BIRD_MIN_SPEED = 8.5
# Bottom edge of a bird at its three heights. Low must be jumped, mid must be ducked
# (or jumped), high passes over a standing dino.
BIRD_BOTTOMS = (5.0, 30.0, 60.0)

# Episodes are truncated here so a perfect agent cannot run forever. 36,000 ticks is ten
# simulated minutes and gives a maximum score of 11,087. The cap is a modelling
# choice, so it is a parameter of the game.
DEFAULT_MAX_TICKS = 36_000

# --- Observation normalisers ----------------------------------------------------------
OBS_DIST_NORM = WORLD_WIDTH
OBS_WIDTH_NORM = 100.0
OBS_HEIGHT_NORM = 60.0
OBS_Y_NORM = 80.0
OBS_DINO_Y_NORM = 100.0
OBS_VY_NORM = 15.0
OBSERVATION_SIZE = 9

NOOP, JUMP, DUCK = 0, 1, 2
ACTION_NAMES = ("nothing", "jump", "duck")


class Mulberry32:
    """Tiny 32-bit PRNG with a one-word state.

    JavaScript equivalent, used by the port::

        next() {
          this.state = (this.state + 0x6d2b79f5) >>> 0;
          let t = this.state;
          t = Math.imul(t ^ (t >>> 15), t | 1);
          t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
          return (t ^ (t >>> 14)) >>> 0;
        }
    """

    __slots__ = ("state",)

    def __init__(self, seed: int) -> None:
        self.state = seed & 0xFFFFFFFF

    def next_u32(self) -> int:
        self.state = (self.state + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.state
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        return (t ^ (t >> 14)) & 0xFFFFFFFF

    def random(self) -> float:
        """Uniform float in [0, 1). Dividing by 2**32 is exact in double precision."""
        return self.next_u32() / 4294967296.0

    def randint(self, lo: int, hi: int) -> int:
        """Uniform integer in [lo, hi], both ends included."""
        return lo + math.floor(self.random() * (hi - lo + 1))


@dataclass(frozen=True)
class EpisodeResult:
    """Outcome of one rollout."""

    seed: int
    ticks: int
    score: float
    distance: float
    died: bool  # False when the episode hit max_ticks instead


class DinoGame:
    """One seeded game.

    ``collisions=False`` turns death off and counts overlapping ticks in ``hit_ticks``.
    The parity fixtures use it to drive long runs through every speed band while still
    checking the collision test.
    """

    __slots__ = (
        "airborne",
        "collisions",
        "dead",
        "death_tick",
        "distance",
        "ducking",
        "gap",
        "hit_ticks",
        "max_ticks",
        "obstacles",
        "rng",
        "seed",
        "speed",
        "tick",
        "vy",
        "y",
    )

    def __init__(
        self,
        seed: int,
        max_ticks: int = DEFAULT_MAX_TICKS,
        collisions: bool = True,
    ) -> None:
        self.max_ticks = max_ticks
        self.collisions = collisions
        self.reset(seed)

    def reset(self, seed: int | None = None) -> None:
        if seed is not None:
            self.seed = seed
        self.rng = Mulberry32(self.seed)
        self.tick = 0
        self.distance = 0.0
        self.speed = START_SPEED
        self.y = 0.0
        self.vy = 0.0
        self.airborne = False
        self.ducking = False
        self.dead = False
        self.death_tick: int | None = None
        self.hit_ticks = 0
        self.gap = 0.0
        # Each obstacle is [x, y, w, h]; the list stays ordered by x, nearest first.
        self.obstacles: list[list[float]] = []

    # --- derived state ---------------------------------------------------------------
    @property
    def done(self) -> bool:
        return self.dead or self.tick >= self.max_ticks

    @property
    def score(self) -> float:
        return self.distance * SCORE_PER_PX

    # --- dynamics --------------------------------------------------------------------
    def step(self, action: int) -> bool:
        """Advance one tick. Returns True while the episode is still running."""
        if self.dead or self.tick >= self.max_ticks:
            return False

        if not self.airborne:
            if action == JUMP:
                self.vy = JUMP_V
                self.airborne = True
                self.ducking = False
            else:
                self.ducking = action == DUCK
        if self.airborne:
            vy = self.vy - (GRAVITY * FAST_FALL if action == DUCK else GRAVITY)
            y = self.y + vy
            if y <= 0.0:
                y = 0.0
                vy = 0.0
                self.airborne = False
            self.y = y
            self.vy = vy

        speed = self.speed
        obs = self.obstacles
        for o in obs:
            o[0] -= speed
        while obs and obs[0][0] + obs[0][2] < 0.0:
            obs.pop(0)

        if self.tick >= START_CLEAR_TICKS:
            if obs:
                last = obs[-1]
                if last[0] + last[2] + self.gap < WORLD_WIDTH:
                    self._spawn()
            else:
                self._spawn()

        self.distance += speed
        self.speed = min(MAX_SPEED, speed + ACCEL)
        self.tick += 1

        if self.obstacles and self._collides():
            if self.collisions:
                self.dead = True
                self.death_tick = self.tick
                return False
            self.hit_ticks += 1
        return self.tick < self.max_ticks

    def _spawn(self) -> None:
        """Add one obstacle at the right edge and draw the gap that must follow it.

        The order of RNG draws is part of the contract: kind, then group size (cacti) or
        height index (birds), then the gap.
        """
        rng = self.rng
        speed = self.speed
        kind = rng.randint(0, 2 if speed >= BIRD_MIN_SPEED else 1)
        if kind < 2:
            unit_w, h, min_gap, group_speed = CACTI[kind]
            size = rng.randint(1, MAX_GROUP)
            if size > 1 and speed < group_speed:
                size = 1
            w = unit_w * size
            y = 0.0
        else:
            w, h, min_gap = BIRD_W, BIRD_H, BIRD_MIN_GAP
            y = BIRD_BOTTOMS[rng.randint(0, 2)]
        lo = math.floor(w * speed + min_gap * GAP_COEFF + 0.5)
        hi = math.floor(lo * MAX_GAP_COEFF + 0.5)
        self.gap = float(rng.randint(lo, hi))
        self.obstacles.append([WORLD_WIDTH, y, w, h])

    def _collides(self) -> bool:
        """AABB test between the shrunken dino and each shrunken obstacle."""
        if self.ducking:
            dw, dh = DUCK_W, DUCK_H
        else:
            dw, dh = DINO_W, DINO_H
        d_left = DINO_X + DINO_INSET
        d_right = DINO_X + dw - DINO_INSET
        d_bottom = self.y + DINO_INSET
        d_top = self.y + dh - DINO_INSET
        for ox, oy, ow, oh in self.obstacles:
            if ox >= d_right:  # sorted by x, so nothing further right can overlap
                break
            if (
                ox + ow - OBSTACLE_INSET > d_left
                and ox + OBSTACLE_INSET < d_right
                and oy + oh - OBSTACLE_INSET > d_bottom
                and oy + OBSTACLE_INSET < d_top
            ):
                return True
        return False

    # --- views of the state ----------------------------------------------------------
    def upcoming(self) -> tuple[list[float] | None, list[float] | None]:
        """The next two obstacles the dino has not fully passed yet."""
        first: list[float] | None = None
        for o in self.obstacles:
            if o[0] + o[2] > DINO_X:
                if first is None:
                    first = o
                else:
                    return first, o
        return first, None

    def distance_to(self, obstacle: list[float]) -> float:
        """Gap between the dino's standing front edge and the obstacle, floored at 0."""
        return max(0.0, obstacle[0] - (DINO_X + DINO_W))

    def observe(self) -> list[float]:
        """Network inputs, each roughly in [0, 1]. Missing obstacles read as far away."""
        first, second = self.upcoming()
        if first is None:
            near_dist, near_w, near_h, near_y = 1.0, 0.0, 0.0, 0.0
        else:
            near_dist = min(1.0, self.distance_to(first) / OBS_DIST_NORM)
            near_w = first[2] / OBS_WIDTH_NORM
            near_h = first[3] / OBS_HEIGHT_NORM
            near_y = first[1] / OBS_Y_NORM
        far_dist = 1.0 if second is None else min(1.0, self.distance_to(second) / OBS_DIST_NORM)
        return [
            near_dist,
            near_w,
            near_h,
            near_y,
            far_dist,
            (self.speed - START_SPEED) / (MAX_SPEED - START_SPEED),
            self.y / OBS_DINO_Y_NORM,
            self.vy / OBS_VY_NORM,
            1.0,
        ]

    def snapshot(self) -> dict[str, object]:
        """Full state as plain data, for parity traces and debugging."""
        return {
            "tick": self.tick,
            "dead": self.dead,
            "distance": self.distance,
            "speed": self.speed,
            "y": self.y,
            "vy": self.vy,
            "ducking": self.ducking,
            "hit_ticks": self.hit_ticks,
            "rng_state": self.rng.state,
            "obstacles": [list(o) for o in self.obstacles],
        }


Policy = Callable[[DinoGame], int]


def rollout(seed: int, policy: Policy, max_ticks: int = DEFAULT_MAX_TICKS) -> EpisodeResult:
    """Play one episode with ``policy(game) -> action`` until death or the tick cap."""
    game = DinoGame(seed, max_ticks)
    step = game.step
    while step(policy(game)):
        pass
    return EpisodeResult(seed, game.tick, game.score, game.distance, game.dead)
