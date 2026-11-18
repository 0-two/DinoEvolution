"""Mechanics of the reference simulation."""

from __future__ import annotations

import pytest

from sim.game import (
    BIRD_BOTTOMS,
    BIRD_MIN_SPEED,
    DINO_X,
    DUCK,
    GAP_COEFF,
    JUMP,
    MAX_SPEED,
    NOOP,
    OBSERVATION_SIZE,
    START_CLEAR_TICKS,
    START_SPEED,
    WORLD_WIDTH,
    DinoGame,
    Mulberry32,
)


def run(game: DinoGame, actions: list[int]) -> None:
    for a in actions:
        game.step(a)


def put(game: DinoGame, x: float, y: float, w: float, h: float) -> None:
    """Place an obstacle by hand. Tests stay under START_CLEAR_TICKS so nothing else spawns."""
    game.obstacles.append([x, y, w, h])


def ground_hit_ticks(obstacle: tuple[float, float, float, float], action: int, ticks: int) -> bool:
    """Does a dino repeating ``action`` from the start die within ``ticks`` ticks?"""
    g = DinoGame(0)
    put(g, *obstacle)
    for _ in range(ticks):
        g.step(action)
    return g.dead


class TestRng:
    def test_same_seed_same_stream(self) -> None:
        a, b = Mulberry32(7), Mulberry32(7)
        assert [a.next_u32() for _ in range(50)] == [b.next_u32() for _ in range(50)]

    def test_seeds_differ(self) -> None:
        assert Mulberry32(1).next_u32() != Mulberry32(2).next_u32()

    def test_random_in_unit_interval(self) -> None:
        g = Mulberry32(3)
        values = [g.random() for _ in range(10_000)]
        assert all(0.0 <= v < 1.0 for v in values)
        assert 0.48 < sum(values) / len(values) < 0.52

    def test_randint_bounds_and_coverage(self) -> None:
        g = Mulberry32(5)
        values = {g.randint(2, 5) for _ in range(500)}
        assert values == {2, 3, 4, 5}

    def test_seed_is_truncated_to_32_bits(self) -> None:
        assert Mulberry32(2**32 + 9).state == 9


class TestDeterminism:
    def test_same_seed_and_actions_give_same_state(self) -> None:
        actions = [NOOP, JUMP, NOOP, DUCK] * 200
        a, b = DinoGame(11, collisions=False), DinoGame(11, collisions=False)
        run(a, actions)
        run(b, actions)
        assert a.snapshot() == b.snapshot()

    def test_seed_changes_obstacle_sequence(self) -> None:
        a, b = DinoGame(1, collisions=False), DinoGame(2, collisions=False)
        run(a, [NOOP] * 1500)
        run(b, [NOOP] * 1500)
        assert a.snapshot()["obstacles"] != b.snapshot()["obstacles"]

    def test_reset_restores_initial_state(self) -> None:
        g = DinoGame(4, collisions=False)
        run(g, [JUMP] + [NOOP] * 400)
        g.reset()
        assert g.snapshot() == DinoGame(4).snapshot()


class TestDino:
    def test_jump_arc(self) -> None:
        g = DinoGame(0)
        heights = []
        g.step(JUMP)
        heights.append(g.y)
        while g.airborne:
            g.step(NOOP)
            heights.append(g.y)
        assert 85.0 < max(heights) < 95.0
        assert 33 <= len(heights) <= 37
        assert heights[-1] == 0.0 and g.vy == 0.0
        # Rises then falls, never twice.
        peak = heights.index(max(heights))
        assert heights[: peak + 1] == sorted(heights[: peak + 1])

    def test_jump_ignored_in_the_air(self) -> None:
        a, b = DinoGame(0), DinoGame(0)
        run(a, [JUMP] + [NOOP] * 10)
        run(b, [JUMP] + [JUMP] * 10)
        assert a.y == b.y and a.vy == b.vy

    def test_duck_on_ground_sets_flag_and_release_clears_it(self) -> None:
        g = DinoGame(0)
        g.step(DUCK)
        assert g.ducking
        g.step(NOOP)
        assert not g.ducking

    def test_duck_in_air_is_a_fast_fall(self) -> None:
        slow, fast = DinoGame(0), DinoGame(0)
        run(slow, [JUMP] + [NOOP] * 8)
        run(fast, [JUMP] + [NOOP] * 3 + [DUCK] * 5)
        assert fast.y < slow.y
        assert not fast.ducking

    def test_fast_fall_lands_sooner(self) -> None:
        def airtime(fall: bool) -> int:
            g = DinoGame(0)
            g.step(JUMP)
            n = 1
            while g.airborne:
                g.step(DUCK if fall else NOOP)
                n += 1
            return n

        assert airtime(True) < airtime(False)


class TestSpeedAndScore:
    def test_speed_ramps_and_caps(self) -> None:
        g = DinoGame(0, collisions=False)
        run(g, [NOOP] * 100)
        assert g.speed == pytest.approx(START_SPEED + 100 * 0.001)
        run(g, [NOOP] * 8000)
        assert g.speed == MAX_SPEED

    def test_distance_is_sum_of_speeds(self) -> None:
        g = DinoGame(0, collisions=False)
        expected = 0.0
        speed = START_SPEED
        for _ in range(300):
            expected += speed
            speed = min(MAX_SPEED, speed + 0.001)
            g.step(NOOP)
        assert g.distance == expected
        assert g.score == pytest.approx(g.distance * 0.025)

    def test_truncation_at_max_ticks(self) -> None:
        g = DinoGame(0, max_ticks=50)
        run(g, [NOOP] * 200)
        assert g.tick == 50 and g.done and not g.dead
        assert g.step(NOOP) is False


class TestSpawning:
    def test_nothing_spawns_before_clear_ticks(self) -> None:
        g = DinoGame(0)
        run(g, [NOOP] * START_CLEAR_TICKS)
        assert g.obstacles == []
        g.step(NOOP)
        assert len(g.obstacles) == 1

    def test_first_obstacle_enters_at_the_right_edge(self) -> None:
        g = DinoGame(0)
        run(g, [NOOP] * (START_CLEAR_TICKS + 1))
        assert g.obstacles[0][0] == WORLD_WIDTH

    def test_gaps_are_respected(self) -> None:
        g = DinoGame(3, collisions=False)
        min_seen = 1e9
        for _ in range(4000):
            g.step(NOOP)
            for a, b in zip(g.obstacles, g.obstacles[1:], strict=False):
                min_seen = min(min_seen, b[0] - (a[0] + a[2]))
        # Smallest possible gap: unit width * speed + GAP_COEFF * 120 at the start speed.
        assert min_seen >= 17 * START_SPEED + GAP_COEFF * 120 - 1

    def test_birds_only_appear_at_speed(self) -> None:
        g = DinoGame(5, collisions=False)
        birds_before, birds_after = 0, 0
        seen: set[int] = set()
        for _ in range(9000):
            speed_before = g.speed
            g.step(NOOP)
            for o in g.obstacles:
                if o[3] == 30.0 and id(o) not in seen:
                    seen.add(id(o))
                    if speed_before < BIRD_MIN_SPEED:
                        birds_before += 1
                    else:
                        birds_after += 1
        assert birds_before == 0
        assert birds_after > 0

    def test_all_bird_heights_occur(self) -> None:
        g = DinoGame(8, collisions=False)
        bottoms: set[float] = set()
        for _ in range(36_000):
            g.step(NOOP)
            bottoms.update(o[1] for o in g.obstacles if o[3] == 30.0)
        assert bottoms == set(BIRD_BOTTOMS)

    def test_offscreen_obstacles_are_removed(self) -> None:
        g = DinoGame(0, collisions=False)
        run(g, [NOOP] * 3000)
        assert all(o[0] + o[2] >= 0 for o in g.obstacles)
        assert len(g.obstacles) <= 6


class TestCollision:
    CACTUS = (300.0, 0.0, 17.0, 35.0)

    def test_standing_dino_hits_cactus(self) -> None:
        assert ground_hit_ticks(self.CACTUS, NOOP, 60)

    def test_jump_clears_cactus(self) -> None:
        g = DinoGame(0)
        put(g, *self.CACTUS)
        # Obstacle reaches the dino about 34 ticks in; jump 16 ticks before that.
        run(g, [NOOP] * 18 + [JUMP] + [NOOP] * 60)
        assert not g.dead and g.tick == 79

    def test_late_jump_hits(self) -> None:
        g = DinoGame(0)
        put(g, *self.CACTUS)
        run(g, [NOOP] * 33 + [JUMP] + [NOOP] * 20)
        assert g.dead

    def test_death_tick_is_recorded_and_state_freezes(self) -> None:
        g = DinoGame(0)
        put(g, *self.CACTUS)
        run(g, [NOOP] * 80)
        assert g.dead and g.death_tick == g.tick
        frozen = g.snapshot()
        g.step(JUMP)
        assert g.snapshot() == frozen

    def test_ghost_mode_counts_instead_of_dying(self) -> None:
        g = DinoGame(0, collisions=False)
        put(g, *self.CACTUS)
        run(g, [NOOP] * 80)
        assert not g.dead and g.hit_ticks > 0

    def test_mid_bird_needs_a_duck(self) -> None:
        bird = (300.0, BIRD_BOTTOMS[1], 46.0, 30.0)
        assert ground_hit_ticks(bird, NOOP, 70)
        assert not ground_hit_ticks(bird, DUCK, 70)

    def test_low_bird_cannot_be_ducked(self) -> None:
        assert ground_hit_ticks((300.0, BIRD_BOTTOMS[0], 46.0, 30.0), DUCK, 70)

    def test_high_bird_passes_over_a_standing_dino(self) -> None:
        assert not ground_hit_ticks((300.0, BIRD_BOTTOMS[2], 46.0, 30.0), NOOP, 70)

    def test_hitbox_forgives_a_graze(self) -> None:
        # Overlaps the dino's right edge by 3 px: inside the 3 px dino margin plus 2 px
        # obstacle margin, so it must not count.
        g = DinoGame(0)
        put(g, DINO_X + 44.0 - 3.0 + g.speed, 0.0, 17.0, 35.0)
        g.step(NOOP)
        assert not g.dead


class TestObservation:
    def test_length_and_bias(self) -> None:
        obs = DinoGame(0).observe()
        assert len(obs) == OBSERVATION_SIZE and obs[-1] == 1.0

    def test_empty_road_reads_as_far(self) -> None:
        obs = DinoGame(0).observe()
        assert obs[0] == 1.0 and obs[4] == 1.0
        assert obs[1:4] == [0.0, 0.0, 0.0]

    def test_values_are_roughly_unit_range_over_a_run(self) -> None:
        g = DinoGame(2, collisions=False)
        lo, hi = 9.0, -9.0
        for i in range(6000):
            g.step(JUMP if i % 40 == 0 else NOOP)
            for v in g.observe():
                lo, hi = min(lo, v), max(hi, v)
        assert lo >= -1.0 and hi <= 1.0 + 1e-9

    def test_next_obstacle_is_the_one_ahead(self) -> None:
        g = DinoGame(0)
        put(g, 200.0, 0.0, 17.0, 35.0)
        put(g, 400.0, 30.0, 46.0, 30.0)
        obs = g.observe()
        assert obs[0] == pytest.approx((200.0 - 94.0) / WORLD_WIDTH)
        assert obs[4] == pytest.approx((400.0 - 94.0) / WORLD_WIDTH)
        assert obs[3] == pytest.approx(0.0)

    def test_passed_obstacle_is_skipped(self) -> None:
        g = DinoGame(0)
        put(g, DINO_X - 30.0, 0.0, 17.0, 35.0)  # fully behind the dino
        put(g, 250.0, 0.0, 25.0, 50.0)
        assert g.observe()[1] == pytest.approx(25.0 / 100.0)

    def test_overlapping_obstacle_reads_zero_distance(self) -> None:
        g = DinoGame(0)
        put(g, DINO_X + 10.0, 0.0, 17.0, 35.0)
        assert g.observe()[0] == 0.0
