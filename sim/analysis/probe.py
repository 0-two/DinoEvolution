"""What does a trained policy do when an obstacle approaches?

A probe puts the dino on the ground, places one obstacle at a chosen distance ahead, sets the
game speed, and asks the policy for an action. Sweeping distance and speed gives a map of
decisions. The map is a property of the policy alone, because the state is set by hand and no
game is played. It shows where each algorithm starts its jump, and whether that start moves
with speed as a good policy's should: faster speed covers more ground per tick, so the jump has
to begin earlier.
"""

from __future__ import annotations

from typing import Any

from sim.game import DINO_W, DINO_X, MAX_SPEED, START_SPEED, DinoGame
from sim.genome_io import load_game_policy

# (name, y of bottom edge, width, height). The right action differs: cactus needs a jump,
# a mid-height bird needs a duck.
OBSTACLES = {
    "small cactus": (0.0, 17.0, 35.0),
    "mid bird": (30.0, 46.0, 30.0),
}
ACTION_WORDS = ("nothing", "jump", "duck")
DISTANCES = list(range(0, 405, 5))  # gap between the dino's front edge and the obstacle, px
SPEEDS = [START_SPEED + i * 0.5 for i in range(int((MAX_SPEED - START_SPEED) / 0.5) + 1)]


def decision_map(payload: dict[str, Any], obstacle: str) -> list[list[int]]:
    """Rows are speeds (slowest first), columns are distances; values are actions 0, 1, 2."""
    act = load_game_policy(payload)
    y, w, h = OBSTACLES[obstacle]
    rows = []
    for speed in SPEEDS:
        row = []
        for d in DISTANCES:
            game = DinoGame(0, 1000, collisions=False)
            game.speed = speed
            game.obstacles = [[DINO_X + DINO_W + d, y, w, h]]
            row.append(int(act(game)))
        rows.append(row)
    return rows


def first_jump_distance(grid: list[list[int]], action: int = 1) -> list[int | None]:
    """For each speed, the largest distance at which the policy already chooses ``action``.

    Scanning from far to near, the first column with that action is where the policy starts
    reacting. None if it never does.
    """
    out: list[int | None] = []
    for row in grid:
        hit = None
        for col in range(len(row) - 1, -1, -1):
            if row[col] == action:
                hit = DISTANCES[col]
                break
        out.append(hit)
    return out
