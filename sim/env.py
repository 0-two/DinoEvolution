"""Gymnasium wrapper around the same game, for the PPO baseline.

Observations are the nine network inputs, unchanged. Actions are the same three discrete
choices. The reward each tick is the score gained on that tick (speed times 0.025), so the
undiscounted return of an episode is exactly its score, the quantity the evolutionary
algorithms maximise. Dying ends the episode and gives no extra reward or penalty; reaching the
tick cap is reported as truncation, so the value function bootstraps past it.

Each ``reset`` draws a new game seed from the environment's own generator, below
``TRAIN_SEED_LIMIT``, so training never sees tuning or test seeds.

The optional ``jump_penalty`` is the same reward-shaping switch the evolutionary fitness has.
"""

from __future__ import annotations

from typing import Any, ClassVar

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from sim.game import DEFAULT_MAX_TICKS, JUMP, OBSERVATION_SIZE, SCORE_PER_PX, DinoGame
from sim.seeds import TRAIN_SEED_LIMIT


class DinoEnv(gym.Env[np.ndarray, int]):
    metadata: ClassVar[dict[str, Any]] = {"render_modes": []}

    def __init__(self, max_ticks: int = DEFAULT_MAX_TICKS, jump_penalty: float = 0.0) -> None:
        super().__init__()
        self.observation_space = spaces.Box(-2.0, 2.0, shape=(OBSERVATION_SIZE,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)
        self.max_ticks = max_ticks
        self.jump_penalty = jump_penalty
        self.game = DinoGame(0, max_ticks)

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        game_seed = options["game_seed"] if options and "game_seed" in options else None
        if game_seed is None:
            game_seed = int(self.np_random.integers(0, TRAIN_SEED_LIMIT))
        self.game.reset(game_seed)
        return self._obs(), {"game_seed": game_seed}

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        game = self.game
        before = game.distance
        jumped = action == JUMP and not game.airborne
        game.step(int(action))
        reward = (game.distance - before) * SCORE_PER_PX
        if jumped:
            reward -= self.jump_penalty
        terminated = game.dead
        truncated = not terminated and game.tick >= game.max_ticks
        info: dict[str, Any] = {}
        if terminated or truncated:
            info["episode_score"] = game.score  # the quantity compared across algorithms
        return self._obs(), reward, terminated, truncated, info

    def _obs(self) -> np.ndarray:
        return np.asarray(self.game.observe(), dtype=np.float32)
