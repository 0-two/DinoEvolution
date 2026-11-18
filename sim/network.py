"""The policy network shared by the GA and CMA-ES: 9 inputs, 8 tanh hidden units, 3 outputs.

The layout is identical to ``web/src/evolve/network.ts``. A genome is a flat vector of 96
numbers: the first layer row by row (hidden unit 0's nine weights, then unit 1's, ...) followed
by the second layer row by row. The ninth input is a constant 1, so its first-layer column is
the hidden bias. The outputs have no bias. The action is the index of the largest output; ties
go to the lowest index, which is "nothing".

Numpy's matrix product and the browser's loop add terms in different orders, so the two can
differ in the last bit. ``sim/tests/golden/network_vectors.json`` pins both to within 1e-12.
"""

from __future__ import annotations

import numpy as np

from sim.game import OBSERVATION_SIZE

INPUTS = OBSERVATION_SIZE
HIDDEN = 8
OUTPUTS = 3
GENOME_LENGTH = HIDDEN * INPUTS + OUTPUTS * HIDDEN


class FixedNetwork:
    """Feed-forward 9-8-3 network built from a flat genome."""

    __slots__ = ("w1", "w2")

    def __init__(self, genome: np.ndarray | list[float]) -> None:
        g = np.asarray(genome, dtype=np.float64)
        if g.shape != (GENOME_LENGTH,):
            raise ValueError(f"expected {GENOME_LENGTH} weights, got shape {g.shape}")
        self.w1 = g[: HIDDEN * INPUTS].reshape(HIDDEN, INPUTS)
        self.w2 = g[HIDDEN * INPUTS :].reshape(OUTPUTS, HIDDEN)

    def forward(self, obs: list[float] | np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Hidden activations and raw outputs."""
        hidden = np.tanh(self.w1 @ np.asarray(obs, dtype=np.float64))
        return hidden, self.w2 @ hidden

    def act(self, obs: list[float] | np.ndarray) -> int:
        """Index of the largest output. ``argmax`` returns the first maximum on ties."""
        return int(np.argmax(self.w2 @ np.tanh(self.w1 @ np.asarray(obs, dtype=np.float64))))
