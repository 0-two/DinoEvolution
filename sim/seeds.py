"""Seed ranges, kept apart so that test seeds can never leak into training.

Evolution draws training episodes from ``0 .. TRAIN_SEED_LIMIT - 1``. Hand-tuned baselines
use ``TUNE_SEEDS``. Final generalisation tests use ``TEST_SEEDS`` and nothing else does.
"""

TRAIN_SEED_LIMIT = 900_000
TUNE_SEEDS: tuple[int, ...] = tuple(range(2_000_000, 2_000_050))
TEST_SEEDS: tuple[int, ...] = tuple(range(1_000_000, 1_000_100))
