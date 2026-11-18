"""Small statistics helpers, so the project does not need scipy for one confidence interval."""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence

# Two-sided 95% critical values of Student's t distribution, by degrees of freedom.
_T975 = {
    1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306,
    9: 2.262, 10: 2.228, 11: 2.201, 12: 2.179, 13: 2.160, 14: 2.145, 15: 2.131,
    16: 2.120, 17: 2.110, 18: 2.101, 19: 2.093, 20: 2.086, 25: 2.060, 30: 2.042,
    40: 2.021, 60: 2.000, 120: 1.980,
}  # fmt: skip


def t_critical(df: int) -> float:
    """95% two-sided critical value. Between table entries the next lower df is used, which
    errs on the side of a slightly wider interval."""
    if df < 1:
        return math.nan
    known = [k for k in _T975 if k <= df]
    return _T975[max(known)] if df < 120 else 1.96


def mean_ci95(values: Sequence[float]) -> tuple[float, float]:
    """Mean and half-width of the 95% confidence interval of the mean (Student's t).
    With a single value the half-width is NaN, because nothing can be said about spread."""
    n = len(values)
    mean = statistics.fmean(values)
    if n < 2:
        return mean, math.nan
    sem = statistics.stdev(values) / math.sqrt(n)
    return mean, t_critical(n - 1) * sem


def permutation_p(a: Sequence[float], b: Sequence[float], n: int = 20_000, seed: int = 0) -> float:
    """Two-sided permutation test for a difference in means.

    The labels of the pooled values are shuffled ``n`` times; the p-value is the share of
    shuffles whose absolute difference in means is at least the observed one (counting the
    observed arrangement itself, so it is never exactly zero). It assumes nothing about the shape
    of the score distributions, which here are far from normal: most runs score a few hundred and
    a few reach the maximum.
    """
    pooled = [*a, *b]
    observed = abs(statistics.fmean(a) - statistics.fmean(b))
    rng = random.Random(seed)
    na = len(a)
    hits = 1
    for _ in range(n):
        rng.shuffle(pooled)
        if abs(statistics.fmean(pooled[:na]) - statistics.fmean(pooled[na:])) >= observed - 1e-12:
            hits += 1
    return hits / (n + 1)
