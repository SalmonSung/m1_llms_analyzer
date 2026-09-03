"""Small statistics helpers shared by the experiment analyses.

Kept here rather than imported from `experiment_figures` (which has private
equivalents) so the analysis code owns the numbers it reports: a figure module
should draw a record, not compute it.
"""

from __future__ import annotations

from typing import Callable, Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float],
    fn: Callable[[np.ndarray], float] = np.mean,
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """``(point, lo, hi)``: `fn` of the sample and its percentile bootstrap CI."""
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = float(fn(arr))
    if arr.size < 2:
        return point, point, point
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    stats = np.array([fn(arr[row]) for row in idx])
    lo, hi = np.percentile(stats, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return point, float(lo), float(hi)


def paired_bootstrap_diff(
    a: Sequence[float],
    b: Sequence[float],
    *,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """``mean(a - b)`` with a paired bootstrap CI; `a` and `b` are per-item, same length."""
    a_arr, b_arr = np.asarray(a, float), np.asarray(b, float)
    if a_arr.shape != b_arr.shape:
        raise ValueError("paired_bootstrap_diff needs two sequences of the same length.")
    diff = a_arr - b_arr
    return bootstrap_ci(diff, np.mean, n_boot=n_boot, seed=seed, alpha=alpha)


def trapezoid_area(xs: Sequence[float], ys: Sequence[float]) -> float:
    """Area under a curve given as points, by the trapezoid rule."""
    x, y = np.asarray(xs, float), np.asarray(ys, float)
    if x.size < 2:
        return 0.0
    return float(np.sum((x[1:] - x[:-1]) * (y[1:] + y[:-1]) / 2.0))
