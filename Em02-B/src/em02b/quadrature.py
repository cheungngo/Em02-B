"""Quadrature helpers. Sobol designs stay on complete 2^m point sets."""

from __future__ import annotations

import numpy as np
from scipy.stats import qmc


def sobol_unit(dimension: int, m: int, seed: int = 1) -> np.ndarray:
    if m < 1:
        raise ValueError("m must be positive; n_samples = 2**m")
    engine = qmc.Sobol(d=dimension, scramble=True, seed=seed)
    return engine.random_base2(m)


def replicate_spread(values: list[float]) -> dict[str, float]:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(arr.mean()),
        "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
        "n": float(len(arr)),
    }
