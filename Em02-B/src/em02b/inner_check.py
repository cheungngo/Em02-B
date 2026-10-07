"""Fixed-s four-dimensional check of the overlap reduction, on the overlap box only."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from .overlap_moments import overlap_interval
from .row5a_route_b import seed_density
from .specialization import U_STAR


def tensor_inner(s: np.ndarray, delta: float, n: int = 12) -> float:
    s = np.asarray(s, dtype=float)
    intervals = [overlap_interval(float(s[i]), delta, float(U_STAR[i])) for i in range(4)]
    if any(item is None for item in intervals):
        return 0.0
    nodes = []
    weights = []
    for left, right in intervals:
        x, w = leggauss(n)
        nodes.append(0.5 * (right - left) * x + 0.5 * (right + left))
        weights.append(0.5 * (right - left) * w)
    grids = np.meshgrid(*nodes, indexing="ij")
    U = np.stack(grids, axis=-1)
    ss = np.broadcast_to(s, U.shape[:-1] + (4,))
    density = seed_density(U, ss, delta)["sum"]
    weight = np.ones(grids[0].shape)
    for k, w in enumerate(weights):
        shape = [1, 1, 1, 1]
        shape[k] = n
        weight = weight * w.reshape(shape)
    return float(np.sum(density * weight))
