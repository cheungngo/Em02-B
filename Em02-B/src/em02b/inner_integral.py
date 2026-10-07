"""Factored inner integral at fixed s for the registered separable test."""

from __future__ import annotations

import numpy as np

from .overlap_moments import overlap_moment
from .specialization import MASS, SIGMA_S, S_STAR, U_STAR


def scale_factors(s: np.ndarray) -> tuple[float, float, float]:
    ds0 = float(s[0] - S_STAR[0])
    q = float(np.exp(-np.sum((s - S_STAR) ** 2) / (2.0 * SIGMA_S ** 2)))
    lam1 = -ds0 / SIGMA_S ** 2
    lam2 = -1.0 / SIGMA_S ** 2 + ds0 ** 2 / SIGMA_S ** 4
    return q, lam1, lam2


def moment(a: int, b: int, s: np.ndarray, delta: float, i: int, weight: int = 0) -> float:
    return overlap_moment(a, b, float(s[i]), delta, float(U_STAR[i]), weight)["value"]


def product(specs: list[tuple[int, int, int]], s: np.ndarray, delta: float) -> float:
    value = 1.0
    for i, (a, b, w) in enumerate(specs):
        value *= moment(a, b, s, delta, i, w)
    return value


def inner_terms(s: np.ndarray, delta: float) -> dict[str, np.ndarray | float]:
    s = np.asarray(s, dtype=float)
    q, lam1, lam2 = scale_factors(s)
    terms = []
    for nu in range(4):
        if nu == 0:
            F = product([(0, 1, 1), (0, 0, 0), (0, 0, 0), (0, 0, 0)], s, delta)
            Fs = -product([(0, 2, 1), (0, 0, 0), (0, 0, 0), (0, 0, 0)], s, delta)
            Fss = product([(0, 3, 1), (0, 0, 0), (0, 0, 0), (0, 0, 0)], s, delta)
        else:
            spec_F = [(0, 0, 0)] * 4
            spec_Fs = [(0, 0, 0)] * 4
            spec_Fss = [(0, 0, 0)] * 4
            spec_F[nu] = (0, 1, 1)
            spec_Fs[nu] = (0, 1, 1)
            spec_Fs[0] = (0, 1, 0)
            spec_Fss[nu] = (0, 1, 1)
            spec_Fss[0] = (0, 2, 0)
            F = product(spec_F, s, delta)
            Fs = -product(spec_Fs, s, delta)
            Fss = product(spec_Fss, s, delta)
        terms.append(0.5 * q * (Fss + 2.0 * lam1 * Fs + lam2 * F))
    B = product([(0, 0, 0)] * 4, s, delta)
    Bs = -product([(0, 1, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)], s, delta)
    Bss = product([(0, 2, 0), (0, 0, 0), (0, 0, 0), (0, 0, 0)], s, delta)
    terms.append(0.5 * MASS ** 2 * q * (Bss + 2.0 * lam1 * Bs + lam2 * B))
    values = np.array(terms)
    return {
        "terms": values,
        "sum": float(values.sum()),
        "cancellation": float(np.sum(np.abs(values)) / (abs(float(values.sum())) + 1e-30)),
    }
