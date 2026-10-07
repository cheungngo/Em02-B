"""Support predicates and the non-core 5z counterexample geometry.

These are exact scalar-support tests from Em02-A-v6. They do not
assign a physical coefficient.
"""

from __future__ import annotations

import numpy as np

from .cone_measures import euclidean_sq, minkowski


def null_future(v: np.ndarray, tol: float = 1e-9) -> bool:
    v = np.asarray(v, dtype=float)
    return abs(float(minkowski(v, v))) <= tol and float(v[0]) > 0.0


def null_past(v: np.ndarray, tol: float = 1e-9) -> bool:
    v = np.asarray(v, dtype=float)
    return abs(float(minkowski(v, v))) <= tol and float(v[0]) < 0.0


def isolation_5a_patch(beta: float, epsilon: float, n: np.ndarray, m: np.ndarray) -> dict[str, bool]:
    """Proposition (exact support isolation), item (i).

    b - a = beta ell, r = -epsilon (1, m), m != n.
    Surviving scalar row: 5a.
    """
    n = np.asarray(n, dtype=float)
    m = np.asarray(m, dtype=float)
    ell = np.array([1.0, *n])
    r = -epsilon * np.array([1.0, *m])
    b_minus_a = beta * ell
    b_minus_z = b_minus_a - r
    q_bz = float(minkowski(b_minus_z, b_minus_z))
    parallel = np.linalg.norm(n - m) < 1e-12
    return {
        "q_b_minus_z_positive": q_bz > 0.0,
        "excludes_47": q_bz > 0.0,
        "excludes_5z": q_bz > 0.0,
        "excludes_response": float(euclidean_sq(r)) > 0.0,
        "surviving_row": "5a",
        "nonparallel": not parallel,
        "incidence_consistent": True,
    }


def isolation_47_patch(beta: float, gamma: float, n: np.ndarray, m: np.ndarray) -> dict[str, bool]:
    """Item (ii): b-a and b-z future null and nonparallel. Surviving row: 47."""
    n = np.asarray(n, dtype=float)
    m = np.asarray(m, dtype=float)
    s = beta * np.array([1.0, *n])
    t = gamma * np.array([1.0, *m])
    r = s - t
    nonparallel = abs(float(np.dot(n, m))) < 1.0 - 1e-9
    return {
        "s_future_null": null_future(s),
        "t_future_null": null_future(t),
        "r_spacelike": float(minkowski(r, r)) < 0.0,
        "excludes_5": abs(float(minkowski(r, r))) > 1e-8,
        "excludes_response": float(euclidean_sq(r)) > 0.0,
        "nonparallel": nonparallel,
        "surviving_row": "47",
    }


def counterexample_5z_noncore(beta: float = 2.0, alpha: float = 0.7, u0: float = 0.15, gamma: float = 0.4) -> dict[str, float]:
    """Proposition (failure of automatic internal-core localization).

    Simultaneous crossed cones on the 5z seed with q_R = 4 beta u0 > 0 fixed.
    """
    if not (0.0 < alpha < beta and 0.0 < gamma < beta and u0 > 0.0):
        raise ValueError("chart bounds violated")
    v = ((beta - alpha) / beta) * u0
    xi_sq = (4.0 * alpha * (beta - alpha) / beta) * u0
    q1 = 4.0 * alpha * v - xi_sq
    q2 = 4.0 * (beta - alpha) * (u0 - v) - xi_sq
    qR = 4.0 * beta * u0
    upsilon = ((beta - gamma) / beta) * u0
    omega_sq = (4.0 * gamma * (beta - gamma) / beta) * u0
    return {
        "q1": q1,
        "q2": q2,
        "qR": qR,
        "upsilon": upsilon,
        "omega_sq": omega_sq,
        "on_5z_seed": abs(q1) + abs(q2) < 1e-12 and qR > 0.0 and omega_sq > 0.0,
    }


def sector_zero(row: str, sector: str) -> bool:
    from .conventions import SUPPORT_TABLE

    return SUPPORT_TABLE[row][sector] == "0"
