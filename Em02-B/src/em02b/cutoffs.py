"""Fixed cutoff b_0 used by the coincidence partition and tip localization.

b_0 = 1 on [0, 1] and b_0 = 0 on [4, infinity), smooth in between.
"""

from __future__ import annotations

import numpy as np


def _exp_neg(u: np.ndarray) -> np.ndarray:
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u, dtype=float)
    mask = u > 0.0
    out[mask] = np.exp(-1.0 / u[mask])
    return out


def b0(s: np.ndarray) -> np.ndarray:
    """Smooth plateau: 1 on (-inf, 1], 0 on [4, inf)."""
    s = np.asarray(s, dtype=float)
    out = np.zeros_like(s, dtype=float)
    out = np.where(s <= 1.0, 1.0, out)
    mid = (s > 1.0) & (s < 4.0)
    if np.any(mid):
        t = (s[mid] - 1.0) / 3.0
        left = _exp_neg(t)
        right = _exp_neg(1.0 - t)
        out[mid] = right / (left + right)
    return out


def theta_eps(r_euclid_sq: np.ndarray, eps: float) -> np.ndarray:
    """theta_eps(r) = b_0(|r|_E^2 / eps^2). eps is independent of delta."""
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    return b0(np.asarray(r_euclid_sq, dtype=float) / (eps * eps))
