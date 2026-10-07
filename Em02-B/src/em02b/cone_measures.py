"""Cone-radius measure and the two localization constants of Em02-A-v6.

    dR(T n) = (1/(4 pi)) T dT dn
    c_R = integral_0^inf t b_0(2 t^2) dt = (1/4) integral_0^inf b_0(s) ds
    c_47 = 1/(2 sqrt(2) pi) integral_0^inf t^2 b_0(t^2) dt
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import dblquad, quad

from .cutoffs import b0


def c_R(limit: float = 4.0) -> float:
    """Both formulas in Em02-A-v6 eq:transition-constants. They agree."""
    direct = quad(lambda t: t * float(b0(np.array([2.0 * t * t]))[0]), 0.0, np.sqrt(limit / 2.0), epsabs=1e-12)[0]
    by_s = 0.25 * quad(lambda s: float(b0(np.array([s]))[0]), 0.0, limit, epsabs=1e-12)[0]
    if abs(direct - by_s) > 1e-9:
        raise RuntimeError(f"c_R formulas disagree: {direct} vs {by_s}")
    return by_s


def c_47() -> float:
    pref = 1.0 / (2.0 * np.sqrt(2.0) * np.pi)
    integral = quad(
        lambda t: (t ** 2) * float(b0(np.array([t * t]))[0]),
        0.0,
        2.0,
        epsabs=1e-12,
    )[0]
    return pref * integral


def c_47_from_chart_integral() -> float:
    """Leading 47 coefficient as (1/(8 pi^2)) integral b_0(2 G^2 + |W|^2) dG d^2 W."""

    def integrand(radius: float, g: float) -> float:
        # polar in W: 2 pi R dR, G in R
        return float(b0(np.array([2.0 * g * g + radius * radius]))[0]) * radius

    # G from -2 to 2, R from 0 to 2 covers the support of b_0 (argument <= 4)
    integral, _ = dblquad(integrand, -2.0, 2.0, lambda _g: 0.0, lambda _g: 2.0, epsabs=1e-10)
    return (2.0 * np.pi * integral) / (8.0 * np.pi ** 2)


def cone_mass_5(eps: float, n_theta: int = 64, n_t: int = 256) -> float:
    """Numerically integrate (1/(4 pi)) integral T b_0(2 T^2 / eps^2) dT dn.

    The sphere integral is 1, so this equals c_R eps^2 for the exact b_0.
    """
    t_max = eps * np.sqrt(2.0)  # 2 T^2 / eps^2 <= 4
    ts = np.linspace(0.0, t_max, n_t)
    vals = ts * b0(2.0 * ts ** 2 / (eps * eps))
    radial = np.trapezoid(vals, ts)
    return radial  # sphere average of 1


def minkowski(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Mostly-plus signature eta = diag(1,-1,-1,-1)."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return x[..., 0] * y[..., 0] - np.sum(x[..., 1:] * y[..., 1:], axis=-1)


def euclidean_sq(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return np.sum(x * x, axis=-1)
