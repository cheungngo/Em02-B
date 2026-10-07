"""One-dimensional overlap moments for the product profile.

The registered Gaussian test factors in U, so this reduction is a named
evaluation of that same test, not a replacement family. A derivative in the
second slot is not treated as a total U derivative.
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .profile_derivs import psi_derivative
from .specialization import SIGMA_U, U_STAR


def overlap_interval(s_i: float, delta: float, u_star: float) -> tuple[float, float] | None:
    left = max(u_star - delta, s_i - delta)
    right = min(u_star + delta, s_i + delta)
    if left >= right:
        return None
    return left, right


def _scaled_deriv(order: int, x: float, delta: float) -> float:
    return float(psi_derivative(order, np.array([x / delta]))[0]) * delta ** (-1 - order)


def gaussian_weight(u: float, center: float, derivative: int) -> float:
    z = u - center
    g = np.exp(-(z ** 2) / (2.0 * SIGMA_U ** 2))
    if derivative == 0:
        return float(g)
    if derivative == 1:
        return float(-z / SIGMA_U ** 2 * g)
    if derivative == 2:
        return float((z ** 2 / SIGMA_U ** 4 - 1.0 / SIGMA_U ** 2) * g)
    raise ValueError("weight derivative order not used")


def overlap_moment(a: int, b: int, s_i: float, delta: float, u_star: float, weight_order: int = 0, n: int = 48) -> dict[str, float]:
    window = overlap_interval(s_i, delta, u_star)
    if window is None:
        return {"value": 0.0, "abs_estimate": 0.0, "quad_error": 0.0, "empty": 1.0}
    left, right = window
    xs, ws = np.polynomial.legendre.leggauss(n)
    u = 0.5 * (right - left) * xs + 0.5 * (right + left)
    w = 0.5 * (right - left) * ws
    vals = np.array([
        _scaled_deriv(a, float(ui) - u_star, delta)
        * _scaled_deriv(b, float(ui) - s_i, delta)
        * gaussian_weight(float(ui), u_star, weight_order)
        for ui in u
    ])
    value = float(np.sum(vals * w))
    return {"value": value, "abs_estimate": float(np.sum(np.abs(vals) * w)), "quad_error": 0.0, "empty": 0.0}


def parts_moment(s_i: float, delta: float, u_star: float) -> dict[str, float]:
    """Independent integration by parts for one second-slot derivative.

    Boundary terms vanish because both bumps vanish at the overlap endpoints.
    This is a comparison, not the adopted representation.
    """
    direct = overlap_moment(0, 1, s_i, delta, u_star, weight_order=0)
    transferred = overlap_moment(1, 0, s_i, delta, u_star, weight_order=0)
    weight_hit = overlap_moment(0, 0, s_i, delta, u_star, weight_order=1)
    return {
        "direct": direct["value"],
        "parts": -transferred["value"] - weight_hit["value"],
        "gap": direct["value"] + transferred["value"] + weight_hit["value"],
    }
