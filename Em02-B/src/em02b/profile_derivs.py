"""Analytic partials of the product profile, up to order 4 on one axis."""

from __future__ import annotations

import numpy as np
import sympy as sp

_T = sp.symbols("t")
_RAW = sp.exp(-1 / (1 - _T**2))
_DERIVS = [sp.lambdify(_T, sp.diff(_RAW, _T, k), "numpy") for k in range(5)]
from .profiles import _Z


def psi_derivative(order: int, t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t, dtype=float)
    mask = np.abs(t) < 1.0
    if np.any(mask):
        out[mask] = np.asarray(_DERIVS[order](t[mask]), dtype=float) / _Z
    return out


def scaled_V_partial(shift: np.ndarray, delta: float, v_orders: tuple[int, int, int, int]) -> np.ndarray:
    """Partial of rho_delta in the V slot. v_orders is the order on V^0..V^3.

    shift has shape (..., 8) and is the unscaled (U-U_*, V-V_*).
    """
    shift = np.asarray(shift, dtype=float)
    h = shift / delta
    order = sum(v_orders)
    values = np.ones(h.shape[:-1])
    for axis, deg in enumerate(v_orders):
        values = values * psi_derivative(deg, h[..., 4 + axis])
    for axis in range(4):
        values = values * psi_derivative(0, h[..., axis])
    return values * delta ** (-8 - order)
