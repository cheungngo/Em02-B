"""Scalar simulated P_5a amplitude, differentiated before any seed restriction.

This is not the inherited component map. f = sigma = 1 and the two A
factors are 1, so the displayed formula reduces to four seed derivatives
of the test amplitude. The non-tip density is the Em02-A-v6 factor
beta / (8 pi^2 |gamma|) on gamma < 0.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import sympy as sp
from scipy.integrate import dblquad


@dataclass(frozen=True)
class AdjointIntegral:
    value: float
    error_estimate: float
    gamma_window: tuple[float, float]
    radius_max: float


def scalar_p5a_expression() -> sp.Expr:
    s0, t0 = sp.symbols("s0 t0")
    psi = sp.exp(-((s0 - 1.5) ** 2 + t0 ** 2))
    # Ambient seed derivatives, then the expression is evaluated on the chart.
    return sp.diff(sp.diff(psi, s0), s0, t0, t0)


def scalar_p5a_callable():
    s0, t0 = sp.symbols("s0 t0")
    return sp.lambdify((s0, t0), scalar_p5a_expression(), "numpy")


def non_tip_integral(
    gamma_window: tuple[float, float] = (-1.2, -0.4),
    radius_max: float = 0.8,
    beta: float = 1.5,
) -> AdjointIntegral:
    """Integrate P_5a on upsilon = |omega|^2 / (4 gamma), gamma bounded away from 0."""
    amp = scalar_p5a_callable()
    g0, g1 = gamma_window

    def integrand(radius: float, gamma: float) -> float:
        upsilon = (radius ** 2) / (4.0 * gamma)
        # s0 tracks the b-a cone radius; t0 tracks the past a-z cone.
        value = float(amp(abs(gamma), -upsilon))
        density = beta / (8.0 * np.pi ** 2 * abs(gamma))
        return value * density * 2.0 * np.pi * radius

    value, err = dblquad(
        integrand,
        g0,
        g1,
        lambda _g: 0.0,
        lambda _g: radius_max,
        epsabs=1e-8,
    )
    return AdjointIntegral(float(value), float(err), gamma_window, radius_max)
