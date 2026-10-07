"""Route B: smoothed coefficient, vertex, adjoint, then cone substitution.

Derivatives in s^0 are ambient product-rule derivatives. The cone point
is substituted only after those derivatives exist. This module does not
import the Route A Leibniz expansion.
"""

from __future__ import annotations

import numpy as np

from .profile_derivs import scaled_V_partial
from .row5a_test import q_and_derivatives
from .specialization import MASS, U_STAR, V_STAR


def _shift(U: np.ndarray, s: np.ndarray) -> np.ndarray:
    V = U - s
    return np.concatenate([U - U_STAR, V - V_STAR], axis=-1)


def _partial(U, s, delta, orders) -> np.ndarray:
    return scaled_V_partial(_shift(U, s), delta, orders)


def seed_density(U: np.ndarray, s: np.ndarray, delta: float, differentiated: bool = False) -> dict[str, np.ndarray]:
    """Real seed density after d_s0^2 d_t0^2, before -i and before J_t.

    The t-derivative has already been removed by the factor J_t. What remains
    is d_s0^2 of the vertex bracket in (U, s).
    """
    derivs = q_and_derivatives(U, s)
    terms = []
    for nu in range(4):
        orders = [0, 0, 0, 0]
        orders[nu] = 1
        F = _partial(U, s, delta, tuple(orders))
        orders[0] += 1
        Fs = -_partial(U, s, delta, tuple(orders))
        orders[0] += 1
        Fss = _partial(U, s, delta, tuple(orders))
        piece = (
            Fss * derivs["dU"][..., nu]
            + 2.0 * Fs * derivs["dU_ds0"][..., nu]
            + F * derivs["dU_ds0_2"][..., nu]
        )
        terms.append(0.5 * piece)
    B = _partial(U, s, delta, (0, 0, 0, 0))
    Bs = -_partial(U, s, delta, (1, 0, 0, 0))
    Bss = _partial(U, s, delta, (2, 0, 0, 0))
    mass = 0.5 * MASS**2 * (
        Bss * derivs["q"] + 2.0 * Bs * derivs["ds0"] + B * derivs["ds0_2"]
    )
    stacked = np.stack(terms + [mass], axis=0)
    if differentiated:
        raise RuntimeError("kappa scaling is applied by the caller through profile order, not here")
    return {"terms": stacked, "sum": np.sum(stacked, axis=0)}
