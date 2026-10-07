"""Route A: Leibniz expansion in the relative slot, derived from V = U - s.

This module does not import Route B. The binomial signs are the chain-rule
signs of d/ds^0 = -d/dV^0 acting on B, together with derivatives that land on Q.
"""

from __future__ import annotations

import numpy as np

from .profile_derivs import scaled_V_partial
from .row5a_test import q_and_derivatives
from .specialization import MASS, U_STAR, V_STAR


def _shift(U: np.ndarray, s: np.ndarray) -> np.ndarray:
    return np.concatenate([U - U_STAR, (U - s) - V_STAR], axis=-1)


def leibniz_density(U: np.ndarray, s: np.ndarray, delta: float) -> dict[str, np.ndarray]:
    derivs = q_and_derivatives(U, s)
    terms = []
    for nu in range(4):
        acc = 0.0
        for r in range(3):
            orders = [r + 1 if nu == 0 else r, 0, 0, 0]
            if nu != 0:
                orders[nu] = 1
            coeff = scaled_V_partial(_shift(U, s), delta, tuple(orders))
            if r == 0:
                q_part = derivs["dU_ds0_2"][..., nu]
            elif r == 1:
                q_part = derivs["dU_ds0"][..., nu]
            else:
                q_part = derivs["dU"][..., nu]
            acc = acc + 0.5 * (2 if r == 1 else 1) * ((-1.0) ** r) * coeff * q_part
        terms.append(acc)
    mass = 0.0
    for r in range(3):
        coeff = scaled_V_partial(_shift(U, s), delta, (r, 0, 0, 0))
        q_part = (derivs["ds0_2"], derivs["ds0"], derivs["q"])[r]
        mass = mass + 0.5 * MASS**2 * (2 if r == 1 else 1) * ((-1.0) ** r) * coeff * q_part
    stacked = np.stack(terms + [mass], axis=0)
    return {"terms": stacked, "sum": np.sum(stacked, axis=0)}
