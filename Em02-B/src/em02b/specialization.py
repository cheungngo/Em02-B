"""Declared simulated 5a specialization. Not an inherited physical map."""

from __future__ import annotations

import numpy as np

from .conventions import DatasetLabel

LABEL = DatasetLabel.SINGULAR_DIAGNOSTIC
NAME = "S1-isolated-5a-row"
MASS = 1.0
U_STAR = np.array([1.0, 1.0, 0.0, 0.0])
V_STAR = np.zeros(4)
S_STAR = U_STAR - V_STAR
T_STAR = 0.6 * np.array([1.0, 0.0, 1.0, 0.0])
SIGMA_U = 0.45
SIGMA_S = 0.45
SIGMA_T = 0.30
SPATIAL_SLOPE = 0.4


def specialization_record() -> dict:
    return {
        "label": LABEL.value,
        "name": NAME,
        "physical_admission": "not_claimed",
        "component": {"I": 1, "J": 1, "A_time": 1.0, "f": 1.0, "sigma": 1.0, "g": 1.0},
        "mass": MASS,
        "u_star": U_STAR.tolist(),
        "v_star": V_STAR.tolist(),
        "s_star": S_STAR.tolist(),
        "t_star": T_STAR.tolist(),
        "note": "Displayed mathematical data under a declared simulated component specialization.",
    }
