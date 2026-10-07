"""Outer cone integral of a stable inner value. Past-null t is already in J_t."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from .inner_integral import inner_terms
from .quadrature import sobol_unit
from .row5a_test import support_box


def _H(delta: float, S: float, theta: float, phi: float) -> float:
    direction = np.array([np.cos(theta), np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)])
    s = np.concatenate([[S], S * direction])
    return inner_terms(s, delta)["sum"]


def gauss_outer(delta: float, n_s: int = 8, n_theta: int = 8, n_phi: int = 12) -> float:
    box = support_box(delta)
    su, sw = leggauss(n_s)
    th, thw = leggauss(n_theta)
    S = 0.5 * (box["s_radius_high"] - box["s_radius_low"]) * su + 0.5 * (box["s_radius_high"] + box["s_radius_low"])
    sw = 0.5 * (box["s_radius_high"] - box["s_radius_low"]) * sw
    theta = 0.5 * box["theta_max"] * (th + 1.0)
    thw = 0.5 * box["theta_max"] * thw
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi
    total = 0.0
    for i, Sv in enumerate(S):
        for j, thv in enumerate(theta):
            for phv in phi:
                total += _H(delta, float(Sv), float(thv), float(phv)) * (Sv / (4.0 * np.pi)) * np.sin(thv) * sw[i] * thw[j] * dphi
    return float(total)


def sobol_outer(delta: float, m: int = 8, seed: int = 2) -> float:
    box = support_box(delta)
    pts = sobol_unit(3, m, seed=seed)
    S = box["s_radius_low"] + pts[:, 0] * (box["s_radius_high"] - box["s_radius_low"])
    theta = pts[:, 1] * box["theta_max"]
    phi = pts[:, 2] * 2.0 * np.pi
    jac = (box["s_radius_high"] - box["s_radius_low"]) * box["theta_max"] * 2.0 * np.pi / pts.shape[0]
    total = 0.0
    for i in range(pts.shape[0]):
        total += _H(delta, float(S[i]), float(theta[i]), float(phi[i])) * (S[i] / (4.0 * np.pi)) * np.sin(theta[i]) * jac
    return float(total)
