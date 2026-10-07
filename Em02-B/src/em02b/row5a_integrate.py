"""Integrate the reduced 5a row. Geometry utilities are shared; integrands are not."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from .quadrature import sobol_unit
from .row5a_route_a import leibniz_density
from .row5a_route_b import seed_density
from .row5a_test import J_t, support_box
from .specialization import U_STAR


def _pack(delta: float, u, s_radius, theta, phi):
    U = U_STAR + delta * u
    direction = np.stack([np.cos(theta), np.sin(theta) * np.cos(phi), np.sin(theta) * np.sin(phi)], axis=-1)
    s = np.concatenate([s_radius[..., None], s_radius[..., None] * direction], axis=-1)
    return U, s


def _weight(delta, s_radius, theta, jac) -> np.ndarray:
    # d^4U = delta^4 du; dR = S dS dn / 4pi; dn = sin(theta) dtheta dphi
    return (delta**4) * (s_radius / (4.0 * np.pi)) * np.sin(theta) * jac


def gauss_seed(delta: float, n_u: int, n_s: int, n_theta: int, n_phi: int, route: str) -> dict[str, float]:
    box = support_box(delta)
    nodes = []
    weights = []
    for _ in range(4):
        x, w = leggauss(n_u)
        nodes.append(x)
        weights.append(w)
    su, sw = leggauss(n_s)
    th, thw = leggauss(n_theta)
    s = 0.5 * (box["s_radius_high"] - box["s_radius_low"]) * su + 0.5 * (box["s_radius_high"] + box["s_radius_low"])
    sw = 0.5 * (box["s_radius_high"] - box["s_radius_low"]) * sw
    theta = 0.5 * box["theta_max"] * (th + 1.0)
    thw = 0.5 * box["theta_max"] * thw
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi
    grids = np.meshgrid(*nodes, s, theta, phi, indexing="ij")
    wgrids = np.meshgrid(*weights, sw, thw, np.ones(n_phi), indexing="ij")
    u = np.stack(grids[:4], axis=-1)
    jac = np.prod(wgrids[:4], axis=0) * wgrids[4] * wgrids[5] * dphi
    U, ss = _pack(delta, u, grids[4], grids[5], grids[6])
    density = seed_density(U, ss, delta)["sum"] if route == "B" else leibniz_density(U, ss, delta)["sum"]
    value = float(np.sum(density * _weight(delta, grids[4], grids[5], jac)))
    return {"route": route, "method": "gauss", "seed_real": value, "n": int(density.size)}


def sobol_seed(delta: float, m: int, seed: int, route: str) -> dict[str, float]:
    box = support_box(delta)
    pts = sobol_unit(7, m, seed=seed)
    u = 2.0 * pts[:, :4] - 1.0
    s = box["s_radius_low"] + pts[:, 4] * (box["s_radius_high"] - box["s_radius_low"])
    theta = pts[:, 5] * box["theta_max"]
    phi = pts[:, 6] * 2.0 * np.pi
    jac = (2.0**4) * (box["s_radius_high"] - box["s_radius_low"]) * box["theta_max"] * 2.0 * np.pi / pts.shape[0]
    U, ss = _pack(delta, u, s, theta, phi)
    density = leibniz_density(U, ss, delta)["sum"] if route == "A" else seed_density(U, ss, delta)["sum"]
    value = float(np.sum(density * _weight(delta, s, theta, jac)))
    return {"route": route, "method": "sobol", "seed_real": value, "n": int(pts.shape[0]), "scramble_seed": seed}


def complex_row(seed_real: float, jt: float | None = None) -> dict[str, float]:
    if jt is None:
        jt = J_t()
    real_before_i = jt * seed_real
    value = -1j * real_before_i
    return {
        "J_t": jt,
        "seed_real": seed_real,
        "real_before_minus_i": real_before_i,
        "row_real": float(value.real),
        "row_imag": float(value.imag),
    }
