"""Isolated test, J_t, and the support box of the product profile."""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from .cone_measures import minkowski
from .specialization import S_STAR, SIGMA_T, SPATIAL_SLOPE, T_STAR, U_STAR


def isolation_check() -> dict[str, float | bool]:
    s = S_STAR
    t = T_STAR
    r = -t
    b_minus_z = s + t
    n = s[1:] / np.linalg.norm(s[1:])
    m = t[1:] / np.linalg.norm(t[1:])
    return {
        "s_future_null": bool(abs(float(minkowski(s, s))) < 1e-12 and s[0] > 0),
        "t_future_null": bool(abs(float(minkowski(t, t))) < 1e-12 and t[0] > 0),
        "r_nonzero": bool(np.linalg.norm(r) > 0),
        "b_minus_z_timelike": bool(float(minkowski(b_minus_z, b_minus_z)) > 0),
        "directions_nonparallel": bool(abs(float(np.dot(n, m))) < 1.0 - 1e-9),
        "q_b_minus_z": float(minkowski(b_minus_z, b_minus_z)),
        "surviving_row_name": "5a",
        "isolation_is_not_a_coefficient": True,
    }


def q_t(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    radial = np.exp(-np.sum((x - T_STAR) ** 2, axis=-1) / (2.0 * SIGMA_T**2))
    return radial * (1.0 + SPATIAL_SLOPE * x[..., 1])


def q_t_second_time(x: np.ndarray) -> np.ndarray:
    """Ambient d^2/d(x^0)^2 before cone restriction."""
    x = np.asarray(x, dtype=float)
    d0 = x[..., 0] - T_STAR[0]
    gauss = np.exp(-np.sum((x - T_STAR) ** 2, axis=-1) / (2.0 * SIGMA_T**2))
    second = gauss * (d0**2 / SIGMA_T**4 - 1.0 / SIGMA_T**2)
    return second * (1.0 + SPATIAL_SLOPE * x[..., 1])


def J_t(n_t: int = 24, n_theta: int = 16, n_phi: int = 24) -> float:
    """Cone integral of the ambient second time derivative. Must be nonzero."""
    ts, tw = leggauss(n_t)
    th, thw = leggauss(n_theta)
    T = 0.5 * (1.2 - 0.05) * ts + 0.5 * (1.2 + 0.05)
    tw = 0.5 * (1.2 - 0.05) * tw
    theta = 0.5 * np.pi * (th + 1.0)
    thw = 0.5 * np.pi * thw
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi
    TT, TH, PH = np.meshgrid(T, theta, phi, indexing="ij")
    TW, THW, _ = np.meshgrid(tw, thw, np.ones(n_phi), indexing="ij")
    direction = np.stack(
        [np.sin(TH) * np.cos(PH), np.sin(TH) * np.sin(PH), np.cos(TH)],
        axis=-1,
    )
    point = np.concatenate([TT[..., None], TT[..., None] * direction], axis=-1)
    weight = (TT / (4.0 * np.pi)) * np.sin(TH) * TW * THW * dphi
    return float(np.sum(q_t_second_time(point) * weight))


def support_box(delta: float) -> dict[str, float]:
    """Product-profile support: each relative coordinate is strictly inside (-delta, delta)."""
    return {
        "delta": delta,
        "u_half_width": delta,
        "v_half_width": delta,
        "s_radius_low": max(1e-6, 1.0 - 2.0 * delta),
        "s_radius_high": 1.0 + 2.0 * delta,
        "theta_max": float(min(np.pi, np.arcsin(min(1.0, 2.5 * delta)) + 0.05)),
        "injected_nodes": False,
    }


def q_and_derivatives(U: np.ndarray, s: np.ndarray) -> dict[str, np.ndarray]:
    from .specialization import SIGMA_S, SIGMA_U

    U = np.asarray(U, dtype=float)
    s = np.asarray(s, dtype=float)
    dU = U - U_STAR
    ds = s - S_STAR
    q = np.exp(-np.sum(dU**2, axis=-1) / (2.0 * SIGMA_U**2) - np.sum(ds**2, axis=-1) / (2.0 * SIGMA_S**2))
    dU_q = -dU / SIGMA_U**2 * q[..., None]
    ds0 = ds[..., 0]
    ds0_q = -ds0 / SIGMA_S**2 * q
    ds0_2_q = (-1.0 / SIGMA_S**2 + ds0**2 / SIGMA_S**4) * q
    dU_ds0_q = -ds0[..., None] / SIGMA_S**2 * dU_q
    dU_ds0_2_q = (-1.0 / SIGMA_S**2 + ds0**2 / SIGMA_S**4)[..., None] * dU_q
    return {"q": q, "dU": dU_q, "ds0": ds0_q, "ds0_2": ds0_2_q, "dU_ds0": dU_ds0_q, "dU_ds0_2": dU_ds0_2_q}
