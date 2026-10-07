"""Weighted atomic 5a cone integral with two independent quadratures.

The passive cone is not an arbitrary box. It is integrated against
zeta_T(T)=exp(-T), whose moment integral_0^inf T exp(-T) dT equals 1,
and against the uniform sphere probability. The resulting factor is

    I = integral S c(S, n) dn dS / (4 pi).

Route B evaluates the smoothed atomic coefficient. Route A evaluates the
translated profile and does not call Route B. For dataset S1 the
integrands agree; the quadratures do not share nodes.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .profiles import kappa, rho
from .quadrature import sobol_unit
from .simulated_inputs import AtomicSite


@dataclass(frozen=True)
class AtomicIntegral:
    method: str
    route: str
    value: float
    n_samples: int
    delta: float
    differentiated: bool


def future_null_site() -> AtomicSite:
    """b-a = (1,1,0,0), future null. A timelike trial site is rejected upstream."""
    return AtomicSite(np.array([1.0, 1.0, 0.0, 0.0]), np.zeros(4))


def _direction(theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Spatial direction with polar axis along (1,0,0)."""
    theta = np.asarray(theta, dtype=float)
    phi = np.asarray(phi, dtype=float)
    return np.stack(
        [
            np.cos(theta),
            np.sin(theta) * np.cos(phi),
            np.sin(theta) * np.sin(phi),
        ],
        axis=-1,
    )


def translated_shift(site: AtomicSite, s: np.ndarray, n: np.ndarray) -> np.ndarray:
    """Relative slot (U-U_*, V-V_*) at fixed U=U_*. Shape (..., 8)."""
    s = np.asarray(s, dtype=float)
    n = np.asarray(n, dtype=float)
    direction = np.concatenate([np.ones(n.shape[:-1] + (1,)), n], axis=-1)
    mismatch = site.b_minus_a - s[..., None] * direction
    zeros = np.zeros(mismatch.shape)
    return np.concatenate([zeros, mismatch], axis=-1)


def _profile_on_cone(site: AtomicSite, s, n, delta: float, differentiated: bool) -> np.ndarray:
    shift = translated_shift(site, s, n)
    raw = kappa(shift / delta) if differentiated else rho(shift / delta)
    return site.tensor_scale * (delta ** (-8)) * raw


def gauss_atomic_integral(
    site: AtomicSite,
    delta: float = 0.35,
    differentiated: bool = False,
    n_s: int = 24,
    n_theta: int = 16,
    n_phi: int = 24,
    route: str = "B",
) -> AtomicIntegral:
    """Tensor Gauss/trapezoid rule on the cap that contains the mollifier support."""
    beta = float(site.b_minus_a[0])
    s_nodes, s_w = np.polynomial.legendre.leggauss(n_s)
    s = 0.5 * (2.0 * delta) * s_nodes + beta
    s_w = 0.5 * (2.0 * delta) * s_w
    theta_max = min(np.pi, np.arcsin(min(1.0, 1.5 * delta / max(beta - delta, 1e-6))) + 0.05)
    th_nodes, th_w = np.polynomial.legendre.leggauss(n_theta)
    theta = 0.5 * theta_max * (th_nodes + 1.0)
    th_w = 0.5 * theta_max * th_w
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi
    ss, tt, pp = np.meshgrid(s, theta, phi, indexing="ij")
    sw, tw, _pw = np.meshgrid(s_w, th_w, np.ones(n_phi), indexing="ij")
    n = _direction(tt, pp)
    coeff = _profile_on_cone(site, ss, n, delta, differentiated)
    weight = (ss / (4.0 * np.pi)) * np.sin(tt) * sw * tw * dphi
    value = float(np.sum(coeff * weight))
    return AtomicIntegral("gauss_cap", route, value, int(coeff.size), delta, differentiated)


def sobol_atomic_integral(
    site: AtomicSite,
    delta: float = 0.35,
    differentiated: bool = False,
    m: int = 12,
    seed: int = 7,
    route: str = "A",
) -> AtomicIntegral:
    """Scrambled Sobol rule on the same cap. Complete 2^m design only."""
    beta = float(site.b_minus_a[0])
    theta_max = min(np.pi, np.arcsin(min(1.0, 1.5 * delta / max(beta - delta, 1e-6))) + 0.05)
    pts = sobol_unit(3, m, seed=seed)
    s = beta - delta + pts[:, 0] * (2.0 * delta)
    theta = pts[:, 1] * theta_max
    phi = pts[:, 2] * 2.0 * np.pi
    n = _direction(theta, phi)
    coeff = _profile_on_cone(site, s, n, delta, differentiated)
    jac = (2.0 * delta) * theta_max * (2.0 * np.pi)
    weight = (s / (4.0 * np.pi)) * np.sin(theta) * jac / pts.shape[0]
    value = float(np.sum(coeff * weight))
    return AtomicIntegral("sobol_cap", route, value, int(pts.shape[0]), delta, differentiated)


def route_pair(delta: float = 0.35, differentiated: bool = False) -> dict[str, AtomicIntegral]:
    site = future_null_site()
    return {
        "B": gauss_atomic_integral(site, delta=delta, differentiated=differentiated, route="B"),
        "A": sobol_atomic_integral(site, delta=delta, differentiated=differentiated, route="A"),
    }
