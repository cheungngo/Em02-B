"""Compact even mollifier on R^8 and its logarithmic-scale derivative.

The profile identities are the Em02-A-v6 moment relations

    integral kappa = 0,
    integral h^alpha kappa(h) dh = |alpha| integral h^alpha rho(h) dh.

kappa(h) = -8 rho(h) - h · grad rho(h).
Scaling is rho_delta(h) = delta^{-8} rho(h/delta).
"""

from __future__ import annotations

import numpy as np
from scipy.integrate import quad

from .conventions import RELATIVE_DIM

_DIM = RELATIVE_DIM


def _raw_bump(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t, dtype=float)
    mask = np.abs(t) < 1.0
    u = t[mask]
    out[mask] = np.exp(-1.0 / (1.0 - u * u))
    return out


def _raw_bump_prime(t: np.ndarray) -> np.ndarray:
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t, dtype=float)
    mask = np.abs(t) < 1.0
    u = t[mask]
    bump = np.exp(-1.0 / (1.0 - u * u))
    out[mask] = bump * (-2.0 * u) / (1.0 - u * u) ** 2
    return out


_Z = quad(lambda t: float(_raw_bump(np.array([t]))[0]), -1.0, 1.0, epsabs=1e-12)[0]


def psi(t: np.ndarray) -> np.ndarray:
    """Normalized even 1D factor. Product over 8 axes is rho."""
    return _raw_bump(t) / _Z


def psi_prime(t: np.ndarray) -> np.ndarray:
    return _raw_bump_prime(t) / _Z


def rho(h: np.ndarray) -> np.ndarray:
    """Even compact profile on R^8 with unit mass. h shape (..., 8)."""
    h = np.asarray(h, dtype=float)
    return np.prod(psi(h), axis=-1)


def grad_rho(h: np.ndarray) -> np.ndarray:
    """Analytic gradient. Shape (..., 8)."""
    h = np.asarray(h, dtype=float)
    factors = psi(h)
    primes = psi_prime(h)
    # d/dh_i of product = prime_i * product_{j!=i} psi_j
    safe = np.where(np.abs(factors) > 0.0, factors, 1.0)
    total = np.prod(factors, axis=-1, keepdims=True)
    return total / safe * primes


def kappa(h: np.ndarray) -> np.ndarray:
    """kappa = -8 rho - h · grad rho. Zero mean by construction."""
    h = np.asarray(h, dtype=float)
    g = grad_rho(h)
    return -_DIM * rho(h) - np.sum(h * g, axis=-1)


def rho_delta(h: np.ndarray, delta: float) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    return delta ** (-_DIM) * rho(h / delta)


def kappa_delta(h: np.ndarray, delta: float) -> np.ndarray:
    h = np.asarray(h, dtype=float)
    return delta ** (-_DIM) * kappa(h / delta)


def moment_1d(weight_power: int, use_kappa_factor: bool = False) -> float:
    """Factored moment helper for a single axis.

    Returns integral t^p psi(t) dt, or the kappa-weighted 1D factor
    integral t^p (t psi'(t)) dt when use_kappa_factor is True.
    """

    def integrand(t: float) -> float:
        tt = np.array([t])
        if use_kappa_factor:
            return float((tt ** weight_power * tt * psi_prime(tt))[0])
        return float((tt ** weight_power * psi(tt))[0])

    val, _ = quad(integrand, -1.0, 1.0, epsabs=1e-12)
    return val


def check_profile_identities() -> dict[str, float]:
    """Exact product reductions of the Em02-A moment identities."""
    mass = moment_1d(0)  # should be 1
    # integral h_i psi'(related): integral t psi' = -1
    edge = moment_1d(0, use_kappa_factor=True)
    # second moment identity on one axis, other axes integrating to 1:
    # integral h_j^2 kappa = 2 integral h_j^2 rho
    m2 = moment_1d(2)
    # kappa = -8 rho - sum_k h_k psi'/psi * rho
    # integral h_j^2 kappa = -8 m2 - integral h_j^2 (h_j psi'/psi) rho
    #   - sum_{k!=j} integral h_j^2 rho * integral (h_k psi'/psi)
    # integral (t psi') = -1, and there are 7 such axes, plus the j axis.
    weighted = moment_1d(3, use_kappa_factor=False)  # placeholder, replaced below
    # integral t^2 * t psi'(t) dt = integral t^3 psi'
    weighted = quad(
        lambda t: float((np.array([t]) ** 3 * psi_prime(np.array([t])))[0]),
        -1.0,
        1.0,
        epsabs=1e-12,
    )[0]
    # other axes contribute m2 * (integral t psi') = m2 * (-1) each
    kappa_m2 = -_DIM * m2 - weighted - (_DIM - 1) * m2 * (-1.0)
    return {
        "mass": mass,
        "edge_integral_t_psi_prime": edge,
        "m2": m2,
        "kappa_m2": kappa_m2,
        "expected_kappa_m2": 2.0 * m2,
        "kappa_m2_residual": kappa_m2 - 2.0 * m2,
    }
