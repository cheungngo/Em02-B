"""Independent Route A / Route B evaluators on a scalar seed.

Route B convolves the crossed array, then integrates the seed measure.
Route A integrates the seed against translated profile derivatives.
They share the mathematical identity and not the computational path.

The seed used here is the radial 5a cone at fixed passive coordinates,
which is enough to audit agreement, isolation, and scale dependence.
It is not the seven-row physical pairing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cone_measures import minkowski
from .profiles import kappa, rho
from .simulated_inputs import AtomicSite


def _sphere_points(n_side: int = 6, extra: np.ndarray | None = None) -> np.ndarray:
    """Deterministic spherical design, not a thinned Sobol set."""
    k = np.arange(n_side * n_side)
    ga = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (k + 0.5) / (n_side * n_side)
    rad = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    theta = ga * k
    pts = np.stack([np.cos(theta) * rad, np.sin(theta) * rad, z], axis=1)
    if extra is not None:
        pts = np.vstack([np.asarray(extra, dtype=float).reshape(1, 3), pts])
    return pts


@dataclass
class ScalarPairing:
    route: str
    value: float
    n_samples: int
    row: str


def _sample_grid(n_radial: int, n_side: int, site: AtomicSite) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    spatial = site.b_minus_a[1:]
    norm = np.linalg.norm(spatial)
    direction = spatial / norm if norm > 0.0 else np.array([1.0, 0.0, 0.0])
    radius = float(site.b_minus_a[0])
    ss = np.unique(np.concatenate([np.linspace(0.05, 1.2, n_radial), [radius]]))
    ts = np.linspace(0.05, 1.2, n_radial)
    dirs_n = _sphere_points(n_side, extra=direction)
    dirs_m = _sphere_points(max(2, n_side // 2))
    return ss, ts, dirs_n, dirs_m


def route_b_atomic_5a(
    site: AtomicSite,
    delta: float,
    differentiated: bool = False,
    n_radial: int = 24,
    n_side: int = 4,
) -> ScalarPairing:
    """Seed integral of the already smoothed atomic coefficient.

    Passive a, y are fixed so U = y - a = U_star. Then V = U_star - S n,
    and the atomic center contributes when S n = U_star - V_star = b-a.
    """
    profile = kappa if differentiated else rho
    ss, ts, dirs_n, dirs_m = _sample_grid(n_radial, n_side, site)
    # weights: dS dT dn dm * S * T / (16 pi^2); sphere weights are uniform
    ds = ss[1] - ss[0]
    dt = ts[1] - ts[0]
    wn = 4.0 * np.pi / len(dirs_n)
    wm = 4.0 * np.pi / len(dirs_m)
    total = 0.0
    count = 0
    target = site.b_minus_a
    for s in ss:
        for n in dirs_n:
            direction = np.array([1.0, *n])
            # V = U_star - S n_dir, relative slot (U, V)
            v = site.u_star - s * direction
            shift = np.concatenate([np.zeros(4), v - site.v_star])
            coeff = site.tensor_scale * (delta ** (-8)) * float(profile(shift / delta))
            for t in ts:
                for _m in dirs_m:
                    weight = (s * t) / (16.0 * np.pi ** 2) * ds * dt * wn * wm
                    total += weight * coeff
                    count += 1
    return ScalarPairing("B", total, count, "5a")


def route_a_atomic_5a(
    site: AtomicSite,
    delta: float,
    differentiated: bool = False,
    n_radial: int = 24,
    n_side: int = 4,
) -> ScalarPairing:
    """Difference-pushforward pairing. Does not call the Route B evaluator.

    For S1 the v-pairing is the translated profile at pi(w) - (U_star, V_star).
    """
    profile = kappa if differentiated else rho
    ss, ts, dirs_n, dirs_m = _sample_grid(n_radial, n_side, site)
    ds = ss[1] - ss[0]
    dt = ts[1] - ts[0]
    wn = 4.0 * np.pi / len(dirs_n)
    wm = 4.0 * np.pi / len(dirs_m)
    total = 0.0
    count = 0
    for s in ss:
        for n in dirs_n:
            direction = np.array([1.0, *n])
            v = site.u_star - s * direction
            # translated derivative of the test profile, evaluated directly
            h = np.concatenate([np.zeros(4), v - site.v_star]) / delta
            translated = site.tensor_scale * (delta ** (-8)) * float(profile(h))
            for t in ts:
                for _m in dirs_m:
                    weight = (s * t) / (16.0 * np.pi ** 2) * ds * dt * wn * wm
                    total += weight * translated
                    count += 1
    return ScalarPairing("A", total, count, "5a")


def incidence_residual(site: AtomicSite) -> float:
    """How far U_star - V_star is from being future null. Zero is the contributing locus."""
    d = site.b_minus_a
    return float(minkowski(d, d))


def cancellation_indicator(row_values: dict[str, complex], floor: float = 1e-30) -> float:
    num = sum(abs(v) for v in row_values.values())
    den = abs(sum(row_values.values())) + floor
    return float(num / den)
