"""Scalar coincidence-localization experiment (Em02-B benchmark E2).

This checks the internal epsilon expansion of Theorem (calculated scalar
seed transition coefficients). It is not a delta-smoothing certificate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import quad

from .cone_measures import c_47, c_R
from .cutoffs import b0


@dataclass
class TransitionReport:
    eps: float
    scaled_5: float
    c_R: float
    residual_5: float
    scaled_47: float
    c_47: float
    residual_47: float
    amplitude_error_order_probe: float


def _radial_5_mass(eps: float) -> float:
    """integral_0^inf T b_0(2 T^2 / eps^2) dT, which equals c_R eps^2."""
    t_max = eps * np.sqrt(2.0) + 1e-12
    ts = np.linspace(0.0, t_max, 2001)
    vals = ts * b0(2.0 * ts * ts / (eps * eps))
    return float(np.trapezoid(vals, ts))


def _chart_47_mass(eps: float, beta: float = 1.5, n: int = 81) -> float:
    """Integrate the 47 tip density. For H=1 and unit passive measure this is c_47 eps^3 + O(eps^4)."""
    g = np.linspace(-2.2, 2.2, n)
    rad = np.linspace(0.0, 2.2, n)
    dg = g[1] - g[0]
    dr = rad[1] - rad[0]
    gg, rr = np.meshgrid(g, rad, indexing="ij")
    upsilon = -(eps ** 2) * (rr ** 2) / (4.0 * (beta - eps * gg))
    rsq = 2.0 * (eps * gg) ** 2 + 2.0 * upsilon ** 2 + (eps * rr) ** 2
    theta = b0(rsq / (eps * eps))
    # eq:47-integral density after gamma=eps G, omega=eps W; polar weight 2 pi R
    density = (eps ** 3) * beta / (8.0 * np.pi ** 2 * (beta - eps * gg))
    weight = density * theta * (2.0 * np.pi * rr)
    return float(np.sum(weight) * dg * dr)


def exact_amplitude_probe_target(slope: float = 0.3) -> float:
    """Exact value of the scaled probe for H = 1 + slope * T.

    The change of variables T = eps t removes eps. The target is
    slope * integral_0^inf t^2 b_0(2 t^2) dt.
    """
    return slope * quad(
        lambda t: (t ** 2) * float(b0(np.array([2.0 * t * t]))[0]),
        0.0,
        np.sqrt(2.0),
        epsabs=1e-12,
    )[0]


def chart_47_mesh_study(eps: float = 0.1, sizes: tuple[int, ...] = (21, 41, 81)) -> list[dict[str, float]]:
    """Fixed-eps mesh refinement. A shrinking residual supports grid bias."""
    target = c_47()
    rows = []
    for n in sizes:
        scaled = _chart_47_mass(eps, n=n) / (eps ** 3)
        rows.append({"n": float(n), "eps": eps, "scaled_47": scaled, "residual": scaled - target})
    return rows


def amplitude_remainder_probe(eps: float, slope: float = 0.3) -> float:
    """Scaled probe. For this amplitude the exact value is independent of eps."""
    t_max = eps * np.sqrt(2.0) + 1e-12
    ts = np.linspace(0.0, t_max, 2001)
    theta = b0(2.0 * ts * ts / (eps * eps))
    pairing = float(np.trapezoid(ts * (1.0 + slope * ts) * theta, ts))
    predicted = c_R() * (eps ** 2) * 1.0
    return (pairing - predicted) / (eps ** 3)


def run_transition_benchmark(eps_grid: tuple[float, ...] = (0.2, 0.1, 0.05)) -> list[TransitionReport]:
    cr = c_R()
    c47 = c_47()
    reports = []
    for eps in eps_grid:
        mass5 = _radial_5_mass(eps)
        scaled5 = mass5 / (eps ** 2)
        scaled47 = _chart_47_mass(eps) / (eps ** 3)
        reports.append(
            TransitionReport(
                eps=eps,
                scaled_5=scaled5,
                c_R=cr,
                residual_5=scaled5 - cr,
                scaled_47=scaled47,
                c_47=c47,
                residual_47=scaled47 - c47,
                amplitude_error_order_probe=amplitude_remainder_probe(eps),
            )
        )
    return reports
