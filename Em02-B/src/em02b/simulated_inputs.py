"""Simulated crossed families S0 and S1.

S0: smooth compact array, polynomial times a bump.
S1: atomic crossed distribution. Convolution is closed form.
Locations must satisfy U_star - V_star = b - a at a contributing point.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .conventions import DatasetLabel
from .profiles import kappa_delta, rho_delta


@dataclass(frozen=True)
class AtomicSite:
    """S1 site. U_star - V_star is the relative b-a that the seed point must match."""

    u_star: np.ndarray
    v_star: np.ndarray
    tensor_scale: float = 1.0

    def __post_init__(self) -> None:
        object.__setattr__(self, "u_star", np.asarray(self.u_star, dtype=float).reshape(4))
        object.__setattr__(self, "v_star", np.asarray(self.v_star, dtype=float).reshape(4))

    @property
    def b_minus_a(self) -> np.ndarray:
        return self.u_star - self.v_star

    def relative(self) -> np.ndarray:
        return np.concatenate([self.u_star, self.v_star])


def s0_profile(h: np.ndarray) -> np.ndarray:
    """Compact smooth scalar factor B(U, V). Polynomial prefactor is nontrivial.

    B = (1 + U^0 + V^0^2) * rho_unit, with rho_unit the even bump, so B is
    compact and smooth. The polynomial is not even; odd moments need not vanish.
    """
    h = np.asarray(h, dtype=float)
    from .profiles import rho

    u0 = h[..., 0]
    v0 = h[..., 4]
    return (1.0 + 0.2 * u0 + 0.05 * v0 ** 2) * rho(h)


def s1_coefficient(h: np.ndarray, site: AtomicSite, delta: float, differentiated: bool = False) -> np.ndarray:
    """Closed form c_delta or dot c_delta for the atomic family. No inner convolution."""
    center = site.relative()
    shifted = np.asarray(h, dtype=float) - center
    raw = kappa_delta(shifted, delta) if differentiated else rho_delta(shifted, delta)
    return site.tensor_scale * raw


def dataset_header(label: DatasetLabel, name: str) -> dict[str, str]:
    if label is DatasetLabel.PHYSICAL_IDENTIFIED:
        raise RuntimeError("physical_identified is unavailable until the inherited inputs are supplied")
    return {"label": label.value, "name": name, "physical_admission": "not_claimed"}


def require_simulated(label: DatasetLabel, name: str) -> dict[str, str]:
    """Entry point used by the runner. physical_identified is rejected here."""
    return dataset_header(label, name)
