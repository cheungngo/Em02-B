import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em02b.profiles import check_profile_identities, kappa, rho


def test_mass_and_evenness():
    ident = check_profile_identities()
    assert abs(ident["mass"] - 1.0) < 1e-10
    assert abs(ident["edge_integral_t_psi_prime"] + 1.0) < 1e-10
    assert abs(ident["kappa_m2_residual"]) < 1e-9


def test_kappa_zero_on_axes_sample():
    rng = np.random.default_rng(0)
    pts = rng.uniform(-0.5, 0.5, size=(20, 8))
    # even profile: rho(-h) = rho(h)
    assert np.allclose(rho(pts), rho(-pts))
    # kappa is even as well for this radial-product construction? 
    # h·grad rho is even if rho is even, so kappa is even.
    assert np.allclose(kappa(pts), kappa(-pts))
