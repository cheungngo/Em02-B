import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em02b.adjoints import restriction_order_gap
from em02b.quadrature import sobol_unit
from em02b.route_ab import cancellation_indicator, incidence_residual, route_a_atomic_5a, route_b_atomic_5a
from em02b.seed_geometry import counterexample_5z_noncore, isolation_47_patch, isolation_5a_patch
from em02b.simulated_inputs import AtomicSite


def test_isolation_and_counterexample():
    n = np.array([1.0, 0.0, 0.0])
    m = np.array([0.0, 1.0, 0.0])
    iso = isolation_5a_patch(beta=1.0, epsilon=0.3, n=n, m=m)
    assert iso["surviving_row"] == "5a"
    assert iso["excludes_47"] and iso["excludes_response"]
    iso47 = isolation_47_patch(beta=1.0, gamma=1.0, n=n, m=m)
    assert iso47["surviving_row"] == "47"
    assert iso47["excludes_5"]
    cx = counterexample_5z_noncore()
    assert cx["on_5z_seed"]
    assert cx["qR"] > 0.0


def test_routes_agree_on_atomic_site():
    site = AtomicSite(
        u_star=np.array([1.0, 1.0, 0.0, 0.0]),
        v_star=np.array([0.0, 0.0, 0.0, 0.0]),
    )
    assert abs(incidence_residual(site)) < 1e-12
    a = route_a_atomic_5a(site, delta=0.4, n_radial=8, n_side=2)
    b = route_b_atomic_5a(site, delta=0.4, n_radial=8, n_side=2)
    assert a.n_samples == b.n_samples
    assert abs(a.value - b.value) < 1e-12
    assert abs(a.value) > 0.0


def test_manufactured_cancellation_indicator():
    rows = {"6": 1.0 + 0.0j, "9m": -1.0 + 1e-14j, "47": 0.0j}
    indicator = cancellation_indicator(rows)
    assert indicator > 1e6


def test_sobol_is_power_of_two():
    pts = sobol_unit(4, m=3, seed=2)
    assert pts.shape == (8, 4)


def test_ambient_derivative_precedes_restriction():
    gap = restriction_order_gap()
    assert gap != 0
