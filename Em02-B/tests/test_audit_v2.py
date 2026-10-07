import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em02b.adjoint_5a import non_tip_integral, scalar_p5a_expression
from em02b.atomic_integral import gauss_atomic_integral, route_pair
from em02b.conventions import DatasetLabel
from em02b.simulated_inputs import require_simulated
from em02b.transition import chart_47_mesh_study, exact_amplitude_probe_target


def test_physical_label_is_rejected():
    with pytest.raises(RuntimeError):
        require_simulated(DatasetLabel.PHYSICAL_IDENTIFIED, "missing")


def test_atomic_routes_agree_under_independent_quadrature():
    pair = route_pair(delta=0.35)
    assert abs(pair["B"].value) > 0.0
    assert abs(pair["A"].value - pair["B"].value) / abs(pair["B"].value) < 0.05


def test_gauss_refinement_moves_less_than_coarse_gap():
    from em02b.atomic_integral import future_null_site

    site = future_null_site()
    coarse = gauss_atomic_integral(site, n_s=12, n_theta=8, n_phi=12)
    fine = gauss_atomic_integral(site, n_s=20, n_theta=12, n_phi=16)
    assert abs(fine.value - coarse.value) / abs(fine.value) < 0.03


def test_probe_matches_exact_target():
    from em02b.transition import amplitude_remainder_probe

    target = exact_amplitude_probe_target()
    assert abs(amplitude_remainder_probe(0.1) - target) < 1e-4


def test_mesh_residual_shrinks():
    rows = chart_47_mesh_study(eps=0.1, sizes=(21, 41))
    assert abs(rows[1]["residual"]) < abs(rows[0]["residual"])


def test_scalar_adjoint_is_not_identically_zero():
    assert scalar_p5a_expression() != 0
    result = non_tip_integral()
    assert result.error_estimate < 1e-6
    assert result.value != 0.0
