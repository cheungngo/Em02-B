import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em02b.conventions import DatasetLabel
from em02b.row5a_route_a import leibniz_density
from em02b.row5a_route_b import seed_density
from em02b.row5a_test import J_t, isolation_check
from em02b.simulated_inputs import require_simulated
from em02b.vertex import specialized_vertex, vertex_from_metric


def test_physical_rejection_survives_the_row():
    with pytest.raises(RuntimeError):
        require_simulated(DatasetLabel.PHYSICAL_IDENTIFIED, "row")


def test_vertex_reduction_matches_metric_formula():
    B = np.array(0.3)
    grad = np.array([0.2, -0.1, 0.4, 0.05])
    dQ = np.array([1.0, -2.0, 0.5, 0.0])
    Q = np.array(0.7)
    assert specialized_vertex(B, grad, dQ, Q) == pytest.approx(vertex_from_metric(B, grad, dQ, Q))
    assert abs(specialized_vertex(B, grad, dQ, Q) - (0.5 * grad[0] * dQ[0])) > 1e-6


def test_constructions_agree_termwise():
    U = np.array([1.02, 0.97, 0.01, -0.02])
    s = np.array([0.99, 0.96, -0.01, 0.02])
    gap = np.max(np.abs(seed_density(U, s, 0.3)["terms"] - leibniz_density(U, s, 0.3)["terms"]))
    assert gap < 1e-8


def test_isolation_and_nonzero_Jt():
    iso = isolation_check()
    assert iso["s_future_null"] and iso["t_future_null"]
    assert iso["b_minus_z_timelike"] and iso["directions_nonparallel"]
    assert abs(J_t(n_t=12, n_theta=8, n_phi=12)) > 1e-6
