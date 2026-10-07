import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em02b.cone_measures import c_47, c_47_from_chart_integral, c_R
from em02b.transition import run_transition_benchmark


def test_transition_constants_agree():
    assert abs(c_47() - c_47_from_chart_integral()) < 2e-3
    assert c_R() > 0.0
    assert c_47() > 0.0


def test_scaled_5_approaches_c_R():
    reports = run_transition_benchmark((0.2, 0.1))
    for rep in reports:
        assert abs(rep.residual_5) < 5e-3
        assert abs(rep.amplitude_error_order_probe) < 5.0
