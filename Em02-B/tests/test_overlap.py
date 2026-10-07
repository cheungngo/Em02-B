import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em02b.inner_check import tensor_inner
from em02b.inner_integral import inner_terms
from em02b.overlap_moments import parts_moment
from em02b.specialization import S_STAR


def test_parts_comparison_is_not_adopted_blindly():
    gap = parts_moment(float(S_STAR[1]), 0.35, 1.0)["gap"]
    assert abs(gap) < 1e-6


def test_factored_inner_matches_overlap_box():
    value = inner_terms(S_STAR, 0.35)["sum"]
    check = tensor_inner(S_STAR, 0.35, n=16)
    assert abs(value - check) < 0.01 * max(1.0, abs(value))
