"""Run the B3 localization benchmark and the S1 route-agreement check."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from em02b.cone_measures import c_47, c_47_from_chart_integral, c_R
from em02b.profiles import check_profile_identities
from em02b.route_ab import route_a_atomic_5a, route_b_atomic_5a
from em02b.simulated_inputs import AtomicSite
from em02b.transition import run_transition_benchmark


def main() -> None:
    ident = check_profile_identities()
    reports = run_transition_benchmark((0.25, 0.125, 0.0625))
    site = AtomicSite(np.array([1.0, 1.0, 0.0, 0.0]), np.array([0.0, 0.0, 0.0, 0.0]))
    route_a = route_a_atomic_5a(site, delta=0.35, n_radial=12, n_side=3)
    route_b = route_b_atomic_5a(site, delta=0.35, n_radial=12, n_side=3)
    payload = {
        "label": "singular_diagnostic",
        "physical_admission": "not_claimed",
        "profile_kappa_m2_residual": ident["kappa_m2_residual"],
        "c_R": c_R(),
        "c_47": c_47(),
        "c_47_chart": c_47_from_chart_integral(),
        "transition": [
            {
                "eps": r.eps,
                "scaled_5": r.scaled_5,
                "residual_5": r.residual_5,
                "scaled_47": r.scaled_47,
                "residual_47": r.residual_47,
                "amplitude_probe": r.amplitude_error_order_probe,
            }
            for r in reports
        ],
        "route_agreement": {
            "A": route_a.value,
            "B": route_b.value,
            "abs_diff": abs(route_a.value - route_b.value),
            "status": "implementation_check_not_physical_asymptotics",
        },
    }
    out = Path(__file__).resolve().parents[2] / "results" / "tables" / "b3_transition.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
