"""Write the versioned audit manifest used by the manuscripts."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em02b.adjoint_5a import non_tip_integral
from em02b.atomic_integral import route_pair
from em02b.cone_measures import c_47, c_47_from_chart_integral, c_R
from em02b.conventions import DatasetLabel
from em02b.profiles import check_profile_identities
from em02b.simulated_inputs import require_simulated
from em02b.transition import (
    chart_47_mesh_study,
    exact_amplitude_probe_target,
    run_transition_benchmark,
)


def _source_id() -> str:
    root = Path(__file__).resolve().parents[2] / "src" / "em02b"
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def main() -> None:
    header = require_simulated(DatasetLabel.SINGULAR_DIAGNOSTIC, "S1-weighted-5a")
    ident = check_profile_identities()
    reports = run_transition_benchmark((0.25, 0.125, 0.0625))
    pair = route_pair(delta=0.35, differentiated=False)
    pair_dot = route_pair(delta=0.35, differentiated=True)
    adjoint = non_tip_integral()
    mesh = chart_47_mesh_study(eps=0.1, sizes=(21, 41, 81))
    payload = {
        **header,
        "source_id": _source_id(),
        "profile_kappa_m2_residual": ident["kappa_m2_residual"],
        "c_R": c_R(),
        "c_47": c_47(),
        "c_47_chart": c_47_from_chart_integral(),
        "amplitude_probe_target": exact_amplitude_probe_target(),
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
        "mesh_47_at_eps_0.1": mesh,
        "atomic_5a": {
            "delta": 0.35,
            "passive": "exp(-T) on the second cone; sphere probability on its angle",
            "B_gauss": pair["B"].value,
            "A_sobol": pair["A"].value,
            "abs_diff": abs(pair["B"].value - pair["A"].value),
            "B_samples": pair["B"].n_samples,
            "A_samples": pair["A"].n_samples,
            "dot_B_gauss": pair_dot["B"].value,
            "dot_A_sobol": pair_dot["A"].value,
            "dot_abs_diff": abs(pair_dot["B"].value - pair_dot["A"].value),
            "status": "independent_quadrature_of_closed_form_atomic_integrand",
        },
        "scalar_p5a_nontip": {
            "value": adjoint.value,
            "dblquad_error_estimate": adjoint.error_estimate,
            "gamma_window": list(adjoint.gamma_window),
            "status": "simulated_scalar_adjoint_not_inherited_components",
        },
    }
    out = Path(__file__).resolve().parents[2] / "results" / "tables" / "audit_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
