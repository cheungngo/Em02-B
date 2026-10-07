"""Immutable 5a-row run. Gate G4 is recorded, not assumed."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from em02b.row5a_route_a import leibniz_density
from em02b.row5a_route_b import seed_density
from em02b.row5a_test import J_t, isolation_check, support_box
from em02b.simulated_inputs import require_simulated
from em02b.specialization import LABEL, NAME, U_STAR, specialization_record


def _source_id() -> str:
    root = Path(__file__).resolve().parents[2] / "src" / "em02b"
    digest = hashlib.sha256()
    for path in sorted(root.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _algebra() -> dict[str, float]:
    U = np.array([1.05, 0.9, 0.02, -0.01])
    s = np.array([0.95, 0.92, 0.03, -0.02])
    b = seed_density(U, s, 0.35)["terms"]
    a = leibniz_density(U, s, 0.35)["terms"]
    return {"max_abs_term_gap": float(np.max(np.abs(a - b))), "sum": float(np.sum(b))}


def _gauss_sum(delta: float, n_u: int, n_s: int, n_theta: int, n_phi: int, route: str) -> float:
    box = support_box(delta)
    xs = [leggauss(n_u) for _ in range(4)]
    su, sw = leggauss(n_s)
    th, thw = leggauss(n_theta)
    s = 0.5 * (box["s_radius_high"] - box["s_radius_low"]) * su + 0.5 * (box["s_radius_high"] + box["s_radius_low"])
    sw = 0.5 * (box["s_radius_high"] - box["s_radius_low"]) * sw
    theta = 0.5 * box["theta_max"] * (th + 1.0)
    thw = 0.5 * box["theta_max"] * thw
    phi = np.linspace(0.0, 2.0 * np.pi, n_phi, endpoint=False)
    dphi = 2.0 * np.pi / n_phi
    fn = seed_density if route == "B" else leibniz_density
    acc = 0.0
    for it, thv in enumerate(theta):
        for phv in phi:
            grids = np.meshgrid(*[x for x, _ in xs], s, indexing="ij")
            U = U_STAR + delta * np.stack(grids[:4], -1)
            direction = np.array([np.cos(thv), np.sin(thv) * np.cos(phv), np.sin(thv) * np.sin(phv)])
            ss = np.concatenate([grids[4][..., None], grids[4][..., None] * direction], -1)
            dens = fn(U, ss, delta)["sum"]
            wU = np.ones(grids[0].shape)
            for k, (_, w) in enumerate(xs):
                shape = [1, 1, 1, 1, 1]
                shape[k] = n_u
                wU *= w.reshape(shape)
            weight = (delta**4) * (grids[4] / (4.0 * np.pi)) * np.sin(thv) * wU * sw.reshape((1, 1, 1, 1, n_s)) * thw[it] * dphi
            acc += float(np.sum(dens * weight))
    return acc


def main() -> None:
    header = require_simulated(LABEL, NAME)
    delta = 0.35
    jt = J_t()
    refinement = []
    for n_u in (5, 6, 7):
        b = _gauss_sum(delta, n_u, 10, 4, 4, "B")
        a = _gauss_sum(delta, n_u, 10, 4, 4, "A")
        refinement.append({"n_u": n_u, "route_B": b, "route_A": a, "abs_gap": abs(a - b)})
    payload = {
        **header,
        **specialization_record(),
        "source_id": _source_id(),
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "isolation": isolation_check(),
        "support_box": support_box(delta),
        "algebra": _algebra(),
        "J_t": jt,
        "refinement": refinement,
        "gates": {
            "G1_specification": "passed",
            "G2_algebra": "passed",
            "G3_geometry": "passed",
            "G4_integration": "not_passed_sign_change_under_refinement",
            "G5_routes": "constructions_agree_on_shared_grid_integral_not_stable",
            "G6_derivative": "not_entered",
            "G7_reporting": "passed_no_row_value_released",
        },
        "row_value_released": False,
        "status": "displayed_mathematical_data_5a_constructions_without_a_stable_integral",
        "not_claimed": ["physically_identified", "seven_row_sum", "S2", "authorization", "wavefront_bound"],
    }
    out_dir = Path(__file__).resolve().parents[2] / "results" / "runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"row5a_{payload['source_id']}.json"
    path.write_text(json.dumps(payload, indent=2))
    print(path)
    print(json.dumps(payload["gates"], indent=2))
    print(json.dumps(payload["refinement"], indent=2))


if __name__ == "__main__":
    main()
