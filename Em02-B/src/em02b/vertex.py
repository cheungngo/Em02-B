"""Vertex from the metric-index formula, then the declared 5a reduction.

Z^nu = (1/2) d_{V^nu} B for every nu, including spatial. A time-only A
specialization does not delete those terms. The simulated adjoint is
d_s0^2 d_t0^2, applied after the vertex.
"""

from __future__ import annotations

import numpy as np

from .specialization import MASS


def metric_Z(grad_V: np.ndarray) -> np.ndarray:
    """grad_V is d_V B, shape (..., 4). Returns Z = grad_V / 2."""
    return 0.5 * np.asarray(grad_V, dtype=float)


def vertex_from_metric(B: np.ndarray, grad_V: np.ndarray, dQ_dy: np.ndarray, Q: np.ndarray, mass: float = MASS) -> np.ndarray:
    """W = Z^nu d_{y^nu} Q + (m^2/2) h Q, with h = B."""
    Z = metric_Z(grad_V)
    return np.sum(Z * dQ_dy, axis=-1) + 0.5 * mass**2 * B * Q


def specialized_vertex(B: np.ndarray, grad_V: np.ndarray, dQ_dy: np.ndarray, Q: np.ndarray, mass: float = MASS) -> np.ndarray:
    """Same expression written as the plan's specialized formula."""
    return 0.5 * np.sum(grad_V * dQ_dy, axis=-1) + 0.5 * mass**2 * B * Q
