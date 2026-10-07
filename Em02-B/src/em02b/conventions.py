"""Labels and index conventions for Em02-B simulations.

No entry here is an inherited physical input. Every configuration
must carry one of the dataset labels below.
"""

from __future__ import annotations

from enum import Enum

ROWS = ("47", "5a", "5z", "6", "8", "9d", "9m")
TWO_LINE_ROWS = ("47", "5a", "5z")
RESPONSE_ROWS = ("6", "8", "9d", "9m")
SECTORS = ("00", "0R", "10", "01", "11")

# Post-adjoint support table from Em02-A-v6. "P" is not a nonzero claim.
SUPPORT_TABLE = {
    "47": {"00": "P", "0R": "0", "10": "P", "01": "P", "11": "P", "away": "P"},
    "5a": {"00": "P", "0R": "0", "10": "P", "01": "P", "11": "P", "away": "P"},
    "5z": {"00": "P", "0R": "P", "10": "P", "01": "P", "11": "P", "away": "P"},
    "6": {"00": "P", "0R": "0", "10": "P", "01": "P", "11": "P", "away": "0"},
    "8": {"00": "P", "0R": "0", "10": "P", "01": "P", "11": "P", "away": "0"},
    "9d": {"00": "P", "0R": "0", "10": "P", "01": "P", "11": "P", "away": "0"},
    "9m": {"00": "P", "0R": "0", "10": "P", "01": "P", "11": "P", "away": "0"},
}

RELATIVE_DIM = 8  # (U, V) in R^4 x R^4


class DatasetLabel(str, Enum):
    SMOOTH_BENCHMARK = "smooth_benchmark"
    SINGULAR_DIAGNOSTIC = "singular_diagnostic"
    MANUFACTURED_CANCELLATION = "manufactured_cancellation"
    PHYSICAL_IDENTIFIED = "physical_identified"  # unavailable until inputs exist


class ClaimStatus(str, Enum):
    OBSERVED_SCALING = "observed_scaling"
    CANDIDATE_COEFFICIENT = "candidate_coefficient"
    ANALYTICALLY_PROVED = "analytically_proved"
    NUMERICAL_QUADRATURE = "numerical_integration"
    UNRESOLVED_PHYSICAL = "unresolved_physical"
