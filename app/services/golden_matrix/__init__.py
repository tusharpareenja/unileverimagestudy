"""
Golden Matrix: per-respondent RDE design generator.
Each respondent gets a unique T×P block (T tasks, P elements).
Silos = categories; variables = elements within each silo.
"""

from .config import DesignConfig
from .exceptions import (
    RDEError,
    InfeasibleConfigError,
    RowConstraintsError,
    BlockConstructionError,
    BlockUniquenessError,
    DiagnosticsFailedError,
)
from .generator import generate_golden_matrix
from .design_builder import StackBuilder
from .diagnostics import run_all_diagnostics
from .preflight import preflight_check
from .validation import validate_design

__all__ = [
    "DesignConfig",
    "generate_golden_matrix",
    "preflight_check",
    "StackBuilder",
    "run_all_diagnostics",
    "RDEError",
    "InfeasibleConfigError",
    "RowConstraintsError",
    "BlockConstructionError",
    "BlockUniquenessError",
    "DiagnosticsFailedError",
    "validate_design",
]
