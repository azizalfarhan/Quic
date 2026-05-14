"""Phase 3 model package.

Exposes the shared training utilities and the per-model factory functions
used by train_phase3.py.
"""

from .base import (
    FEATURE_COLS,
    LABEL_COL,
    SOURCE_COL,
    evaluate_and_plot,
    load_split_scale,
    save_artifacts,
)
from .client_sgd import build_client_sgd
from .random_forest import build_random_forest

__all__ = [
    "FEATURE_COLS",
    "LABEL_COL",
    "SOURCE_COL",
    "load_split_scale",
    "evaluate_and_plot",
    "save_artifacts",
    "build_random_forest",
    "build_client_sgd",
]
