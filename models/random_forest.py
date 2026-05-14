"""Random Forest factory."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier

import config


def build_random_forest(
    n_estimators: int = 300,
    max_depth: int | None = None,
    min_samples_split: int = 2,
    min_samples_leaf: int = 1,
    random_state: int = config.RANDOM_STATE,
    n_jobs: int = -1,
) -> RandomForestClassifier:
    # class_weight='balanced' reweights samples by inverse class frequency,
    # so the rare benign class doesn't get drowned out by the DDoS majority.
    # Keeps us away from SMOTE / oversampling for a tabular problem this small.
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )
