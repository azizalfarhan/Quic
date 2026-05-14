"""Stub for the client's SGD Logistic Regression.

Hyperparameters here are a placeholder — swap them when the client
delivers his spec. The data loading / split stays in models.base so
both models share the same scaler and the same train/test split.
"""

from __future__ import annotations

from sklearn.linear_model import SGDClassifier

import config


def build_client_sgd(
    loss: str = "log_loss",
    alpha: float = 1e-4,
    max_iter: int = 1000,
    tol: float = 1e-3,
    random_state: int = config.RANDOM_STATE,
    n_jobs: int = -1,
) -> SGDClassifier:
    # loss='log_loss' makes this mathematically a Logistic Regression
    # optimised via SGD — exactly what the client asked for. Mirroring
    # class_weight='balanced' from the RF keeps the model comparison fair.
    return SGDClassifier(
        loss=loss,
        alpha=alpha,
        max_iter=max_iter,
        tol=tol,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=n_jobs,
    )
