"""Shared training utilities.

Everything that could let the test set leak into training lives here so
RF and the client's SGD see identical splits and identical scaler state.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import config

log = logging.getLogger(__name__)

# Re-export so call sites can keep doing `from models.base import FEATURE_COLS`
FEATURE_COLS = config.FEATURE_COLS
LABEL_COL    = config.LABEL_COL
SOURCE_COL   = config.SOURCE_COL


@dataclass
class SplitData:
    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    scaler: StandardScaler
    feature_cols: list[str]
    # Aligned with X_test rows. Phase 3 keeps this around for per-source
    # error analysis and SHAP slicing.
    source_test: pd.Series | None


def load_split_scale(
    data_path: Path = config.PARQUET_PATH,
    test_size: float = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
) -> SplitData:
    if not data_path.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_path} — run pcap_to_flow.py first"
        )

    log.info("Loading %s", data_path)
    df = pd.read_parquet(str(data_path), engine="pyarrow")
    log.info("Got %s rows, %d features + label", f"{len(df):,}", len(FEATURE_COLS))

    for lbl, cnt in df[LABEL_COL].value_counts().sort_index().items():
        tag = "benign" if lbl == 0 else "ddos  "
        log.info("  %s (%d) : %s", tag, lbl, f"{cnt:,}")

    if SOURCE_COL in df.columns:
        log.info("Source x label breakdown (full dataset):")
        for (src, lbl), cnt in df.groupby([SOURCE_COL, LABEL_COL]).size().items():
            tag = "benign" if lbl == 0 else "ddos"
            log.info("  %-10s %-6s : %s", src, tag, f"{cnt:,}")
        source_series: pd.Series | None = df[SOURCE_COL]
    else:
        log.warning(
            "No 'source' column — Phase 2 parquet detected, "
            "per-source diagnostics disabled."
        )
        source_series = None

    X = df[FEATURE_COLS].fillna(0.0)
    y = df[LABEL_COL].astype(int)

    # stratify=y is non-negotiable: the benign class is ~7% of rows, an
    # unstratified split routinely lands 5+ test folds with zero benign
    # samples and breaks every metric.
    if source_series is not None:
        X_train, X_test, y_train, y_test, _, src_test = train_test_split(
            X, y, source_series,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=test_size,
            random_state=random_state,
            stratify=y,
        )
        src_test = None

    log.info("Split: %s train / %s test", f"{len(X_train):,}", f"{len(X_test):,}")

    if src_test is not None:
        log.info("Source x label breakdown (TEST set):")
        combined = pd.concat(
            [src_test.rename(SOURCE_COL), y_test.rename(LABEL_COL)], axis=1
        )
        for (src, lbl), cnt in combined.groupby([SOURCE_COL, LABEL_COL]).size().items():
            tag = "benign" if lbl == 0 else "ddos"
            log.info("  %-10s %-6s : %s", src, tag, f"{cnt:,}")

    # Fit on train ONLY. Fitting on the full X (a classic junior mistake)
    # silently leaks test-set mean/std into the model.
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    return SplitData(
        X_train=X_train_s,
        X_test=X_test_s,
        y_train=y_train.to_numpy(),
        y_test=y_test.to_numpy(),
        scaler=scaler,
        feature_cols=list(FEATURE_COLS),
        source_test=src_test,
    )


def evaluate_and_plot(
    name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    plots_dir: Path = config.PLOTS_DIR,
) -> dict[str, float]:
    scores = {
        "accuracy":  accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall":    recall_score(y_true, y_pred, zero_division=0),
        "f1":        f1_score(y_true, y_pred, zero_division=0),
    }
    cm = confusion_matrix(y_true, y_pred)

    log.info("")
    log.info("=" * 56)
    log.info("  Model : %s", name)
    log.info("=" * 56)
    for k, v in scores.items():
        log.info("  %-10s: %.4f", k.capitalize(), v)
    log.info("")
    log.info("  Confusion Matrix (rows=actual, cols=predicted):")
    log.info("               Pred 0    Pred 1")
    log.info("  Actual  0   %8s  %8s", f"{cm[0, 0]:,}", f"{cm[0, 1]:,}")
    log.info("  Actual  1   %8s  %8s", f"{cm[1, 0]:,}", f"{cm[1, 1]:,}")

    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Benign", "DDoS"],
                yticklabels=["Benign", "DDoS"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    out = plots_dir / f"cm_{name}.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("  -> %s", out)

    return scores


def save_artifacts(
    artifacts: dict[str, object],
    artifacts_dir: Path = config.ARTIFACTS_DIR,
) -> None:
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    for name, obj in artifacts.items():
        out = artifacts_dir / f"{name}.joblib"
        joblib.dump(obj, out)
        log.info("  saved %s", out)


def plot_model_comparison(
    scores_by_model: dict[str, dict[str, float]],
    plots_dir: Path = config.PLOTS_DIR,
    metrics: tuple[str, ...] = ("accuracy", "precision", "recall", "f1"),
) -> None:
    plots_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(scores_by_model.keys())
    n_models = len(model_names)
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(metrics))
    for i, name in enumerate(model_names):
        vals = [scores_by_model[name][m] for m in metrics]
        offset = (i - (n_models - 1) / 2) * width
        ax.bar(x + offset, vals, width, label=name)

    ax.set_xticks(x)
    ax.set_xticklabels([m.capitalize() for m in metrics])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    plt.tight_layout()
    out = plots_dir / "model_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    log.info("  -> %s", out)
