"""SHAP explanations for the Phase 3 models.

Explainer choice matters here:
    RandomForest -> TreeExplainer   (exact Shapley values for trees)
    SGD LogReg   -> LinearExplainer (exact for linear models)

KernelExplainer would only approximate either case at ~100x the cost,
so we don't use it.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

import config
from models.base import load_split_scale

config.configure_logging()
log = logging.getLogger(__name__)


def _select_positive_class(shap_values, expected_value):
    """Always return SHAP values for class 1 (DDoS).

    shap.TreeExplainer on a binary RandomForestClassifier returns one of:
      - list [class_0, class_1] of arrays   (older shap / sklearn combos), or
      - ndarray with the class axis at the end (newer combos).
    Both shapes need different indexing; the client's plots should always
    show "what pushes a flow towards DDoS".
    """
    ev_is_per_class = isinstance(expected_value, (list, np.ndarray))
    ev_pos = expected_value[1] if ev_is_per_class else expected_value

    if isinstance(shap_values, list):
        return shap_values[1], ev_pos
    if isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
        return shap_values[..., 1], ev_pos
    return shap_values, expected_value


def _save_summary(shap_values, X, out_path: Path, plot_type: str | None) -> None:
    plt.figure()
    shap.summary_plot(
        shap_values, X,
        feature_names=config.FEATURE_COLS,
        plot_type=plot_type,   # None -> beeswarm
        show=False,
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("  -> %s", out_path)


def explain_random_forest(rf, X_test_df: pd.DataFrame, plots_dir: Path) -> None:
    log.info("Computing SHAP values for RandomForest (TreeExplainer) ...")
    explainer = shap.TreeExplainer(rf)
    raw = explainer.shap_values(X_test_df)
    shap_values, _ = _select_positive_class(raw, explainer.expected_value)

    _save_summary(shap_values, X_test_df, plots_dir / "shap_rf_bar.png", "bar")
    _save_summary(shap_values, X_test_df, plots_dir / "shap_rf_beeswarm.png", None)


def explain_client_sgd(
    sgd,
    X_train_df: pd.DataFrame,
    X_test_df: pd.DataFrame,
    plots_dir: Path,
    background_size: int,
) -> None:
    log.info("Computing SHAP values for Client SGD (LinearExplainer) ...")
    # Background sample keeps memory bounded and is enough for an exact
    # LinearExplainer fit — going past ~200 buys no accuracy here.
    background = shap.sample(
        X_train_df, background_size, random_state=config.RANDOM_STATE,
    )
    explainer = shap.LinearExplainer(sgd, background)
    shap_values = explainer.shap_values(X_test_df)

    _save_summary(shap_values, X_test_df, plots_dir / "shap_sgd_bar.png", "bar")
    _save_summary(shap_values, X_test_df, plots_dir / "shap_sgd_beeswarm.png", None)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SHAP explanations for Phase 3 models")
    p.add_argument("--data",            type=Path, default=config.PARQUET_PATH)
    p.add_argument("--artifacts-dir",   type=Path, default=config.ARTIFACTS_DIR)
    p.add_argument("--plots-dir",       type=Path, default=config.PLOTS_DIR)
    p.add_argument("--test-size",       type=float, default=config.TEST_SIZE)
    p.add_argument("--random-state",    type=int, default=config.RANDOM_STATE,
                   help="MUST match train_phase3.py to regenerate the same split.")
    p.add_argument("--background-size", type=int, default=config.SHAP_BACKGROUND_SIZE)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    args.plots_dir.mkdir(parents=True, exist_ok=True)

    rf_path     = args.artifacts_dir / "rf.joblib"
    sgd_path    = args.artifacts_dir / "sgd.joblib"
    scaler_path = args.artifacts_dir / "scaler.joblib"

    try:
        log.info("Loading artifacts from %s", args.artifacts_dir)
        rf     = joblib.load(rf_path)
        sgd    = joblib.load(sgd_path)
        _      = joblib.load(scaler_path)  # presence-checked; not used directly here
    except FileNotFoundError as exc:
        log.critical("Trained artifacts missing — did you run train_phase3.py yet?")
        log.critical("Details: %s", exc)
        sys.exit(1)

    try:
        data = load_split_scale(
            data_path=args.data,
            test_size=args.test_size,
            random_state=args.random_state,
        )
    except FileNotFoundError as exc:
        log.critical("Dataset is missing — did you run pcap_to_flow.py yet?")
        log.critical("Details: %s", exc)
        sys.exit(1)

    # DataFrame so SHAP picks the feature names off the columns
    X_train_df = pd.DataFrame(data.X_train, columns=config.FEATURE_COLS)
    X_test_df  = pd.DataFrame(data.X_test,  columns=config.FEATURE_COLS)

    explain_random_forest(rf, X_test_df, args.plots_dir)
    explain_client_sgd(
        sgd, X_train_df, X_test_df,
        plots_dir=args.plots_dir,
        background_size=args.background_size,
    )

    log.info("Done — 4 SHAP plots saved to %s", args.plots_dir)


if __name__ == "__main__":
    main()
