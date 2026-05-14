"""Main training runner for Phase 3.

Loads the merged Phase 3 parquet, trains the RF + the client's SGD stub
on the same split with the same scaler, and persists everything that
explain_shap.py needs.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import config
from models import (
    build_client_sgd,
    build_random_forest,
    evaluate_and_plot,
    load_split_scale,
    save_artifacts,
)
from models.base import plot_model_comparison

config.configure_logging()
log = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Phase 3 training pipeline")
    p.add_argument("--data",          type=Path, default=config.PARQUET_PATH)
    p.add_argument("--test-size",     type=float, default=config.TEST_SIZE)
    p.add_argument("--random-state",  type=int, default=config.RANDOM_STATE)
    p.add_argument("--plots-dir",     type=Path, default=config.PLOTS_DIR)
    p.add_argument("--artifacts-dir", type=Path, default=config.ARTIFACTS_DIR)
    return p.parse_args()


def main() -> None:
    args = parse_args()

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

    log.info("Training RandomForest ...")
    rf = build_random_forest(random_state=args.random_state)
    rf.fit(data.X_train, data.y_train)
    rf_scores = evaluate_and_plot(
        "RandomForest", data.y_test, rf.predict(data.X_test), args.plots_dir,
    )

    # Hyperparameters in build_client_sgd are a placeholder — swap them
    # once the client delivers his actual spec.
    log.info("Training Client SGD LogReg (stub) ...")
    sgd = build_client_sgd(random_state=args.random_state)
    sgd.fit(data.X_train, data.y_train)
    sgd_scores = evaluate_and_plot(
        "ClientSGD", data.y_test, sgd.predict(data.X_test), args.plots_dir,
    )

    plot_model_comparison(
        {"RandomForest": rf_scores, "ClientSGD": sgd_scores},
        args.plots_dir,
    )

    log.info("Saving artifacts ...")
    save_artifacts(
        {"rf": rf, "sgd": sgd, "scaler": data.scaler},
        args.artifacts_dir,
    )

    log.info("Done.")


if __name__ == "__main__":
    main()
