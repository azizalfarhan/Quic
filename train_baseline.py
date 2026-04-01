"""train_baseline.py — traditional ML baselines for QUIC DDoS detection.

Trains Logistic Regression and Random Forest on the flow features from
pcap_to_flow.py.  Prints metrics + saves confusion-matrix heatmaps.

The dataset is heavily imbalanced (~306 benign vs ~10k DDoS).  We tried
AdaBoost first but it collapsed under the skew.  Switched to Random Forest
with class_weight='balanced' which handles it natively — no need for SMOTE
or any other synthetic resampling.
"""

from __future__ import annotations

import logging
import sys
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

# No IPs or ports — only statistical features so the model doesn't just
# memorize which hosts are attackers in this particular capture
FEATURE_COLS = [
    "flow_duration",
    "fwd_pkts", "bwd_pkts",
    "fwd_bytes", "bwd_bytes",
    "iat_mean", "iat_std", "iat_max", "iat_min",
    "pkt_len_mean", "pkt_len_std", "pkt_len_max", "pkt_len_min",
]
LABEL_COL = "label"

log = logging.getLogger(__name__)


def log_metrics(name: str, y_true: np.ndarray, y_pred: np.ndarray) -> None:
    """Print the standard classification report for one model."""
    cm = confusion_matrix(y_true, y_pred)
    log.info("")
    log.info("=" * 56)
    log.info("  Model : %s", name)
    log.info("=" * 56)
    log.info("  Accuracy  : %.4f", accuracy_score(y_true, y_pred))
    log.info("  Precision : %.4f", precision_score(y_true, y_pred, zero_division=0))
    log.info("  Recall    : %.4f", recall_score(y_true, y_pred, zero_division=0))
    log.info("  F1-Score  : %.4f", f1_score(y_true, y_pred, zero_division=0))
    log.info("")
    log.info("  Confusion Matrix (rows=actual, cols=predicted):")
    log.info("               Pred 0    Pred 1")
    log.info("  Actual  0   %8s  %8s", f"{cm[0,0]:,}", f"{cm[0,1]:,}")
    log.info("  Actual  1   %8s  %8s", f"{cm[1,0]:,}", f"{cm[1,1]:,}")


def save_cm_plot(y_true: np.ndarray, y_pred: np.ndarray, name: str) -> None:
    """Render confusion matrix as a heatmap and save to plots/."""
    Path("plots").mkdir(exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Benign", "DDoS"],
                yticklabels=["Benign", "DDoS"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix — {name}")
    plt.tight_layout()
    plt.savefig(f"plots/cm_{name}.png", dpi=150)
    plt.close()
    log.info("  -> plots/cm_%s.png", name)


def main() -> None:
    parser = argparse.ArgumentParser(description="train baselines on QUIC flow dataset")
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data" / "processed" / "quic_dataset.parquet")
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--rf-estimators", type=int, default=100)
    args = parser.parse_args()

    if not args.data.exists():
        log.error("Dataset not found: %s  (run pcap_to_flow.py first)", args.data)
        sys.exit(1)

    # -- load & split --------------------------------------------------------
    log.info("Loading %s", args.data)
    df = pd.read_parquet(str(args.data), engine="pyarrow")
    log.info("Got %s rows, %d features + label", f"{len(df):,}", len(FEATURE_COLS))
    for lbl, cnt in df[LABEL_COL].value_counts().sort_index().items():
        tag = "benign" if lbl == 0 else "ddos  "
        log.info("  %s (%d) : %s", tag, lbl, f"{cnt:,}")

    X = df[FEATURE_COLS].fillna(0.0)   # NaN from single-packet flows
    y = df[LABEL_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y,
    )
    log.info("Split: %s train / %s test", f"{len(X_train):,}", f"{len(X_test):,}")

    # fit scaler on train only — don't leak test statistics
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # -- Logistic Regression -------------------------------------------------
    log.info("Training LogisticRegression ...")
    lr = LogisticRegression(max_iter=1000, solver="lbfgs", C=1.0,
                            random_state=args.random_state)
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)
    log_metrics("LogisticRegression", y_test, lr_pred)
    save_cm_plot(y_test, lr_pred, "LogisticRegression")

    # -- Random Forest -------------------------------------------------------
    # Switched to RF because AdaBoost struggled heavily with the 306/10k
    # split.  class_weight='balanced' does the trick without needing SMOTE —
    # sklearn adjusts sample weights inversely proportional to class freq.
    log.info("Training RandomForest (n_estimators=%d) ...", args.rf_estimators)
    rf = RandomForestClassifier(
        n_estimators=args.rf_estimators,
        class_weight='balanced',
        random_state=args.random_state,
    )
    rf.fit(X_train_s, y_train)
    rf_pred = rf.predict(X_test_s)
    log_metrics("RandomForestClassifier", y_test, rf_pred)
    save_cm_plot(y_test, rf_pred, "RandomForestClassifier")

    # -- comparison bar chart ------------------------------------------------
    models = ["LogisticRegression", "RandomForestClassifier"]
    scores = {
        "Accuracy":  [accuracy_score(y_test, p) for p in (lr_pred, rf_pred)],
        "Precision": [precision_score(y_test, p, zero_division=0) for p in (lr_pred, rf_pred)],
        "Recall":    [recall_score(y_test, p, zero_division=0) for p in (lr_pred, rf_pred)],
        "F1":        [f1_score(y_test, p, zero_division=0) for p in (lr_pred, rf_pred)],
    }

    x = range(len(scores))
    w = 0.3
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - w/2 for i in x], [v[0] for v in scores.values()], w, label=models[0])
    ax.bar([i + w/2 for i in x], [v[1] for v in scores.values()], w, label=models[1])
    ax.set_xticks(list(x))
    ax.set_xticklabels(scores.keys())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/model_comparison.png", dpi=150)
    plt.close()
    log.info("  -> plots/model_comparison.png")

    log.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
