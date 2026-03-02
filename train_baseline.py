# train_baseline.py
# loads the parquet dataset from pcap_to_flow.py and trains two baseline classifiers
# outputs accuracy, precision, recall, f1 and confusion matrix for each model

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import matplotlib.pyplot as plt
import seaborn as sns

# feature columns — no IPs or ports, only statistical flow features
FEATURE_COLS = [
    "flow_duration",
    "fwd_pkts", "bwd_pkts",
    "fwd_bytes", "bwd_bytes",
    "iat_mean", "iat_std", "iat_max", "iat_min",
    "pkt_len_mean", "pkt_len_std", "pkt_len_max", "pkt_len_min",
]
LABEL_COL = "label"


def print_metrics(model_name, y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    print()
    print("=" * 56)
    print(f"  Model : {model_name}")
    print("=" * 56)
    print(f"  Accuracy  : {accuracy_score(y_true, y_pred):.4f}")
    print(f"  Precision : {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  Recall    : {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"  F1-Score  : {f1_score(y_true, y_pred, zero_division=0):.4f}")
    print()
    print("  Confusion Matrix  (rows = actual, cols = predicted):")
    print(f"               Pred 0    Pred 1")
    print(f"  Actual  0   {cm[0, 0]:>8,}  {cm[0, 1]:>8,}")
    print(f"  Actual  1   {cm[1, 0]:>8,}  {cm[1, 1]:>8,}")


def save_plots(y_true, y_pred, model_name):
    # plot confusion matrix as heatmap and save to plots/
    Path("plots").mkdir(exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt=",d", cmap="Blues",
                xticklabels=["Benign", "DDoS"],
                yticklabels=["Benign", "DDoS"])
    # FIXME: maybe switch to grayscale cmap for print version of thesis
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Confusion Matrix - {model_name}")
    plt.tight_layout()
    plt.savefig(f"plots/cm_{model_name}.png", dpi=150)
    plt.close()
    print(f"  -> saved plots/cm_{model_name}.png")


def main():
    parser = argparse.ArgumentParser(description="train baseline models on QUIC flow dataset")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data" / "processed" / "quic_dataset.parquet",
        help="path to parquet dataset",
    )
    parser.add_argument("--test-size", type=float, default=0.2, help="test split size")
    parser.add_argument("--random-state", type=int, default=42, help="random seed")
    parser.add_argument("--adaboost-estimators", type=int, default=100, help="n_estimators for AdaBoost")
    args = parser.parse_args()

    if not args.data.exists():
        print(f"[ERROR] Dataset not found: {args.data}")
        print("Run pcap_to_flow.py first to generate the dataset.")
        sys.exit(1)

    print(f"Loading dataset: {args.data}")
    df = pd.read_parquet(str(args.data), engine="pyarrow")
    # print(df.head())
    print(f"Loaded {len(df):,} rows, {len(df.columns)} columns")
    print("Label distribution:")
    for lbl, cnt in df[LABEL_COL].value_counts().sort_index().items():
        name = "benign (0)" if lbl == 0 else "ddos   (1)"
        print(f"  {name} : {cnt:,}")

    X = df[FEATURE_COLS].copy()
    y = df[LABEL_COL].copy()

    # fill NaN for single-packet flows (no IAT -> stats are 0)
    X = X.fillna(0.0)

    # stratified split to preserve class ratio (dataset is quite imbalanced)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )
    print(f"\nTrain samples : {len(X_train):,}")
    print(f"Test  samples : {len(X_test):,}")

    # fit scaler only on train set, not test (avoid leakage)
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # TODO: try higher C values or l1 penalty if underfitting
    print("\nTraining LogisticRegression ...")
    lr = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        C=1.0,
        random_state=args.random_state,
    )
    lr.fit(X_train_s, y_train)
    lr_pred = lr.predict(X_test_s)
    print_metrics("LogisticRegression", y_test, lr_pred)
    save_plots(y_test, lr_pred, "LogisticRegression")

    print(f"\nTraining AdaBoostClassifier (n_estimators={args.adaboost_estimators}) ...")
    ada = AdaBoostClassifier(
        n_estimators=args.adaboost_estimators,
        random_state=args.random_state,
        algorithm="SAMME",  # explicit: SAMME.R was deprecated in sklearn 1.4 and removed in 1.6
    )
    ada.fit(X_train_s, y_train)
    ada_pred = ada.predict(X_test_s)
    print_metrics("AdaBoostClassifier", y_test, ada_pred)
    save_plots(y_test, ada_pred, "AdaBoostClassifier")
    # TODO: maybe add random forest as third model for comparison

    # quick model comparison chart
    models = ["LogisticRegression", "AdaBoostClassifier"]
    metrics = {
        "Accuracy":  [accuracy_score(y_test, lr_pred),  accuracy_score(y_test, ada_pred)],
        "Precision": [precision_score(y_test, lr_pred, zero_division=0), precision_score(y_test, ada_pred, zero_division=0)],
        "Recall":    [recall_score(y_test, lr_pred, zero_division=0),    recall_score(y_test, ada_pred, zero_division=0)],
        "F1":        [f1_score(y_test, lr_pred, zero_division=0),        f1_score(y_test, ada_pred, zero_division=0)],
    }

    x = range(len(metrics))
    width = 0.3
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar([i - width / 2 for i in x], [v[0] for v in metrics.values()], width, label=models[0])
    ax.bar([i + width / 2 for i in x], [v[1] for v in metrics.values()], width, label=models[1])
    ax.set_xticks(list(x))
    ax.set_xticklabels(metrics.keys())
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    plt.tight_layout()
    plt.savefig("plots/model_comparison.png", dpi=150)
    plt.close()
    print("  -> saved plots/model_comparison.png")

    print("\nTraining complete.\n")


if __name__ == "__main__":
    main()
