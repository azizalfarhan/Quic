"""train_mlp.py — PyTorch MLP for QUIC DDoS flow classification.

Complements the sklearn baselines in train_baseline.py with a simple
neural-network approach.  The architecture is deliberately shallow (two
hidden layers) because we're working with 13 tabular features, not images —
going deeper just overfits faster.

Imbalance strategy:
    We use pos_weight in BCEWithLogitsLoss to down-weight the majority class
    (DDoS = label 1).  This is the PyTorch equivalent of what class_weight=
    'balanced' does for the Random Forest.  No SMOTE, no oversampling.

Outputs per-epoch accuracy/precision/recall on the test set and saves three
line plots to plots/ for the thesis.
"""

from __future__ import annotations

import logging
import sys
import argparse
from pathlib import Path
from typing import List, Tuple

import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix,
)

# Same 13 features as in train_baseline.py — no IPs/ports
FEATURE_COLS = [
    "flow_duration",
    "fwd_pkts", "bwd_pkts",
    "fwd_bytes", "bwd_bytes",
    "iat_mean", "iat_std", "iat_max", "iat_min",
    "pkt_len_mean", "pkt_len_std", "pkt_len_max", "pkt_len_min",
]
LABEL_COL = "label"

# Defaults — all overridable via CLI args
EPOCHS      = 50
BATCH_SIZE  = 256
LR          = 1e-3
TEST_SIZE   = 0.2
RANDOM_STATE = 42

log = logging.getLogger(__name__)


class QuicMLP(nn.Module):
    """Shallow feedforward net: 13 -> 64 -> 32 -> 1.

    Two hidden layers with ReLU + dropout.  Kept simple on purpose — for
    tabular data this size, more layers just mean more overfitting headaches.
    Output is a raw logit (sigmoid is inside BCEWithLogitsLoss).
    """

    def __init__(self, n_features: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def main() -> None:
    parser = argparse.ArgumentParser(description="train MLP on QUIC flow dataset")
    parser.add_argument("--data", type=Path,
                        default=Path(__file__).parent / "data" / "processed" / "quic_dataset.parquet")
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--test-size", type=float, default=TEST_SIZE)
    parser.add_argument("--random-state", type=int, default=RANDOM_STATE)
    args = parser.parse_args()

    if not args.data.exists():
        log.error("Dataset not found: %s", args.data)
        sys.exit(1)

    # -- load & split --------------------------------------------------------
    log.info("Loading %s", args.data)
    df = pd.read_parquet(str(args.data), engine="pyarrow")
    log.info("Got %s rows", f"{len(df):,}")

    X = df[FEATURE_COLS].fillna(0.0).values
    y = df[LABEL_COL].values

    n_benign = int((y == 0).sum())
    n_ddos   = int((y == 1).sum())
    log.info("Classes: %s benign / %s ddos", f"{n_benign:,}", f"{n_ddos:,}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=args.random_state, stratify=y,
    )
    log.info("Split: %s train / %s test", f"{len(X_train):,}", f"{len(X_test):,}")

    # fit on train only
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    # -- tensors -------------------------------------------------------------
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_test_t  = torch.tensor(X_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=args.batch_size, shuffle=True,
    )

    # pos_weight < 1 because ddos (label=1) is the MAJORITY class here.
    # This tells the loss to care less about getting ddos right and more
    # about not missing the rare benign flows.
    pw = torch.tensor([n_benign / n_ddos], dtype=torch.float32)
    log.info("pos_weight = %.4f  (benign/ddos ratio)", pw.item())

    # -- model ---------------------------------------------------------------
    model     = QuicMLP(n_features=len(FEATURE_COLS))
    criterion = nn.BCEWithLogitsLoss(pos_weight=pw)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # track metrics every epoch for the thesis plots
    history: dict[str, List[float]] = {
        "acc": [], "prec": [], "rec": [],
    }

    log.info("Training for %d epochs ...", args.epochs)

    for epoch in range(1, args.epochs + 1):
        # -- train -----------------------------------------------------------
        model.train()
        epoch_loss = 0.0
        for xb, yb in train_loader:
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(xb)
        epoch_loss /= len(X_train_t)

        # -- evaluate on test set every epoch --------------------------------
        # (yes, every epoch — we need the full learning curve for the thesis)
        model.eval()
        with torch.no_grad():
            preds = (torch.sigmoid(model(X_test_t)) >= 0.5).int().squeeze().numpy()

        acc  = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, zero_division=0)
        rec  = recall_score(y_test, preds, zero_division=0)
        history["acc"].append(acc)
        history["prec"].append(prec)
        history["rec"].append(rec)

        if epoch == 1 or epoch % 5 == 0:
            log.info("  Epoch %3d/%d | loss %.4f | acc %.4f | prec %.4f | rec %.4f",
                     epoch, args.epochs, epoch_loss, acc, prec, rec)

    # -- final report --------------------------------------------------------
    model.eval()
    with torch.no_grad():
        preds = (torch.sigmoid(model(X_test_t)) >= 0.5).int().squeeze().numpy()

    cm = confusion_matrix(y_test, preds)
    log.info("")
    log.info("=" * 56)
    log.info("  Model : MLP (PyTorch)")
    log.info("=" * 56)
    log.info("  Accuracy  : %.4f", accuracy_score(y_test, preds))
    log.info("  Precision : %.4f", precision_score(y_test, preds, zero_division=0))
    log.info("  Recall    : %.4f", recall_score(y_test, preds, zero_division=0))
    log.info("  F1-Score  : %.4f", f1_score(y_test, preds, zero_division=0))
    log.info("")
    log.info("  Confusion Matrix (rows=actual, cols=predicted):")
    log.info("               Pred 0    Pred 1")
    log.info("  Actual  0   %8s  %8s", f"{cm[0,0]:,}", f"{cm[0,1]:,}")
    log.info("  Actual  1   %8s  %8s", f"{cm[1,0]:,}", f"{cm[1,1]:,}")

    # -- plots for thesis ----------------------------------------------------
    Path("plots").mkdir(exist_ok=True)
    epochs_x = list(range(1, args.epochs + 1))

    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14,
                         "axes.labelsize": 12, "figure.facecolor": "white"})

    plot_specs: List[Tuple[str, str, str, str]] = [
        ("acc",  "Accuracy",  "#2563eb", "plots/mlp_accuracy_epoch.png"),
        ("prec", "Precision", "#16a34a", "plots/mlp_precision_epoch.png"),
        ("rec",  "Recall",    "#dc2626", "plots/mlp_recall_epoch.png"),
    ]

    for key, label, color, path in plot_specs:
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(epochs_x, history[key], lw=2, color=color)
        ax.set(xlabel="Epoch", ylabel=label, ylim=(0, 1.05),
               title=f"MLP — Test {label} per Epoch")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        log.info("  -> %s", path)

    log.info("Done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%H:%M:%S",
    )
    main()
