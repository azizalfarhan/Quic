"""Project-wide configuration.

Single source of truth for paths, the canonical feature list, the
random seed (so every script lands on the same train/test split), and
the logging setup. If a constant shows up in more than one file, it
belongs here — not duplicated across scripts.
"""

from __future__ import annotations

import logging
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT   = Path(__file__).resolve().parent

DATA_DIR       = PROJECT_ROOT / "data"
RAW_PCAP_DIR   = DATA_DIR / "raw_pcap"
BENIGN_DIR     = RAW_PCAP_DIR / "benign"
DDOS_DIR       = RAW_PCAP_DIR / "ddos"
PROCESSED_DIR  = DATA_DIR / "processed"
PARQUET_PATH   = PROCESSED_DIR / "quic_dataset.parquet"

PLOTS_DIR      = PROJECT_ROOT / "plots"

ARTIFACTS_DIR  = PROJECT_ROOT / "models" / "artifacts"
RF_ARTIFACT    = ARTIFACTS_DIR / "rf.joblib"
SGD_ARTIFACT   = ARTIFACTS_DIR / "sgd.joblib"
SCALER_ARTIFACT = ARTIFACTS_DIR / "scaler.joblib"

# ---------------------------------------------------------------------------
# Features / labels
# ---------------------------------------------------------------------------
# Strictly the 13 statistical features. No IPs / MACs / Ports — see the
# scientific note in pcap_to_flow.py for the reasoning.
FEATURE_COLS: list[str] = [
    "flow_duration",
    "fwd_pkts", "bwd_pkts",
    "fwd_bytes", "bwd_bytes",
    "iat_mean", "iat_std", "iat_max", "iat_min",
    "pkt_len_mean", "pkt_len_std", "pkt_len_max", "pkt_len_min",
]
LABEL_COL  = "label"
SOURCE_COL = "source"

# ---------------------------------------------------------------------------
# Train / split / SHAP knobs
# ---------------------------------------------------------------------------
# Fixed seed across the whole project: train_phase3 and explain_shap must
# regenerate IDENTICAL splits, otherwise SHAP would explain data the model
# never saw.
RANDOM_STATE         = 42
TEST_SIZE            = 0.2
SHAP_BACKGROUND_SIZE = 100   # LinearExplainer background sample

# ---------------------------------------------------------------------------
# Flow-extraction tunables (pcap_to_flow.py)
# ---------------------------------------------------------------------------
FLOW_TIMEOUT_SEC   = 60.0
FLUSH_EVERY        = 10_000
PROGRESS_EVERY     = 100_000
MAX_LIST_LEN       = 500
PARQUET_BATCH_SIZE = 5_000

# Sources we recognise from the folder layout. Anything else falls back
# to "original" — keeps old Phase 2 captures working without a rename.
KNOWN_SOURCES = ("kaggle", "client", "original")

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FORMAT  = "%(asctime)s - %(levelname)s - %(message)s"
LOG_DATEFMT = "%Y-%m-%d %H:%M:%S"


def configure_logging(level: int = logging.INFO) -> None:
    """Call once at the start of every executable script.

    Idempotent — safe to re-invoke; existing handlers stay put.
    """
    root = logging.getLogger()
    if root.handlers:
        return
    logging.basicConfig(level=level, format=LOG_FORMAT, datefmt=LOG_DATEFMT)
