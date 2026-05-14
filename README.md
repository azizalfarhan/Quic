# QUIC DDoS Detection — Phase 3

Machine-learning pipeline for detecting DDoS traffic in QUIC (UDP) flows.
Phase 3 enriches the dataset with public Kaggle captures, trains a Random
Forest alongside the client's SGD Logistic Regression, and produces SHAP
explanations for both models.

## What Phase 3 adds

- **Data enrichment**: ingests Kaggle's `adam357/quic-network-capture-data`
  alongside the original Phase 2 captures. Lifts the benign class from
  306 flows to **838** (~2.7x) without resorting to SMOTE / oversampling.
- **Source tracking**: every flow carries a `source` tag (`kaggle`,
  `client`, `original`) so per-source error analysis is one groupby away.
- **Modular training**: a single `train_phase3.py` runs both estimators
  on identical splits with identical scaler state.
- **Explainable AI**: `explain_shap.py` uses `TreeExplainer` for the
  Random Forest and `LinearExplainer` for the SGD LogReg — both exact,
  no approximations.

## Project layout

```
Quic/
├── config.py              # paths, feature list, RNG seed, logging
├── pcap_to_flow.py        # PCAP -> parquet (13 statistical features)
├── setup_kaggle.py        # curated download of the Kaggle dataset
├── train_phase3.py        # train + evaluate RF + SGD, save artifacts
├── explain_shap.py        # SHAP plots for both trained models
├── models/
│   ├── base.py            # load/split/scale, evaluation, artifact I/O
│   ├── random_forest.py
│   └── client_sgd.py      # SGD LogReg stub — swap kwargs on client delivery
├── data/                  # gitignored — see "Distribution"
├── models/artifacts/      # gitignored — see "Distribution"
└── plots/                 # gitignored — see "Distribution"
```

## Installation

Python 3.13 is what this was developed and tested on. Earlier 3.10+
versions should work but are not covered by the pinned wheel versions.

```powershell
pip install -r requirements.txt
```

The Kaggle CLI also needs an API token at
`%USERPROFILE%\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json`
(Unix). Create it from your Kaggle account settings.

## How to run

```powershell
# 1. Download the curated Kaggle subset (~2.5 GB, 18 PCAPs)
python setup_kaggle.py

# 2. Parse every PCAP under data/raw_pcap/ into a single parquet
python pcap_to_flow.py

# 3. Train RandomForest + Client SGD LogReg on the same split,
#    save artifacts to models/artifacts/, write CM + comparison plots.
python train_phase3.py

# 4. Generate SHAP plots (bar + beeswarm) for both models.
python explain_shap.py
```

Each script accepts `--help` for its CLI flags. Defaults come from
`config.py`, so for the standard pipeline no flags are needed.

## Outputs

After step 4, `plots/` contains seven PNGs:

| File | What it shows |
|---|---|
| `cm_RandomForest.png` | Confusion matrix — Random Forest |
| `cm_ClientSGD.png` | Confusion matrix — Client SGD LogReg |
| `model_comparison.png` | Side-by-side Accuracy/Precision/Recall/F1 |
| `shap_rf_bar.png` | Global feature importance (RF, class 1 = DDoS) |
| `shap_rf_beeswarm.png` | Per-sample SHAP distribution (RF) |
| `shap_sgd_bar.png` | Global feature importance (SGD) |
| `shap_sgd_beeswarm.png` | Per-sample SHAP distribution (SGD) |

`models/artifacts/` holds `rf.joblib`, `sgd.joblib`, `scaler.joblib`.

## Feature engineering

The 13 statistical features (see `config.FEATURE_COLS`) are computed
per bidirectional flow and exclude all identifiers (IPs, MACs, ports) by
design — see the scientific note in `pcap_to_flow.py`. The model is
forced to learn behavioral patterns rather than memorising attacker IPs.

## Imbalance handling

Both models use `class_weight='balanced'` from scikit-learn. This
reweights samples by inverse class frequency and produces the same
effect as `pos_weight` in the loss without needing synthetic
oversampling on a dataset this size.

## Distribution

The repository tracks only code. The trained models, processed parquet
and PCAP captures are distributed separately:

- **Code**: this Git repository.
- **Artifacts + processed data + plots**: `Aziz_Phase3_Results.zip` on
  Google Drive.
- **Raw PCAPs**: download via `setup_kaggle.py` (Kaggle) and the
  client's own captures, dropped into `data/raw_pcap/`.

## Handing over to the client

When the client delivers his final SGD hyperparameters:

1. Update the kwargs in `models/client_sgd.py::build_client_sgd()`.
2. Re-run `train_phase3.py` and `explain_shap.py`.
3. Compare metrics and SHAP plots against the previous run.

No other file needs to change.
