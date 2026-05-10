# Behavioural Risk Scoring for Financial Transactions

A UEBA-inspired risk scoring framework for transactional fraud detection, combining a supervised gradient-boosting model with unsupervised anomaly detection, calibrated probabilities, and per-transaction SHAP explanations. Built on the PaySim simulated mobile-money dataset.

This repository accompanies the Master's thesis *Design and Evaluation of a Behavioural Risk Scoring Framework for Financial Transactions using Anomaly Detection* (Universitat Oberta de Catalunya, MSc in Data Science).

**Author:** Marc Pérez Bernús
**Supervisor:** Blas Torregrosa García

---

## What this project does

Fraud detection in payments is a rare-event problem (≈0.13% positive rate in PaySim, often below 0.3% in real card data) where headline accuracy is meaningless and the operational cost of a false positive is very different from that of a missed fraud. The project addresses this in three layers:

1. **Detection.** A supervised XGBoost model captures known fraud patterns. An Isolation Forest captures deviations from typical transactional behaviour without using labels at training time. Their scores are combined into a single risk score.
2. **Decision.** The continuous score is mapped to four operational bands (Low / Medium / High / Critical), each linked to an action (allow / monitor / review / block). The threshold is chosen by minimising expected operational cost rather than picked from a default.
3. **Explanation.** Calibrated probabilities make the bands interpretable as probability cuts. SHAP attributions provide per-transaction reasons that an investigator (or, under FADP / GDPR Article 22, an affected customer) can be shown.

The supervised model alone is more accurate; the hybrid is more defensible operationally because the anomaly component doesn't depend on the labels matching future fraud patterns.

---

## Dataset

[PaySim](https://www.kaggle.com/datasets/ealaxi/paysim1) — a simulated mobile-money dataset built from real transaction patterns by Lopez-Rojas et al. (2016). 6.36M transactions, 11 columns, 0.13% fraud rate.

PaySim is used here because publicly available, labelled, real-world payments data does not exist at this scale due to privacy and regulatory constraints. Two consequences of that choice are addressed throughout the notebooks:

- **Fraud is structural in PaySim.** Fraudulent transactions are limited to `TRANSFER` and `CASH_OUT` types and follow a deterministic accounting pattern (sender's balance is fully drained). Tree-based models can essentially memorise this rule, which inflates PR-AUC.
- **The headline supervised PR-AUC of ~0.998 is not the honest number to cite.** An ablation that removes the balance-residual features puts PR-AUC at 0.97, which is what would generalise to data without the simulator's hard rule. The thesis cites 0.97.

The dataset is not committed to this repository. Download it from Kaggle and place it at `data/raw/PS_20174392719_1491204439457_log.csv`.

---

## Pipeline

The work is organised as five sequential notebooks. Each one assumes the previous has been run (artefacts are persisted to disk).

| Notebook | Purpose | Key output |
|---|---|---|
| `01_eda.ipynb` | Distributions, class imbalance, correlation structure, identification of the balance-residual signal | Findings that shape feature engineering |
| `02_feature_engineering.ipynb` | Stratified train/test split *before* preprocessing, log transforms, balance-residual features, RobustScaler | `data/processed/train.csv`, `test.csv`, `scaler.pkl` |
| `03_model_baseline.ipynb` | Supervised baselines: dummy, logistic regression, Random Forest, XGBoost, LightGBM, CatBoost. Ablation on balance residuals. SHAP summary | `xgb_baseline.pkl` |
| `04_anomaly_detection.ipynb` | Isolation Forest, OCSVM, LOF, ECOD, COPOD, autoencoder. Comparison on PR-AUC and operational cost | `isolation_forest.pkl` |
| `05_risk_scoring_and_explainability.ipynb` | Probability calibration, score combination, risk bands, cost-based threshold selection, SHAP waterfall plots, regulatory context | `reports/figures/*.png` |

---

## Methodology highlights

A few decisions worth flagging because they're the things that would be checked in a model validation review:

**Train/test split happens before any preprocessing.** Scaler statistics, encoders, and any cross-feature aggregates are fit on training data only. This eliminates a common source of optimistic results in fraud notebooks.

**`isFlaggedFraud` and account identifiers are dropped.** `isFlaggedFraud` is a hand-coded rule on the same target signal and would act as label leakage. Account names are unique identifiers without predictive value.

**RobustScaler over StandardScaler.** Two engineered features have skewness above 29; mean and standard deviation are unreliable summaries on that distribution. Tree models don't need scaling, but it matters for the logistic baseline and for OCSVM / LOF.

**PR-AUC over ROC-AUC as the primary ranking metric.** Under 0.13% prevalence, ROC-AUC is dominated by the false-positive rate against a 770× larger negative class and produces optimistically high numbers (0.995 for plain logistic regression). PR-AUC reflects the operational trade-off between missed fraud and false alarms.

**Probability calibration with `CalibratedClassifierCV` (isotonic).** XGBoost scores are good for ranking but not calibrated as probabilities. Calibration is necessary before mapping bands to probability cuts and before computing expected cost in monetary terms.

**Cost-based threshold rather than default 0.5.** The threshold is chosen by minimising expected cost on a stated unit-cost matrix (false-negative cost = average fraud loss; false-positive cost = analyst time + customer friction). The 0.5 default is reported only for comparability.

**Production-track models persisted: XGBoost + Isolation Forest.** OCSVM and LOF are competitive on detection but require subsampling at this data volume, which makes them less attractive operationally. ECOD / COPOD would be acceptable substitutes for IForest on stability grounds.

---

## Headline results

Reported on the held-out test set (1.27M transactions, stratified):

| Model | PR-AUC | Notes |
|---|---|---|
| Dummy (stratified) | 0.001 | Floor; equal to fraud base rate |
| Logistic Regression | 0.394 | Linear-accessible signal in engineered features |
| Random Forest | 0.998 | Inflated by deterministic simulator pattern |
| XGBoost | 0.990 | Production-track supervised model |
| LightGBM | 0.997 | |
| CatBoost | 0.998 | |
| **XGBoost (no balance residuals — ablation)** | **0.97** | **Honest figure for thesis citation** |
| Isolation Forest (unsupervised) | reported in notebook 04 | Not directly comparable to supervised PR-AUC |

The gap between supervised (0.97 honest) and the best anomaly detector quantifies what the labels are doing. The hybrid risk score keeps both because the anomaly component is more robust to fraud-pattern drift the labels haven't seen.

---

## Regulatory context

The risk band design and the SHAP layer are aligned with three regulatory regimes that apply to an automated fraud-decision system in a Swiss-banking context:

- **FINMA** circulars on operational risk: independent model validation, documentation, ongoing monitoring. Calibration analysis, PSI instrumentation, and the cost-sensitive framework are the kinds of artefacts validation expects.
- **Swiss FADP** and **GDPR Article 22**: rights related to automated decision-making and explanation. Per-transaction SHAP attributions support this operationally, not only academically.
- **EU AI Act** (high-risk obligations applying from 2026): documentation, logging, and human oversight. Automated blocks are reserved for the Critical band; the High band routes through human review before any customer-facing action.

---

## Repository structure

```
.
├── README.md
├── LICENSE
├── requirements.txt
├── .gitignore
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_engineering.ipynb
│   ├── 03_model_baseline.ipynb
│   ├── 04_anomaly_detection.ipynb
│   └── 05_risk_scoring_and_explainability.ipynb
├── src/
│   ├── features/
│   │   └── build_features.py     # feature logic shared with API
│   ├── models/                   # persisted models and scalers (gitignored)
│   └── api/
│       └── main.py               # FastAPI scoring prototype
├── data/
│   ├── raw/                      # PaySim CSV (gitignored, see Dataset section)
│   ├── processed/                # train/test splits (gitignored)
│   └── README.md                 # data dictionary and source
├── reports/
│   └── figures/                  # plots referenced in notebooks and thesis
└── docs/
    ├── TFM_final.pdf             # full thesis (when finalised)
    └── SoA.pdf                   # state-of-the-art review
```

---

## Reproducing the results

```bash
# 1. Clone and create environment
git clone <this-repo-url>
cd <repo-name>
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Place the PaySim CSV at data/raw/PS_20174392719_1491204439457_log.csv

# 3. Run the notebooks in order
jupyter lab notebooks/
```

Random seeds are fixed to `42` throughout. The supervised baselines and Isolation Forest run on a single CPU machine in reasonable time on the full dataset; the autoencoder benefits from a GPU but is not required for the headline results.

---

## API prototype

`src/api/main.py` is a FastAPI service that loads the persisted scaler, feature columns, XGBoost model, and Isolation Forest, applies the same feature transformation used in training, and returns the risk score, band, action, and top SHAP contributors for a single transaction. It's a single-process demo and not a production deployment — see *Limitations* below.

```bash
uvicorn src.api.main:app --reload
# POST /score with a JSON body matching the PaySim transaction schema
```

---

## Limitations

The thesis discusses these in detail. In short:

- **Synthetic data.** PaySim's fraud pattern is more deterministic than real fraud. Headline supervised PR-AUC reflects this; the ablated 0.97 figure is what generalises.
- **No temporal validation.** The split is stratified by class but not by time. PaySim's `step` field encodes hours, but the dataset is short enough that a strict temporal hold-out would leave too few fraud cases to evaluate. A real deployment requires walk-forward validation against concept drift.
- **No fairness audit.** PaySim has no demographic features, so the question doesn't arise here. A real banking dataset would require pre-deployment bias testing on protected attributes and on proxies.
- **No streaming infrastructure.** The FastAPI prototype is single-process. Production would need a model registry, A/B routing, latency budgets in the low-millisecond range, and a feature store guaranteeing train/serve consistency.
- **No champion/challenger evaluation.** The hybrid score weights are fixed in the notebook. In deployment they would be tuned against business metrics under controlled rollout.

---

## License

[MIT]

---

## Citation

If you reference this work:

```bibtex
@mastersthesis{perezbernus2026risk,
  author = {Pérez Bernús, Marc},
  title  = {Design and Evaluation of a Behavioural Risk Scoring Framework for Financial Transactions using Anomaly Detection},
  school = {Universitat Oberta de Catalunya},
  year   = {2026}
}
```