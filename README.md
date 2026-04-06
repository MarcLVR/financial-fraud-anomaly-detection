## Project Overview
Behavioural fraud detection using the PaySim dataset.

## Dataset
- PaySim (synthetic financial transactions)
- ~6.3M transactions
- Highly imbalanced fraud distribution

## Project Structure
- notebooks/
  - 01_eda.ipynb
  - 02_feature_engineering.ipynb
  - 03_model_baseline.ipynb
  - 04_anomaly_detection.ipynb

## Workflow
EDA → Feature Engineering → Baseline Models → Anomaly Detection

## Key Insights
- Fraud follows structured balance constraints
- Transaction type is a strong discriminator
- Amount alone is insufficient

## Next Steps
- Model development
- Evaluation under class imbalance
- Unsupervised anomaly detection
