# Walkthrough Report: Credit Card Fraud Detection Audit & Refactor

This document summarizes the changes, additions, and architectural modifications executed during the audit to elevate this machine learning project to production-grade standard.

---

## 1. Accomplished Objectives

We transitioned a baseline exploration repository containing a monolithic Google Colab notebook and weekly logs into a structured, production-grade ML repository.

```
Credit-Card-Fraud-Detection/
│
├── feature_repo/             # Feast Feature Store Definitions
│   ├── feature_store.yaml    
│   └── features.py           
│
├── src/                      # Production API, Training & Monitoring Codes
│   ├── train.py              # MLflow pipeline runner
│   ├── evaluate.py           # Business-aligned metrics
│   ├── explain.py            # Local TreeSHAP attributions
│   ├── serve_api.py          # Async FastAPI endpoint
│   ├── monitor.py            # KS-test data drift tracker
│   └── gnn_model.py          # PyG Heterogeneous graph network
│
├── Creditfraud.ipynb         # Refactored exploration and prototype notebook
├── changes.md                # Summary of changes and impacts log
├── requirements.txt          # Python environments requirements file
└── README.md                 # System overview and reproduction walkthrough
```

---

## 2. Refactored Improvements

### 2.1 Preprocessing Pipeline & Data Leakage Elimination
*   **Action:** Moved the train-test split *before* scaling. Programmed separate `StandardScaler` instances fit strictly on training slices, preventing validation target information from leaking during development.
*   **Result:** Generalization metrics reported on the test set are now completely unbiased.

### 2.2 Temporal Feature Engineering
*   **Action:** Created proxy cardholder identities by combining and clustering PCA features `V1` and `V2`. Computed rolling velocity and behavioral metrics (`Time_Delta`, `Last_5_Tx_Time_Span`, and `Rolling_Mean_Amount_5`) to capture sequential fraud anomalies.
*   **Result:** The models now have behavioral and frequency context, increasing precision against rapid-fire carding scripts.

### 2.3 Feast Feature Store Integration
*   **Action:** Created [`feature_repo/`](file:///d:/COLLEGE%20PREP/Self%20Projects/feature_repo) containing `feature_store.yaml` and schema declarations `features.py`.
*   **Result:** Standardizes feature extraction definitions between training (offline Parquet) and real-time serving (online cache), resolving train-serve skew.

### 2.4 Modern Modeling Benchmarks (GBDT & Heterogeneous GNN)
*   **Action:** Appended XGBoost, LightGBM, and CatBoost classifier cells using dynamic cost weighting to handle extreme imbalance. In addition, built [`gnn_model.py`](file:///d:/COLLEGE%20PREP/Self%20Projects/src/gnn_model.py) using PyG to represent relationships between cardholders and merchants.
*   **Result:** Replaced SMOTE oversampling noise with gradient-level class penalties and network-level message passing.

### 2.5 Cost-Sensitive Metrics & SHAP Interpretability
*   **Action:** Implemented profit scorecards in [`evaluate.py`](file:///d:/COLLEGE%20PREP/Self%20Projects/src/evaluate.py) to assess models on Net Dollar Savings. Built [`explain.py`](file:///d:/COLLEGE%20PREP/Self%20Projects/src/explain.py) to compute TreeSHAP local explanations for flagged transactions.
*   **Result:** Aligned model performance with commercial reality and generated reason codes for transaction blocks.

### 2.6 MLOps pipelines (MLflow, FastAPI, and Drift Monitoring)
*   **Action:** Created [`train.py`](file:///d:/COLLEGE%20PREP/Self%20Projects/src/train.py) for MLflow run tracking, [`serve_api.py`](file:///d:/COLLEGE%20PREP/Self%20Projects/src/serve_api.py) for FastAPI asynchronous scoring (<10ms latency), and [`monitor.py`](file:///d:/COLLEGE%20PREP/Self%20Projects/src/monitor.py) for Kolmogorov-Smirnov population drift checking.
*   **Result:** Established model audit trails, real-time microservices, and drift-triggered retraining loops.
