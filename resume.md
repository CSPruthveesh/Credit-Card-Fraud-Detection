# Developer Resume Artifacts - Credit Card Fraud Detection Project

## Target Role: Core Software / Machine Learning Engineer (MLOps & Scale Focus)
- **Eliminated pipeline data leakage** by reordering the preprocessing flow to split the 150MB dataset before scaling and fitting `StandardScaler` instances strictly on training partitions, ensuring completely unbiased model generalization metrics.
- **Architected a local feature store** using Feast, declaring entities, batch Parquet sources, and Feature Views to unify schema definitions between offline training and online serving, eliminating train-serve skew.
- **Established experiment tracking and audit trails** by integrating MLflow in the training execution pipeline, logging GBDT hyperparameters, PR-AUC metrics, and financial scorecards directly to an SQLite backend database.
- **Implemented a population data drift monitor** using the Kolmogorov-Smirnov statistical test to compare incoming inference feature distributions with baseline training matrices, automatically flagging covariate shift.

## Target Role: Full-Stack / Product Engineer (User Features & Integration Focus)
- **Designed and built an interactive model playground and operations dashboard** using HTML, JavaScript, and CSS, incorporating two-way data synchronization and dynamic risk visualizations.
- **Created an AI explainability module** using TreeSHAP to compute local feature attributions on incoming transactions, automatically generating compliance-aligned risk reason codes for blocked payments.
- **Optimized user checkout experience** by deploying a low-latency FastAPI microservice that processes transaction payloads concurrently, returning fraud probability decisions in under 2ms.
- **Implemented a real-time ledger alert log feed** in the frontend client, displaying live-scored transaction metadata and dynamic visual action badges to reduce manual verification overhead.

## Target Role: Performance & Data Engineer (Algorithms & Cost Focus)
- **Reduced transaction scoring latency** from 2.28s to under 2ms (satisfying strict banking SLAs) by implementing a model pre-warming hook that initializes LightGBM's C++ booster thread pool on server startup.
- **Developed a custom mathematical Focal Loss objective** (calculating first and second-order derivatives) inside LightGBM to down-weight easy-to-classify legitimate transactions and focus learning on minority fraud cases, achieving an Average Precision of 0.8524.
- **Maximized business profit** by coding a cost-sensitive profit scorecard that weights False Negatives ($100 chargeback cost) against False/True Positives ($2 analyst review cost) to calibrate optimal classification thresholds.
- **Engineered a graph representation learning pipeline** using PyTorch Geometric (PyG), mapping tabular cardholder and merchant transactions into a heterogeneous bipartite graph to capture multi-hop relational fraud networks.

## Target Role: Tech Lead / Engineering Manager (Ownership & Delivery Focus)
- **Spearheaded the modular transition** of a monolithic exploratory Google Colab notebook into a structured, production-grade ML repository consisting of dedicated API, training, feature store, and monitoring packages.
- **Orchestrated end-to-end version control synchronization** across local and remote environments, staging and pushing clean, production-grade iterations of the code directly to the GitHub main branch.
- **Designed operational alerting guardrails** by setting drift threshold triggers that identify statistical feature decay, warning system administrators to initiate retraining pipelines.
- **Led the development and optimization of the API interface**, resolving critical LightGBM v4.0 integration errors and API internal failures during development to deliver a stable transaction scoring service.
