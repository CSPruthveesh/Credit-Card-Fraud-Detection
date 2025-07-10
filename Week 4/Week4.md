# ✅ Week 4: Baseline Modeling & Evaluation

---

## 🎯 Objective

To train and evaluate baseline machine learning models (**Logistic Regression**, **Decision Tree**, **Random Forest**) using the **original imbalanced dataset**.  
Focus is on **Recall**, **Precision**, **F1-score**, and **ROC-AUC**, rather than accuracy, which can be **misleading** in highly imbalanced datasets.

---

## 📊 Model-wise Performance Insights

### 🔹 1. Logistic Regression (`class_weight='balanced'`)

- **Recall (Fraud)**: ✅ 91.84% — Excellent fraud detection  
- **Precision (Fraud)**: ❌ 6.09% — Very poor, high false positives  
- **F1-score (Fraud)**: ❌ 0.1141 — Low due to imbalance in precision/recall  
- **ROC-AUC**: ✅ 0.9722 — Excellent separation between classes  
- **Accuracy**: 97.55% — High, but **misleading**

📌 **Verdict**:  
Prioritizes catching **all frauds**, but at the cost of flagging **too many legitimate transactions** as fraud.  
Not viable alone without **filtering or review pipeline**.

---

### 🔹 2. Decision Tree (`class_weight='balanced'`)

- **Recall (Fraud)**: ⚠️ 71.43% — Catches most but not all frauds  
- **Precision (Fraud)**: ✅ 70.71% — Good balance  
- **F1-score (Fraud)**: ✅ 0.7107 — Strong trade-off  
- **ROC-AUC**: ⚠️ 0.8569 — Moderate separation ability  
- **Accuracy**: 99.90% — Still **misleading**

📌 **Verdict**:  
Better **fraud/no-fraud balance**, but less powerful than logistic or random forest in **overall discrimination**.

---

### 🔹 3. Random Forest (`class_weight='balanced'`)

- **Recall (Fraud)**: ✅ 75.51%  
- **Precision (Fraud)**: ✅ 96.10% — Very few false alarms  
- **F1-score (Fraud)**: ✅ 0.8457 — Best so far  
- **ROC-AUC**: ✅ 0.9581  
- **Accuracy**: ✅ 99.95% — High but not relied upon

📌 **Verdict**:  
**Best baseline performer overall** — excellent **precision**, strong **F1-score**, and decent **recall**.  
A good candidate for **real-world deployment** or as a foundation for **future tuning**.

---

## 🧠 Important Insight: Accuracy is Misleading

Due to **class imbalance** (only ~0.17% frauds), **accuracy isn't meaningful**.  
A model can achieve 99%+ accuracy by predicting all transactions as **non-fraud**.

👉 Instead, we focus on:

- **Recall** → Catching as many frauds as possible  
- **Precision** → Avoiding false alarms  
- **F1-score** → Balancing precision and recall  
- **ROC-AUC** → Measuring discrimination ability

---

## 📁 Assets Produced

- ✅ Trained models: Logistic Regression, Decision Tree, Random Forest  
- ✅ Evaluation metrics & ROC curve visualizations  
- ✅ Comparison of model performance by class

---

## 📌 Takeaway

- 🌲 **Random Forest** is the **best performer** in Week 4.
- 📉 **Logistic Regression** offers **maximum recall**, but is not usable alone due to **high false positives**.
- 🌳 **Decision Tree** provides a **balanced baseline**.

---

## 🔄 Next Steps

Use **SMOTE** or other **resampling techniques** to improve the **recall–precision tradeoff** further in Week 5.

