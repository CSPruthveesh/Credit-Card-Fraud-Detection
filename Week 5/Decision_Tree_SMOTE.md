# 🌳 Decision Tree + SMOTE – Evaluation Summary

---

## ✅ Positive Points

- **Improved Recall (0.7755)** ✅  
  Caught 77.6% of all frauds, compared to 71.4% in the baseline.  
  → SMOTE helped the tree become more sensitive to the minority class.

- **Higher ROC-AUC (0.8865)** ✅  
  A +3% gain over baseline (0.8569) — shows better overall separation between fraud and non-fraud.

- **Good Accuracy (99.70%)**  
  Still very high, though less relevant in imbalanced datasets.

- **Low False Negatives (22)**  
  Only 22 frauds missed (compared to 28 in baseline).

---

## 🧮 Confusion Matrix

|                         | Predicted Non-Fraud (0) | Predicted Fraud (1) |
|-------------------------|--------------------------|----------------------|
| **Actual Non-Fraud (0)** | 56,716                   | 148                  |
| **Actual Fraud (1)**     | 22                       | 76                   |

---

## ❌ Negative Points / Limitations

- **Precision Dropped Sharply (0.3393)** ❌  
  - Before: 70.71% → Now: 33.93%  
  - Meaning: More than half of predicted frauds were false alarms.

- **Lower F1-score (0.4720)** ❌  
  - F1 dropped from 0.7107 → 0.4720  
  - Driven by the imbalance between high recall and poor precision.

- **Higher False Positives (148 vs 29)** ❌  
  - More non-fraud cases wrongly classified as fraud — not ideal for deployment without a human review layer.

---

## 📉 Comparison to Week 4 (Baseline)

| **Metric**         | **Baseline Tree** | **SMOTE Tree** | **Change**         |
|--------------------|-------------------|----------------|--------------------|
| **Precision (1)**  | 0.7071            | 0.3393         | 🔻 Dropped sharply |
| **Recall (1)**     | 0.7143            | 0.7755         | ✅ Improved        |
| **F1-score (1)**   | 0.7107            | 0.4720         | 🔻 Dropped         |
| **ROC-AUC**        | 0.8569            | 0.8865         | ✅ Improved        |
| **False Positives**| 29                | 148            | 🔺 Worse           |

---

## 🧠 Interpretation

SMOTE made the Decision Tree more aggressive in detecting fraud (**↑ recall**), but at the cost of **more false positives** (**↓ precision**).  
This reduced its **F1-score** and may hurt practical deployment.

Still a **better choice than Logistic Regression + SMOTE**, which had **horrible precision**.

Performance could be improved with:

- Tree pruning  
- `max_depth` tuning  
- Switching to ensemble methods

---

## ✅ When to Use

Use this model if:

- ✅ You want **better recall** than baseline Decision Tree  
- ✅ You're okay **reviewing false positives manually**

🚫 **Avoid in production** if **precision is mission-critical**.
