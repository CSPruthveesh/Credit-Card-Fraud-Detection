# 🔍 Logistic Regression – Evaluation Summary

---

## ✅ Positive Points

- **Very High ROC-AUC (0.9722)**  
  Shows excellent separation between fraud and non-fraud classes.

- **Excellent Recall for Fraud Class (0.9184)**  
  Model successfully detected 91.8% of frauds — critical for fraud detection tasks.

- **Very High Accuracy (97.55%)**  
  Despite class imbalance, overall accuracy remains strong.

- **Strong Non-Fraud Detection (Class 0)**  
  - Precision: **99.99%**  
  - F1-score: **98.76%**

---

## 🧮 Confusion Matrix

|               | Predicted Non-Fraud (0) | Predicted Fraud (1) |
|---------------|--------------------------|----------------------|
| **Actual Non-Fraud (0)** | 55,475                   | 1,389                |
| **Actual Fraud (1)**     | 8                        | 90                   |

---

## ❌ Negative Points / Limitations

- **Very Low Precision for Fraud (0.0609)**  
  Only 6% of predicted frauds were actually fraud — many false positives.  
  → Can lead to unnecessary transaction blocks or alerts.

- **Low F1-score for Fraud Class (0.1141)**  
  Indicates imbalance between precision and recall.  
  → Model prioritizes recall at the cost of precision, which may not be acceptable in production.

- **False Positives are High (1,389)**  
  Many non-fraudulent transactions are flagged incorrectly.

- **Model Bias Toward Class 0 (Non-Fraud)**  
  High accuracy partly driven by the large number of true negatives.

---

## 🧠 Interpretation

The model is doing what we often want in fraud detection: catching as many frauds as possible (**high recall**).  
But it does this at the cost of flagging too many normal transactions as fraud (**low precision**).

---

## 📌 When is this acceptable?

- In a **risk-averse setup** where **missing frauds is unacceptable**.
- When there's a **manual review step** after automatic flagging.
- As a **starting baseline**, it's solid — but **needs improvement**, especially in **precision**.

