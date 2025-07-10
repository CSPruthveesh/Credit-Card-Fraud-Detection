# 🌳 Decision Tree – Evaluation Summary

---

## ✅ Positive Points

- **Strong Accuracy (99.90%)**  
  Indicates the model is correctly classifying the majority of transactions.

- **Much Better Precision for Fraud (0.7071)**  
  Over 70% of predicted frauds are actually fraud — significantly better than logistic regression's 6%.

- **Balanced F1-score for Fraud (0.7107)**  
  Indicates a good trade-off between precision and recall for class 1.

- **Low False Positive Rate**  
  Only 29 non-fraud transactions wrongly classified as fraud.

- **Low False Negative Rate**  
  Only 28 fraud cases missed (compared to 8 in logistic regression, but still very few).

---

## 🧮 Confusion Matrix

|                         | Predicted Non-Fraud (0) | Predicted Fraud (1) |
|-------------------------|--------------------------|----------------------|
| **Actual Non-Fraud (0)** | 56,835                   | 29                   |
| **Actual Fraud (1)**     | 28                       | 70                   |

---

## ❌ Limitations / Areas for Improvement

- **Moderate Recall for Fraud (0.7143)**  
  Caught ~71% of frauds; lower than logistic regression's 91%, meaning more frauds were missed.

- **Lower ROC-AUC Score (0.8569)**  
  Indicates weaker overall ability to distinguish between classes compared to logistic regression (which had 0.9722).

- **Risk of Overfitting**  
  Decision trees can memorize patterns and may not generalize well without pruning or depth constraints.

- **Sharp Decisions**  
  Tree-based models can make hard splits, which sometimes fail to capture subtleties in fraud behavior.

---

## 🧠 Interpretation

This decision tree provides a much better balance than logistic regression — high precision, good F1-score, and very few false positives.  
However, it comes at the cost of slightly lower recall and weaker ROC-AUC.

---

## 🏁 When is this good?

- When **false alarms (false positives)** are costly and you want **high precision**.
- In situations where **human review follows up** model flags, reducing unnecessary effort.

