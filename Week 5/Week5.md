# 📘 Week 5: SMOTE-Based Resampling and Model Retraining

---

## 🎯 Objective

To handle **extreme class imbalance** in the dataset using **SMOTE (Synthetic Minority Oversampling Technique)** and retrain models to evaluate if **recall–precision–F1 trade-offs** improve compared to the **baseline (Week 4)**.

---

## 🧪 Methodology

- Applied **SMOTE only on training data**
- Retained original test set for **unbiased evaluation**
- Trained three classifiers:
  - ✅ Logistic Regression
  - ✅ Decision Tree
  - ✅ Random Forest
- Compared performance with **baseline imbalanced models**

---

## 📊 Model Performance After SMOTE

### 🔹 Logistic Regression + SMOTE

| **Metric**       | **Value** | **Observation**                         |
|------------------|-----------|------------------------------------------|
| **Precision (1)**| 0.0581    | ❌ Very low — high false alarms           |
| **Recall (1)**   | 0.9184    | ✅ Excellent — caught nearly all frauds   |
| **F1-score (1)** | 0.1094    | ❌ Poor balance                          |
| **ROC-AUC**      | 0.9698    | ✅ Very good                             |

📌 **Conclusion**: 🔺 Recall-focused but not practical due to **high false positives**

---

### 🔹 Decision Tree + SMOTE

| **Metric**       | **Value** | **Observation**                          |
|------------------|-----------|-------------------------------------------|
| **Precision (1)**| 0.3393    | ⚠️ Much lower than baseline               |
| **Recall (1)**   | 0.7755    | ✅ Improved                               |
| **F1-score (1)** | 0.4720    | ⚠️ Decreased due to precision drop        |
| **ROC-AUC**      | 0.8865    | ✅ Improved                               |

📌 **Conclusion**: Balanced recall, but **huge spike in false positives** hurts usability

---

### 🔹 Random Forest + SMOTE

| **Metric**       | **Value** | **Observation**                          |
|------------------|-----------|-------------------------------------------|
| **Precision (1)**| 0.8144    | ✅ High — most predicted frauds were correct |
| **Recall (1)**   | 0.8061    | ✅ High — caught 80% of frauds             |
| **F1-score (1)** | 0.8103    | ✅ Best balance overall                    |
| **ROC-AUC**      | 0.9688    | ✅ Excellent                               |

📌 **Conclusion**: ⭐ **Best performing model** — strong in all metrics with minimal trade-offs

---

## 📌 Key Takeaways

- ✅ **SMOTE** helped **increase recall** in all models.
- ❌ **Logistic Regression** remained too **imprecise** to be useful.
- ⚠️ **Decision Tree** became **too aggressive**, leading to **many false alarms**.
- 🌲 **Random Forest with SMOTE** struck the **best balance**:
  - ⚖️ High **recall** and **precision**
  - 💡 Strong **F1-score** and **ROC-AUC**
  - ✅ **Recommended as the new baseline** going forward

