# Credit Card Fraud Detection - Project Progress & Weekly Logs

This document compiles the original weekly progression reports, baseline evaluations, and learning logs of the project.

---

# 📅 WEEK1

# ✅ Week 1: Project Initialization & Data Understanding
----
## 🔍 Objective
     -To set up the project environment, load and explore the dataset, and define the core problem statement.
----
## 📁 Setup & Environment
     -Created a structured project directory with folders for each week
     -Installed required Python packages: pandas, numpy, matplotlib, seaborn
----
## 📦 Dataset Used
Source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Description:

     -284,807 transactions
     -492 fraud cases (≈0.17%)
     -Features: 30 columns including:
     -Time, Amount (raw)
     -V1 to V28 (PCA-transformed features)
     -Class: 1 = Fraud, 0 = Not Fraud

---

# 📅 WEEK 2

## Week 2: Exploratory Data Analysis (EDA)

### 🎯 Objective  
To explore the dataset’s structure, understand feature distributions, and identify patterns distinguishing fraud from non-fraud transactions. This helps inform preprocessing and model design in later stages.

---

### 📊 Key Insights

#### 🧮 Class Imbalance Confirmed
- Only ~0.17% of transactions are fraudulent.
- Plotted class distribution using a **log-scaled bar chart** to better visualize the minority class.

#### 🕒 Time Feature
- Shows consistent spread across the dataset.
- **No strong time-based fraud clustering** observed.

#### 💰 Amount Feature
- Highly skewed with extreme outliers.
- Plotted distribution using:
  - Boxplots
  - Log-transformed histograms
  - **Binned range-wise bar plot** (with log y-axis) for clearer fraud/non-fraud comparison

#### 🔗 Correlation Analysis
- Generated **correlation heatmap** to assess inter-feature relationships.

---

# 📅 WEEK3

## Week 3: Data Preprocessing & Feature Scaling

### 🎯 Objective  
Prepare the dataset for model training by scaling features and splitting data into train/test sets while preserving the original class imbalance.

---

### 🔧 Preprocessing Steps Completed

#### 📥 Loaded Clean Dataset
- Re-loaded the original `creditcard.csv` dataset to begin with a fresh, unaltered copy.

#### 🔍 Identified Raw Features for Scaling
- `Amount`: Raw transaction value, highly skewed
- `Time`: Elapsed seconds since the first transaction

#### 📏 Applied Standard Scaling
- Used `StandardScaler` from scikit-learn to scale `Amount` and `Time`
- Created two new columns: `Amount_scaled`, `Time_scaled`
- Dropped the original `Amount` and `Time` columns

#### ✂️ Train-Test Split (Stratified)
- Used `train_test_split()` with `stratify=y` to maintain class imbalance ratio
- Applied an 80/20 split
- Verified that the fraud ratio (~0.17%) was preserved in both training and test sets

---

# 📅 WEEK 4

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


## 📌 Takeaway

- 🌲 **Random Forest** is the **best performer** in Week 4.
- 📉 **Logistic Regression** offers **maximum recall**, but is not usable alone due to **high false positives**.
- 🌳 **Decision Tree** provides a **balanced baseline**.

---

## 🔄 Next Steps

Use **SMOTE** or other **resampling techniques** to improve the **recall–precision tradeoff** further in Week 5.

---

## 📝 Logistic Regression

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

---

## 📝 Decision Tree

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

---

## 📝 Random Forest

# 🌲 Random Forest – Evaluation Summary

---

## ✅ Positive Points

- **Exceptional Accuracy (99.95%)**  
  Among the highest of all models — model correctly classifies nearly all transactions.

- **High Precision for Fraud (0.9610)**  
  Over 96% of predicted frauds were actual frauds — a huge improvement over both Logistic Regression and Decision Tree.

- **Strong F1-score for Fraud (0.8457)**  
  Indicates a good balance between precision and recall — best F1-score so far.

- **Low False Positives (3 total)**  
  Almost no innocent transactions were misclassified as fraud. Great for reducing business/customer impact.

- **High ROC-AUC Score (0.9581)**  
  Indicates excellent overall discriminatory power — almost as good as logistic regression, but with far better precision.

- **Decent Recall for Fraud (0.7551)**  
  Catches 75.5% of actual frauds, slightly more than Decision Tree, though a bit lower than Logistic Regression.

---

## 🧮 Confusion Matrix

|                         | Predicted Non-Fraud (0) | Predicted Fraud (1) |
|-------------------------|--------------------------|----------------------|
| **Actual Non-Fraud (0)** | 56,861                   | 3                    |
| **Actual Fraud (1)**     | 24                       | 74                   |

---

## ❌ Limitations

- **Slightly Lower Recall than Logistic Regression (0.9184)**  
  Random Forest misses more frauds (24) than Logistic Regression (8), but trades this for much better precision.

- **Complexity & Interpretability**  
  Harder to interpret and debug compared to Logistic Regression or a Decision Tree.  
  → Not ideal for quick deployment in explainable-AI contexts unless paired with tools like **SHAP**.

---

## 🧠 Interpretation

Random Forest offers the **best overall balance**:

- ✅ Very high **precision** → Very few false positives  
- ✅ Good **recall** → Catches most frauds  
- ✅ Strong **F1-score** and **ROC-AUC** → Robust across all key metrics

---

---

## 📝 Overall Comparison

# 📊 Summary Comparison – Top Metrics for Fraud Class

| **Metric**     | **Logistic Regression** | **Decision Tree** | **Random Forest** |
|----------------|--------------------------|-------------------|-------------------|
| **Precision**  | 0.0609                   | 0.7071            | 0.9610 ✅         |
| **Recall**     | 0.9184 ✅                | 0.7143            | 0.7551            |
| **F1-score**   | 0.1141                   | 0.7107            | 0.8457 ✅         |
| **ROC-AUC**    | 0.9722 ✅                | 0.8569            | 0.9581            |
| **Accuracy**   | 97.55%                   | 99.90%            | 99.95% ✅         |

---

## 📌 Verdict

- ✅ **Best overall model so far:** **Random Forest**
- ✅ **Recommended as your main benchmark**
- 🧪 **Consider**: Using **SMOTE + Random Forest** to boost **recall** further while maintaining high precision

---

# 📅 WEEK 5

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

---

## 📝 Logistic Regression SMOTE

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

---

## 📝 Decision Tree SMOTE

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

---

## 📝 Random Forest SMOTE

# 🌲 Random Forest + SMOTE – Evaluation Summary

---

## ✅ Positive Points

- **Massively Improved Precision (0.8144)** ✅  
  Slightly lower than baseline RF (0.9610), but still very strong — 81% of predicted frauds are correct.

- **Strong Recall (0.8061)** ✅  
  Caught 80.6% of frauds, up from 75.5% in the baseline.  
  → Much better balance between catching frauds and minimizing false negatives.

- **Best F1-score (0.8103)** ✅  
  Highest F1-score across all models so far — excellent harmony between precision and recall.

- **Low False Positives (18)** ✅  
  Only 18 normal transactions flagged incorrectly — much lower than:
  - Decision Tree + SMOTE: 148  
  - Logistic Regression + SMOTE: 1458

- **Strong ROC-AUC Score (0.9688)** ✅  
  On par with baseline RF (0.9581) and Logistic Regression (0.9722), indicating excellent discrimination power.

- **Excellent Accuracy (99.94%)**  
  While not prioritized, still a solid indicator of model consistency.

---

## 🧮 Confusion Matrix

|                         | Predicted Non-Fraud (0) | Predicted Fraud (1) |
|-------------------------|--------------------------|----------------------|
| **Actual Non-Fraud (0)** | 56,846                   | 18                   |
| **Actual Fraud (1)**     | 19                       | 79                   |

---

## ❌ Minor Trade-offs

- **Slight Drop in Precision vs Baseline RF**  
  - From 0.9610 → 0.8144 🔻  
  - Acceptable tradeoff given better recall and stronger F1.

- **Slightly Higher False Negatives (19 vs 24 in baseline)**  
  Still catching ~81% of all frauds — a very good result.

---

## 📉 Comparison to Week 4 (Baseline RF)

| **Metric**        | **Baseline RF** | **RF + SMOTE** | **Change**         |
|-------------------|------------------|----------------|--------------------|
| **Precision (1)** | 0.9610           | 0.8144         | 🔻 Slightly dropped |
| **Recall (1)**    | 0.7551           | 0.8061         | ✅ Improved         |
| **F1-score (1)**  | 0.8457           | 0.8103         | 🔻 Slight drop      |
| **ROC-AUC**       | 0.9581           | 0.9688         | ✅ Improved         |
| **Accuracy**      | 99.95%           | 99.94%         | ➖ No change        |

---

## 🧠 Interpretation

Random Forest with SMOTE achieved the **best balance** of all models tested:

- ✅ **High precision** → Few false alarms  
- ✅ **High recall** → Caught most frauds  
- ✅ **Strong F1-score** → Balanced performance  
- ✅ **High ROC-AUC** → Very good class discrimination  
- ✅ **Minimal trade-offs** from using SMOTE

---

## ✅ When to Use

This model is ideal if:

- ✅ You want **high recall** without sacrificing **precision**  
- ✅ You need a **reliable, production-ready fraud detector**  
- ✅ You value **consistency across all key metrics**

---

