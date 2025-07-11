# Credit-Card-Fraud-Detection

This project focuses on detecting fraudulent transactions using machine learning techniques on an imbalanced dataset. The goal is to build models that can **identify frauds with high recall** while minimizing false alarms. The project follows a **structured 8-week roadmap** covering data exploration, preprocessing, model building, evaluation, and hyperparameter tuning.

---

## 📊 Dataset

* Source: [Kaggle - Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
* Highly imbalanced (frauds ≈ 0.17% of all transactions)
* Features: 30 numerical columns (PCA-transformed), including `Amount`, `Time`, and `Class` (target)

---

## ✅ Key Highlights

* 📈 **Exploratory Data Analysis**: Class distribution, amount/time trends, log scaling for better visibility
* ⚖️ **SMOTE**: Applied to synthetically balance the dataset
* 🤖 **Models Trained**:

  * Logistic Regression
  * Decision Tree
  * Random Forest
* 📉 **Performance Evaluation**: Confusion matrix, ROC-AUC, precision, recall, F1-score
* 🔄 **Model Comparison** before and after SMOTE & tuning

---

## 🚀 Objective

> To create interpretable and accurate ML models for real-time fraud detection systems with a **focus on recall**, ensuring minimal false negatives in critical financial environments.

---


