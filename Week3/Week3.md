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

