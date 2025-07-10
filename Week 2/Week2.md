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





