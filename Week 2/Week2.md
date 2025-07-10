# Week 2: Exploratory Data Analysis (EDA)
----
## 🎯 Objective
    -To explore the dataset’s structure, understand feature distributions, and identify patterns distinguishing fraud from non-fraud transactions. This helps inform preprocessing and model design in later stages.

## 📊 Key Insights
    -**Class Imbalance Confirmed:**
        Only ~0.17% of transactions are fraudulent.
        Plotted class distribution using a log-scaled bar chart to better visualize the minority class.

Time Feature:

Shows consistent spread across the dataset.

No strong time-based fraud clustering observed.

Amount Feature:

Highly skewed with extreme outliers.

Plotted distribution using:

Boxplots

Log-transformed histograms

Binned range-wise bar plot (with log y-axis) for clearer comparison across fraud/non-fraud

Correlation Analysis:

Generated heatmap to assess inter-feature relationships.

Top positively correlated features with fraud: V14, V10, V17

Features like V20, V7, V4 had lower or inverse correlation

Feature Distribution by Class:

Compared selected features across fraud and non-fraud using histograms.

Notable separation seen in features such as V14, V10, and Amount.

📁 Assets Produced
notebooks/eda_week2.ipynb: EDA notebook with annotated plots

reports/: contains graphs and summary visuals for future use

Feature shortlist to focus on in modeling phase

📌 Takeaway
EDA revealed class imbalance, key feature separations, and data skewness in amount values. These findings will influence scaling decisions, class balancing (SMOTE), and feature selection in Weeks 3–5.
