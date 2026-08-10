import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb
import mlflow
import mlflow.lightgbm
from evaluate import calculate_financial_savings
from sklearn.metrics import average_precision_score, roc_auc_score

def load_and_preprocess():
    """
    Loads transaction data, performs temporal feature engineering,
    splits into train/test partitions, and scales numerical columns.
    """
    local_path = "creditcard.csv"
    if not os.path.exists(local_path):
        raise FileNotFoundError("creditcard.csv not found in the workspace directory. Ensure it is placed here.")
    
    df = pd.read_csv(local_path)
    
    # 1. Feature Engineering
    df = df.sort_values(by="Time").reset_index(drop=True)
    df['Card_Proxy'] = df['V1'].round(1).astype(str) + "_" + df['V2'].round(1).astype(str)
    df['Time_Delta'] = df.groupby('Card_Proxy')['Time'].diff().fillna(-1)
    df['Last_5_Tx_Time_Span'] = df.groupby('Card_Proxy')['Time'].diff(periods=5).fillna(-1)
    df['Rolling_Mean_Amount_5'] = df.groupby('Card_Proxy')['Amount'].transform(lambda x: x.rolling(5, min_periods=1).mean())
    df.drop('Card_Proxy', axis=1, inplace=True)
    
    # 2. Split
    X = df.drop("Class", axis=1)
    y = df["Class"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
    
    # 3. Scaling (preventing data leakage)
    num_features = ['Amount', 'Time', 'Time_Delta', 'Last_5_Tx_Time_Span', 'Rolling_Mean_Amount_5']
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    for col in num_features:
        scaler = StandardScaler()
        X_train[f'{col}_scaled'] = scaler.fit_transform(X_train[[col]])
        X_test[f'{col}_scaled'] = scaler.transform(X_test[[col]])
        X_train.drop(col, axis=1, inplace=True)
        X_test.drop(col, axis=1, inplace=True)
        
    return X_train, X_test, y_train, y_test

def train_and_log():
    # Load and process data
    print("Loading and preprocessing dataset...")
    X_train, X_test, y_train, y_test = load_and_preprocess()
    
    # Set MLflow experiment name
    mlflow.set_experiment("Credit_Card_Fraud_Detection")
    
    print("Starting MLflow run...")
    with mlflow.start_run(run_name="LightGBM_Production_Baseline"):
        # Parameters configuration
        n_estimators = 200
        learning_rate = 0.05
        num_leaves = 31
        neg_count = y_train.value_counts()[0]
        pos_count = y_train.value_counts()[1]
        scale_pos_weight = neg_count / pos_count
        
        mlflow.log_params({
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "num_leaves": num_leaves,
            "scale_pos_weight": scale_pos_weight
        })
        
        # Initialize LightGBM Classifier
        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            verbose=-1
        )
        
        print("Training LightGBM model...")
        model.fit(X_train, y_train)
        
        # Predictions
        y_proba = model.predict_proba(X_test)[:, 1]
        y_pred = (y_proba >= 0.5).astype(int)
        
        # Metrics
        ap = average_precision_score(y_test, y_proba)
        roc_auc = roc_auc_score(y_test, y_proba)
        financials = calculate_financial_savings(y_test, y_pred)
        
        mlflow.log_metrics({
            "Average_Precision": ap,
            "ROC_AUC": roc_auc,
            "Net_Financial_Savings_USD": financials["net_savings"],
            "Operational_Efficiency": financials["efficiency_ratio"]
        })
        
        # Log model
        mlflow.lightgbm.log_model(model, artifact_path="model")
        print("Model and metrics logged to MLflow successfully!")
        print(f"Average Precision achieved: {ap:.4f}")
        print(f"Net Savings calculated: ${financials['net_savings']:.2f}")

if __name__ == "__main__":
    train_and_log()
