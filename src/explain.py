import shap
import pandas as pd
import numpy as np

def get_shap_explanation(model, X_train, X_test_row, top_n=3):
    """
    Computes SHAP feature contributions for a single transaction.
    - model: Trained tree-based model (e.g. LightGBM, XGBoost, CatBoost).
    - X_train: Training features dataframe (used to build background distribution if needed).
    - X_test_row: A single row DataFrame containing the transaction to explain.
    """
    # Initialize Explainer (TreeExplainer is optimized for GBDTs)
    try:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_test_row)
    except Exception:
        # Fallback to general Explainer
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test_row).values

    # Handle output structure differences across models (binary/multiclass/LightGBM formats)
    if isinstance(shap_values, list):
        # Extract class 1 (fraud) contributions
        shap_vals = shap_values[1][0]
    elif len(shap_values.shape) == 3:
        shap_vals = shap_values[0, :, 1]
    elif len(shap_values.shape) == 2:
        shap_vals = shap_values[0]
    else:
        shap_vals = shap_values

    # Get feature names and values
    feature_names = X_test_row.columns
    feature_values = X_test_row.iloc[0].values

    # Sort feature contributions by absolute impact
    sorted_indices = np.argsort(np.abs(shap_vals))[::-1]
    
    explanation = []
    for idx in sorted_indices[:top_n]:
        feat = feature_names[idx]
        val = feature_values[idx]
        impact = shap_vals[idx]
        
        explanation.append({
            "feature": feat,
            "value": float(val),
            "shap_value": float(impact),
            "direction": "increases risk" if impact > 0 else "decreases risk"
        })
        
    return explanation

def print_explanation(explanation):
    print("=" * 60)
    print("           LOCAL TRANSACTION RISK AUDIT REPORT (SHAP)")
    print("=" * 60)
    print(f"Top {len(explanation)} Risk Drivers:")
    for idx, driver in enumerate(explanation, 1):
        print(f"  {idx}. Feature: '{driver['feature']}' (Value: {driver['value']:.4f})")
        print(f"     Contribution Score: {driver['shap_value']:+.4f} ({driver['direction']})")
    print("=" * 60)
