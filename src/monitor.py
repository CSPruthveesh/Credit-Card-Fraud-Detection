import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def check_feature_drift(reference_df, current_df, alpha=0.05, threshold_fraction=0.2):
    """
    Monitors data drift between reference (training) and current (live inference) data
    using the Kolmogorov-Smirnov (KS) two-sample test for numerical columns.
    - reference_df: Historical dataset used to train the model.
    - current_df: Live incoming transaction logs.
    - alpha: Significance level for p-value threshold (default 5%).
    - threshold_fraction: Fraction of features that must drift to trigger retraining alert.
    """
    drift_report = {}
    drift_count = 0
    features = [col for col in reference_df.columns if col != 'Class']
    
    print("=" * 60)
    print("           REAL-TIME TRANSACTION DATA DRIFT AUDIT REPORT")
    print("=" * 60)
    print(f"Analyzing {len(features)} features for covariate shift...")
    
    for col in features:
        # Run KS test
        stat, p_val = ks_2samp(reference_df[col], current_df[col])
        
        # If p-value is smaller than significance level, we reject the null hypothesis (i.e. drift is present)
        is_drifted = p_val < alpha
        drift_report[col] = {
            "statistic": float(stat),
            "p_value": float(p_val),
            "drift_detected": is_drifted
        }
        
        if is_drifted:
            drift_count += 1
            print(f"  [DRIFT DETECTED] Feature: '{col}' (p-value: {p_val:.4e})")
            
    drift_fraction = drift_count / len(features)
    trigger_retrain = drift_fraction >= threshold_fraction
    
    print("-" * 60)
    print(f"Summary: {drift_count}/{len(features)} features drifted ({drift_fraction * 100:.1f}%).")
    print(f"Drift Threshold Trigger: {threshold_fraction * 100:.1f}%.")
    
    if trigger_retrain:
        print("\n[WARNING] DATA DRIFT TRIGGERED! Retraining pipeline is required.")
        print("Action: Trigger Airflow DAG / Prefect workflow for continuous training.")
    else:
        print("\n[INFO] Data distribution is stable. No retraining required.")
    print("=" * 60)
    
    return {
        "drift_detected": trigger_retrain,
        "drift_fraction": drift_fraction,
        "features_report": drift_report
    }

if __name__ == "__main__":
    # Test execution with mock data (shifting mean of current data to simulate drift)
    ref_dummy = pd.DataFrame(np.random.normal(0, 1, (1000, 5)), columns=[f"V{i}" for i in range(1, 6)])
    curr_dummy = pd.DataFrame(np.random.normal(0.5, 1, (1000, 5)), columns=[f"V{i}" for i in range(1, 6)])
    check_feature_drift(ref_dummy, curr_dummy)
