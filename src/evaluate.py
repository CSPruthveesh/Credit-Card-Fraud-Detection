import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)

def calculate_financial_savings(y_true, y_pred, chargeback_cost=100.0, review_cost=2.0):
    """
    Computes net financial savings from using the model.
    - True Positives (Caught Fraud): Operational audit cost = review_cost.
    - False Positives (False Alarms): Operational audit cost = review_cost.
    - False Negatives (Missed Fraud): Cost = chargeback_cost (full loss + penalty).
    - True Negatives (Correct Legitimate): Cost = 0.0.
    """
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    tp = np.sum((y_true == 1) & (y_pred == 1))
    fp = np.sum((y_true == 0) & (y_pred == 1))
    fn = np.sum((y_true == 1) & (y_pred == 0))
    
    # Financial Cost under Model
    model_cost = (fn * chargeback_cost) + ((tp + fp) * review_cost)
    
    # Baseline Cost: Do nothing (All frauds missed, no reviews conducted)
    baseline_cost = np.sum(y_true == 1) * chargeback_cost
    
    savings = baseline_cost - model_cost
    return {
        "model_cost": model_cost,
        "baseline_cost": baseline_cost,
        "net_savings": savings,
        "efficiency_ratio": (savings / baseline_cost) if baseline_cost > 0 else 0
    }

def print_comprehensive_evaluation(y_true, y_proba, threshold=0.5, chargeback_cost=100.0, review_cost=2.0):
    """
    Prints a detailed standard and cost-benefit report for a classifier.
    """
    y_pred = (y_proba >= threshold).astype(int)
    
    # Standard ML Metrics
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Cost-Benefit Analysis
    financials = calculate_financial_savings(y_true, y_pred, chargeback_cost, review_cost)
    
    print("=" * 60)
    print("           CREDIT CARD FRAUD DETECTION PERFORMANCE REPORT")
    print("=" * 60)
    print(f"ROC-AUC:                  {roc_auc:.4f}")
    print(f"Average Precision (AP):   {pr_auc:.4f}  <-- Primary Optimization Metric")
    print(f"Precision (at t={threshold}):      {prec:.4f}")
    print(f"Recall (at t={threshold}):         {rec:.4f}")
    print(f"F1-Score (at t={threshold}):       {f1:.4f}")
    print("-" * 60)
    print("Confusion Matrix:")
    print(f"  True Legitimate (TN):   {cm[0, 0]} | False Alarms (FP): {cm[0, 1]}")
    print(f"  Missed Fraud (FN):      {cm[1, 0]} | Caught Fraud (TP): {cm[1, 1]}")
    print("-" * 60)
    print("Financial Cost-Benefit Analysis:")
    print(f"  Assume Chargeback Cost: ${chargeback_cost:.2f} per transaction")
    print(f"  Assume Review Cost:     ${review_cost:.2f} per alert")
    print(f"  Total Cost (No Model):  ${financials['baseline_cost']:.2f}")
    print(f"  Total Cost (With Model):${financials['model_cost']:.2f}")
    print(f"  NET SAVINGS:            ${financials['net_savings']:.2f} (Savings of {financials['efficiency_ratio']*100:.1f}%)")
    print("=" * 60)

if __name__ == "__main__":
    # Test evaluation output with dummy values
    y_test_dummy = np.array([0]*998 + [1]*2)
    y_proba_dummy = np.array([0.01]*998 + [0.95, 0.2])
    print_comprehensive_evaluation(y_test_dummy, y_proba_dummy)
