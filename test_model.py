"""
FRAUD DETECTION MODEL TESTING SCRIPT
=====================================
This script performs final testing on the unseen test set.
Use this ONLY ONCE after model selection and validation.

Author: [Your Name]
Date: 2025-12-02
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, 
                             auc, precision_recall_curve, average_precision_score,
                             f1_score, precision_score, recall_score, accuracy_score)
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FRAUD DETECTION MODEL - FINAL TESTING")
print("="*80)
print("\nWARNING: This script should be run ONLY ONCE on the final test set!")
print("Make sure you have selected your best model from validation results.")

# ============================================================================
# CONFIGURATION
# ============================================================================
# Specify which model to use for final testing
BEST_MODEL_FILE = 'model_xgboost.pkl'  # Change this based on validation results
BEST_MODEL_NAME = 'XGBoost'  # Change this to match the model name

print(f"\nConfiguration:")
print(f"  Model to test: {BEST_MODEL_NAME}")
print(f"  Model file: {BEST_MODEL_FILE}")

# ============================================================================
# LOAD TEST DATA
# ============================================================================
print("\n" + "-"*80)
print("LOADING TEST DATA")
print("-"*80)

test_df = pd.read_csv('fraud_test_encoded.csv')

X_test = test_df.drop('fraud', axis=1)
y_test = test_df['fraud']

print(f"Test data shape: {test_df.shape}")
print(f"Test samples: {len(X_test):,}")
print(f"Fraud cases: {y_test.sum():,} ({y_test.sum()/len(y_test)*100:.2f}%)")
print(f"Non-fraud cases: {(y_test == 0).sum():,} ({(y_test == 0).sum()/len(y_test)*100:.2f}%)")

# ============================================================================
# LOAD BEST MODEL
# ============================================================================
print("\n" + "-"*80)
print("LOADING BEST MODEL")
print("-"*80)

with open(BEST_MODEL_FILE, 'rb') as f:
    model = pickle.load(f)
print(f"[LOADED] {BEST_MODEL_NAME} from {BEST_MODEL_FILE}")

# ============================================================================
# MAKE PREDICTIONS
# ============================================================================
print("\n" + "-"*80)
print("MAKING PREDICTIONS ON TEST SET")
print("-"*80)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("Predictions complete.")

# ============================================================================
# CALCULATE METRICS
# ============================================================================
print("\n" + "="*80)
print("TEST SET RESULTS")
print("="*80)

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()

print(f"\nConfusion Matrix:")
print(f"  True Negatives:  {tn:,} (Correctly identified non-fraud)")
print(f"  False Positives: {fp:,} (Non-fraud incorrectly flagged as fraud)")
print(f"  False Negatives: {fn:,} (Fraud missed by the model)")
print(f"  True Positives:  {tp:,} (Correctly identified fraud)")

# Calculate all metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
specificity = tn / (tn + fp) if (tn + fp) > 0 else 0

# ROC-AUC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

# Precision-Recall AUC
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

print(f"\nPerformance Metrics:")
print(f"  Accuracy:    {accuracy:.4f} ({accuracy*100:.2f}%)")
print(f"  Precision:   {precision:.4f} ({precision*100:.2f}%)")
print(f"  Recall:      {recall:.4f} ({recall*100:.2f}%)")
print(f"  F1-Score:    {f1:.4f}")
print(f"  Specificity: {specificity:.4f} ({specificity*100:.2f}%)")
print(f"  ROC-AUC:     {roc_auc:.4f}")
print(f"  PR-AUC:      {pr_auc:.4f}")

# Business Metrics
fraud_detection_rate = recall * 100
false_alarm_rate = (fp / (tn + fp)) * 100 if (tn + fp) > 0 else 0

print(f"\nBusiness Metrics:")
print(f"  Fraud Detection Rate: {fraud_detection_rate:.2f}% (of all frauds)")
print(f"  False Alarm Rate:     {false_alarm_rate:.2f}% (of all non-frauds)")
print(f"  Frauds Caught:        {tp:,} out of {tp+fn:,}")
print(f"  Frauds Missed:        {fn:,}")

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "-"*80)
print("CREATING VISUALIZATIONS")
print("-"*80)

# Confusion Matrix Heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['Non-Fraud', 'Fraud'],
            yticklabels=['Non-Fraud', 'Fraud'],
            annot_kws={'size': 14})
plt.title(f'Confusion Matrix - {BEST_MODEL_NAME} (Test Set)', fontsize=16)
plt.ylabel('Actual', fontsize=14)
plt.xlabel('Predicted', fontsize=14)
plt.tight_layout()
plt.savefig('test_confusion_matrix.png', dpi=300)
plt.close()
print("[SAVED] test_confusion_matrix.png")

# ROC Curve
plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title(f'ROC Curve - {BEST_MODEL_NAME} (Test Set)', fontsize=14)
plt.legend(loc='lower right', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('test_roc_curve.png', dpi=300)
plt.close()
print("[SAVED] test_roc_curve.png")

# Precision-Recall Curve
plt.figure(figsize=(10, 8))
plt.plot(recall_curve, precision_curve, color='blue', lw=2, 
         label=f'PR curve (AP = {pr_auc:.3f})')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title(f'Precision-Recall Curve - {BEST_MODEL_NAME} (Test Set)', fontsize=14)
plt.legend(loc='best', fontsize=12)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('test_precision_recall_curve.png', dpi=300)
plt.close()
print("[SAVED] test_precision_recall_curve.png")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n" + "-"*80)
print("SAVING RESULTS")
print("-"*80)

# Save test results
test_results = {
    'model_name': BEST_MODEL_NAME,
    'model_file': BEST_MODEL_FILE,
    'test_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'test_samples': int(len(X_test)),
    'fraud_cases': int(y_test.sum()),
    'non_fraud_cases': int((y_test == 0).sum()),
    'confusion_matrix': {
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    },
    'metrics': {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'specificity': float(specificity),
        'roc_auc': float(roc_auc),
        'pr_auc': float(pr_auc)
    },
    'business_metrics': {
        'fraud_detection_rate': float(fraud_detection_rate),
        'false_alarm_rate': float(false_alarm_rate),
        'frauds_caught': int(tp),
        'frauds_missed': int(fn)
    }
}

with open('test_results.json', 'w') as f:
    json.dump(test_results, f, indent=4)
print("[SAVED] test_results.json")

# Save classification report
report = classification_report(y_test, y_pred, target_names=['Non-Fraud', 'Fraud'])
with open('test_classification_report.txt', 'w') as f:
    f.write("FRAUD DETECTION - FINAL TEST RESULTS\n")
    f.write("="*60 + "\n\n")
    f.write(f"Model: {BEST_MODEL_NAME}\n")
    f.write(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
    f.write("CLASSIFICATION REPORT\n")
    f.write("-"*60 + "\n")
    f.write(report)
    f.write("\n\nCONFUSION MATRIX\n")
    f.write("-"*60 + "\n")
    f.write(f"True Negatives:  {tn:,}\n")
    f.write(f"False Positives: {fp:,}\n")
    f.write(f"False Negatives: {fn:,}\n")
    f.write(f"True Positives:  {tp:,}\n")

print("[SAVED] test_classification_report.txt")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("TESTING COMPLETE")
print("="*80)

print(f"\nFinal Model Performance on Test Set:")
print(f"  Model: {BEST_MODEL_NAME}")
print(f"  F1-Score: {f1:.4f}")
print(f"  ROC-AUC: {roc_auc:.4f}")
print(f"  Fraud Detection Rate: {fraud_detection_rate:.2f}%")
print(f"  False Alarm Rate: {false_alarm_rate:.2f}%")

print("\nFiles saved:")
print("  - test_results.json")
print("  - test_classification_report.txt")
print("  - test_confusion_matrix.png")
print("  - test_roc_curve.png")
print("  - test_precision_recall_curve.png")

print("\n" + "="*80)
print("PROJECT COMPLETE!")
print("="*80)
