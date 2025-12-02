"""
FRAUD DETECTION MODEL VALIDATION SCRIPT
========================================
This script performs detailed validation analysis on trained models.
Includes confusion matrix, ROC curves, precision-recall curves, and threshold analysis.

Author: [Your Name]
Date: 2025-12-02
"""

import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (confusion_matrix, classification_report, roc_curve, 
                             auc, precision_recall_curve, average_precision_score)
import json
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FRAUD DETECTION MODEL VALIDATION")
print("="*80)

# ============================================================================
# LOAD VALIDATION DATA
# ============================================================================
print("\nLoading validation data...")
val_df = pd.read_csv('fraud_validation_encoded.csv')

X_val = val_df.drop('fraud', axis=1)
y_val = val_df['fraud']

print(f"Validation data shape: {val_df.shape}")
print(f"Fraud cases: {y_val.sum()} ({y_val.sum()/len(y_val)*100:.2f}%)")

# ============================================================================
# LOAD TRAINED MODELS
# ============================================================================
print("\n" + "-"*80)
print("LOADING TRAINED MODELS")
print("-"*80)

models = {}
model_files = [
    ('Logistic Regression', 'model_logistic_regression.pkl'),
    ('Random Forest', 'model_random_forest.pkl'),
    ('XGBoost', 'model_xgboost.pkl')
]

for model_name, model_file in model_files:
    try:
        with open(model_file, 'rb') as f:
            models[model_name] = pickle.load(f)
        print(f"[LOADED] {model_name}")
    except FileNotFoundError:
        print(f"[SKIP] {model_name} - file not found")

# ============================================================================
# DETAILED VALIDATION FOR EACH MODEL
# ============================================================================
validation_results = []

for model_name, model in models.items():
    print("\n" + "="*80)
    print(f"VALIDATING: {model_name}")
    print("="*80)
    
    # Predictions
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    
    # Confusion Matrix
    cm = confusion_matrix(y_val, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn:,}")
    print(f"  False Positives: {fp:,}")
    print(f"  False Negatives: {fn:,}")
    print(f"  True Positives:  {tp:,}")
    
    # Calculate metrics
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # ROC-AUC
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Precision-Recall AUC
    precision_curve, recall_curve, _ = precision_recall_curve(y_val, y_proba)
    pr_auc = average_precision_score(y_val, y_proba)
    
    print(f"\nMetrics:")
    print(f"  Accuracy:    {accuracy:.4f}")
    print(f"  Precision:   {precision:.4f}")
    print(f"  Recall:      {recall:.4f}")
    print(f"  F1-Score:    {f1:.4f}")
    print(f"  Specificity: {specificity:.4f}")
    print(f"  ROC-AUC:     {roc_auc:.4f}")
    print(f"  PR-AUC:      {pr_auc:.4f}")
    
    # Store results
    validation_results.append({
        'model': model_name,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'specificity': specificity,
        'roc_auc': roc_auc,
        'pr_auc': pr_auc,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    })
    
    # Plot Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Non-Fraud', 'Fraud'],
                yticklabels=['Non-Fraud', 'Fraud'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{model_name.lower().replace(" ", "_")}.png', dpi=300)
    plt.close()
    print(f"[SAVED] confusion_matrix_{model_name.lower().replace(' ', '_')}.png")

# ============================================================================
# COMPARISON PLOTS
# ============================================================================
print("\n" + "="*80)
print("CREATING COMPARISON VISUALIZATIONS")
print("="*80)

# ROC Curves Comparison
plt.figure(figsize=(10, 8))
for model_name, model in models.items():
    y_proba = model.predict_proba(X_val)[:, 1]
    fpr, tpr, _ = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.3f})', linewidth=2)

plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=1)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curves - Model Comparison', fontsize=14)
plt.legend(loc='lower right', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('roc_curves_comparison.png', dpi=300)
plt.close()
print("[SAVED] roc_curves_comparison.png")

# Precision-Recall Curves Comparison
plt.figure(figsize=(10, 8))
for model_name, model in models.items():
    y_proba = model.predict_proba(X_val)[:, 1]
    precision, recall, _ = precision_recall_curve(y_val, y_proba)
    pr_auc = average_precision_score(y_val, y_proba)
    plt.plot(recall, precision, label=f'{model_name} (AP = {pr_auc:.3f})', linewidth=2)

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall', fontsize=12)
plt.ylabel('Precision', fontsize=12)
plt.title('Precision-Recall Curves - Model Comparison', fontsize=14)
plt.legend(loc='best', fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('precision_recall_curves_comparison.png', dpi=300)
plt.close()
print("[SAVED] precision_recall_curves_comparison.png")

# Metrics Comparison Bar Chart
results_df = pd.DataFrame(validation_results)
metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, metric in enumerate(metrics_to_plot):
    ax = axes[idx]
    results_df.plot(x='model', y=metric, kind='bar', ax=ax, legend=False, color='steelblue')
    ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_xlabel('')
    ax.set_ylabel('Score', fontsize=10)
    ax.set_ylim([0, 1])
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

# Remove extra subplot
fig.delaxes(axes[5])
plt.tight_layout()
plt.savefig('metrics_comparison.png', dpi=300)
plt.close()
print("[SAVED] metrics_comparison.png")

# ============================================================================
# SAVE VALIDATION RESULTS
# ============================================================================
results_df.to_csv('validation_results.csv', index=False)
print("\n[SAVED] validation_results.csv")

# Save detailed report
with open('validation_report.json', 'w') as f:
    json.dump(validation_results, f, indent=4)
print("[SAVED] validation_report.json")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("VALIDATION SUMMARY")
print("="*80)
print("\n" + results_df.to_string(index=False))

best_f1_idx = results_df['f1_score'].idxmax()
best_model = results_df.loc[best_f1_idx, 'model']
best_f1 = results_df.loc[best_f1_idx, 'f1_score']

print(f"\nBest Model (by F1-Score): {best_model} (F1 = {best_f1:.4f})")

print("\n" + "="*80)
print("VALIDATION COMPLETE")
print("="*80)
print("\nNext step: Run test_model.py for final testing on unseen test set")
print("="*80)
