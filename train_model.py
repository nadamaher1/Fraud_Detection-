"""
FRAUD DETECTION MODEL TRAINING SCRIPT
======================================
This script trains multiple machine learning models on the fraud detection dataset.
Models included: Logistic Regression, Random Forest, and XGBoost

Author: [Your Name]
Date: 2025-12-02
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, f1_score, precision_score, recall_score
import xgboost as xgb
import pickle
import json
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FRAUD DETECTION MODEL TRAINING")
print("="*80)

# ============================================================================
# CONFIGURATION
# ============================================================================
# Choose which balanced dataset to use:
# - 'fraud_train_encoded.csv' (original with class weights)
# - 'fraud_train_smote.csv' (SMOTE balanced)
# - 'fraud_train_undersampled.csv' (undersampled)

TRAIN_DATA = 'fraud_train_smote.csv'  # Change this to try different balancing methods
VALIDATION_DATA = 'fraud_validation_encoded.csv'
RANDOM_STATE = 42

print(f"\nConfiguration:")
print(f"  Training data: {TRAIN_DATA}")
print(f"  Validation data: {VALIDATION_DATA}")
print(f"  Random state: {RANDOM_STATE}")

# ============================================================================
# LOAD DATA
# ============================================================================
print("\n" + "-"*80)
print("LOADING DATA")
print("-"*80)

train_df = pd.read_csv(TRAIN_DATA)
val_df = pd.read_csv(VALIDATION_DATA)

print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")

# Separate features and target
X_train = train_df.drop('fraud', axis=1)
y_train = train_df['fraud']
X_val = val_df.drop('fraud', axis=1)
y_val = val_df['fraud']

print(f"\nTrain class distribution:")
print(f"  Non-fraud: {(y_train == 0).sum():,} ({(y_train == 0).sum()/len(y_train)*100:.2f}%)")
print(f"  Fraud: {(y_train == 1).sum():,} ({(y_train == 1).sum()/len(y_train)*100:.2f}%)")

# ============================================================================
# MODEL 1: LOGISTIC REGRESSION
# ============================================================================
print("\n" + "="*80)
print("MODEL 1: LOGISTIC REGRESSION")
print("="*80)

lr_model = LogisticRegression(
    max_iter=1000,
    random_state=RANDOM_STATE,
    class_weight='balanced',  # Use balanced weights
    solver='saga',
    n_jobs=-1
)

print("Training Logistic Regression...")
lr_model.fit(X_train, y_train)

# Predictions
y_val_pred_lr = lr_model.predict(X_val)
y_val_proba_lr = lr_model.predict_proba(X_val)[:, 1]

# Metrics
lr_metrics = {
    'model': 'Logistic Regression',
    'accuracy': lr_model.score(X_val, y_val),
    'precision': precision_score(y_val, y_val_pred_lr),
    'recall': recall_score(y_val, y_val_pred_lr),
    'f1_score': f1_score(y_val, y_val_pred_lr),
    'roc_auc': roc_auc_score(y_val, y_val_proba_lr)
}

print(f"\nValidation Results:")
print(f"  Accuracy: {lr_metrics['accuracy']:.4f}")
print(f"  Precision: {lr_metrics['precision']:.4f}")
print(f"  Recall: {lr_metrics['recall']:.4f}")
print(f"  F1-Score: {lr_metrics['f1_score']:.4f}")
print(f"  ROC-AUC: {lr_metrics['roc_auc']:.4f}")

# Save model
with open('model_logistic_regression.pkl', 'wb') as f:
    pickle.dump(lr_model, f)
print("\n[SAVED] model_logistic_regression.pkl")

# ============================================================================
# MODEL 2: RANDOM FOREST
# ============================================================================
print("\n" + "="*80)
print("MODEL 2: RANDOM FOREST")
print("="*80)

rf_model = RandomForestClassifier(
    n_estimators=100,
    max_depth=20,
    min_samples_split=10,
    min_samples_leaf=4,
    random_state=RANDOM_STATE,
    class_weight='balanced',
    n_jobs=-1,
    verbose=1
)

print("Training Random Forest...")
rf_model.fit(X_train, y_train)

# Predictions
y_val_pred_rf = rf_model.predict(X_val)
y_val_proba_rf = rf_model.predict_proba(X_val)[:, 1]

# Metrics
rf_metrics = {
    'model': 'Random Forest',
    'accuracy': rf_model.score(X_val, y_val),
    'precision': precision_score(y_val, y_val_pred_rf),
    'recall': recall_score(y_val, y_val_pred_rf),
    'f1_score': f1_score(y_val, y_val_pred_rf),
    'roc_auc': roc_auc_score(y_val, y_val_proba_rf)
}

print(f"\nValidation Results:")
print(f"  Accuracy: {rf_metrics['accuracy']:.4f}")
print(f"  Precision: {rf_metrics['precision']:.4f}")
print(f"  Recall: {rf_metrics['recall']:.4f}")
print(f"  F1-Score: {rf_metrics['f1_score']:.4f}")
print(f"  ROC-AUC: {rf_metrics['roc_auc']:.4f}")

# Feature importance
feature_importance = pd.DataFrame({
    'feature': X_train.columns,
    'importance': rf_model.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Most Important Features:")
print(feature_importance.head(10).to_string(index=False))

# Save model and feature importance
with open('model_random_forest.pkl', 'wb') as f:
    pickle.dump(rf_model, f)
feature_importance.to_csv('feature_importance_rf.csv', index=False)
print("\n[SAVED] model_random_forest.pkl")
print("[SAVED] feature_importance_rf.csv")

# ============================================================================
# MODEL 3: XGBOOST
# ============================================================================
print("\n" + "="*80)
print("MODEL 3: XGBOOST")
print("="*80)

# Calculate scale_pos_weight for imbalanced data
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

xgb_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    n_jobs=-1,
    eval_metric='auc'
)

print(f"Training XGBoost (scale_pos_weight={scale_pos_weight:.2f})...")
xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

# Predictions
y_val_pred_xgb = xgb_model.predict(X_val)
y_val_proba_xgb = xgb_model.predict_proba(X_val)[:, 1]

# Metrics
xgb_metrics = {
    'model': 'XGBoost',
    'accuracy': xgb_model.score(X_val, y_val),
    'precision': precision_score(y_val, y_val_pred_xgb),
    'recall': recall_score(y_val, y_val_pred_xgb),
    'f1_score': f1_score(y_val, y_val_pred_xgb),
    'roc_auc': roc_auc_score(y_val, y_val_proba_xgb)
}

print(f"\nValidation Results:")
print(f"  Accuracy: {xgb_metrics['accuracy']:.4f}")
print(f"  Precision: {xgb_metrics['precision']:.4f}")
print(f"  Recall: {xgb_metrics['recall']:.4f}")
print(f"  F1-Score: {xgb_metrics['f1_score']:.4f}")
print(f"  ROC-AUC: {xgb_metrics['roc_auc']:.4f}")

# Save model
with open('model_xgboost.pkl', 'wb') as f:
    pickle.dump(xgb_model, f)
print("\n[SAVED] model_xgboost.pkl")

# ============================================================================
# MODEL COMPARISON
# ============================================================================
print("\n" + "="*80)
print("MODEL COMPARISON SUMMARY")
print("="*80)

comparison_df = pd.DataFrame([lr_metrics, rf_metrics, xgb_metrics])
print("\n" + comparison_df.to_string(index=False))

# Save comparison
comparison_df.to_csv('model_comparison.csv', index=False)
print("\n[SAVED] model_comparison.csv")

# Find best model
best_model_idx = comparison_df['f1_score'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'model']
print(f"\nBest Model (by F1-Score): {best_model_name}")

# ============================================================================
# SAVE TRAINING METADATA
# ============================================================================
metadata = {
    'training_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'training_data': TRAIN_DATA,
    'validation_data': VALIDATION_DATA,
    'train_samples': len(X_train),
    'validation_samples': len(X_val),
    'num_features': X_train.shape[1],
    'best_model': best_model_name,
    'models_trained': ['Logistic Regression', 'Random Forest', 'XGBoost']
}

with open('training_metadata.json', 'w') as f:
    json.dump(metadata, f, indent=4)
print("\n[SAVED] training_metadata.json")

print("\n" + "="*80)
print("TRAINING COMPLETE")
print("="*80)
print("\nNext step: Run validate_model.py to perform detailed validation analysis")
print("="*80)
