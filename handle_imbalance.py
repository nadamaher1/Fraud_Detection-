import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("HANDLING CLASS IMBALANCE")
print("="*80)

# Load encoded training data
print("\nLoading encoded training data...")
train_df = pd.read_csv('fraud_train_encoded.csv')

print(f"Training data shape: {train_df.shape}")

# Separate features and target
X_train = train_df.drop('fraud', axis=1)
y_train = train_df['fraud']

print(f"\nOriginal class distribution:")
print(f"  Non-fraud (0): {(y_train == 0).sum():,} ({(y_train == 0).sum()/len(y_train)*100:.2f}%)")
print(f"  Fraud (1): {(y_train == 1).sum():,} ({(y_train == 1).sum()/len(y_train)*100:.2f}%)")
print(f"  Imbalance ratio: 1:{(y_train == 0).sum()/(y_train == 1).sum():.1f}")

# ============================================================================
# TECHNIQUE 1: SMOTE (Synthetic Minority Over-sampling)
# ============================================================================
print("\n" + "-"*80)
print("TECHNIQUE 1: SMOTE (Synthetic Minority Over-sampling)")
print("-"*80)

# Apply SMOTE to balance the classes
smote = SMOTE(random_state=42, k_neighbors=5)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"  Shape: {X_train_smote.shape}")
print(f"  Non-fraud (0): {(y_train_smote == 0).sum():,} ({(y_train_smote == 0).sum()/len(y_train_smote)*100:.2f}%)")
print(f"  Fraud (1): {(y_train_smote == 1).sum():,} ({(y_train_smote == 1).sum()/len(y_train_smote)*100:.2f}%)")
print(f"  Balance ratio: 1:{(y_train_smote == 0).sum()/(y_train_smote == 1).sum():.1f}")

# Save SMOTE balanced dataset
train_smote_df = pd.concat([pd.DataFrame(X_train_smote, columns=X_train.columns), 
                             pd.Series(y_train_smote, name='fraud')], axis=1)
train_smote_df.to_csv('fraud_train_smote.csv', index=False)
print("[SUCCESS] Saved: fraud_train_smote.csv")

# ============================================================================
# TECHNIQUE 2: RANDOM UNDERSAMPLING
# ============================================================================
print("\n" + "-"*80)
print("TECHNIQUE 2: RANDOM UNDERSAMPLING")
print("-"*80)

# Undersample majority class to match minority class
rus = RandomUnderSampler(random_state=42)
X_train_under, y_train_under = rus.fit_resample(X_train, y_train)

print(f"\nAfter Undersampling:")
print(f"  Shape: {X_train_under.shape}")
print(f"  Non-fraud (0): {(y_train_under == 0).sum():,} ({(y_train_under == 0).sum()/len(y_train_under)*100:.2f}%)")
print(f"  Fraud (1): {(y_train_under == 1).sum():,} ({(y_train_under == 1).sum()/len(y_train_under)*100:.2f}%)")
print(f"  Balance ratio: 1:{(y_train_under == 0).sum()/(y_train_under == 1).sum():.1f}")

# Save undersampled dataset
train_under_df = pd.concat([pd.DataFrame(X_train_under, columns=X_train.columns), 
                             pd.Series(y_train_under, name='fraud')], axis=1)
train_under_df.to_csv('fraud_train_undersampled.csv', index=False)
print("[SUCCESS] Saved: fraud_train_undersampled.csv")


# ============================================================================
# TECHNIQUE 3: CLASS WEIGHTS (No resampling - for model parameter)
# ============================================================================
print("\n" + "-"*80)
print("TECHNIQUE 4: CLASS WEIGHTS CALCULATION")
print("-"*80)

# Calculate class weights for use in model training
from sklearn.utils.class_weight import compute_class_weight

class_weights = compute_class_weight('balanced', 
                                     classes=np.unique(y_train), 
                                     y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

print(f"\nCalculated class weights:")
print(f"  Class 0 (Non-fraud): {class_weight_dict[0]:.4f}")
print(f"  Class 1 (Fraud): {class_weight_dict[1]:.4f}")
print(f"  Weight ratio: 1:{class_weight_dict[1]/class_weight_dict[0]:.1f}")
print("\nNote: Use these weights in model training (e.g., class_weight parameter)")
print("      This approach doesn't modify the dataset, just adjusts model learning")

# Save class weights to a text file
with open('class_weights.txt', 'w') as f:
    f.write("CLASS WEIGHTS FOR MODEL TRAINING\n")
    f.write("="*50 + "\n\n")
    f.write(f"Class 0 (Non-fraud): {class_weight_dict[0]:.6f}\n")
    f.write(f"Class 1 (Fraud): {class_weight_dict[1]:.6f}\n\n")
    f.write("Usage in scikit-learn:\n")
    f.write(f"class_weight = {class_weight_dict}\n")

print("[SUCCESS] Saved: class_weights.txt")

# ============================================================================
# COMPARISON SUMMARY
# ============================================================================
print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

summary_data = {
    'Technique': ['Original', 'SMOTE', 'Undersampling','Class Weights'],
    'Total Rows': [
        len(y_train),
        len(y_train_smote),
        len(y_train_under),
        len(y_train_hybrid),
        len(y_train)
    ],
    'Non-Fraud': [
        (y_train == 0).sum(),
        (y_train_smote == 0).sum(),
        (y_train_under == 0).sum(),
        (y_train_hybrid == 0).sum(),
        (y_train == 0).sum()
    ],
    'Fraud': [
        (y_train == 1).sum(),
        (y_train_smote == 1).sum(),
        (y_train_under == 1).sum(),
        (y_train_hybrid == 1).sum(),
        (y_train == 1).sum()
    ],
    'File': [
        'fraud_train_encoded.csv',
        'fraud_train_smote.csv',
        'fraud_train_undersampled.csv',
        'fraud_train_hybrid.csv',
        'fraud_train_encoded.csv + weights'
    ]
}

summary_df = pd.DataFrame(summary_data)
summary_df['Balance Ratio'] = summary_df['Non-Fraud'] / summary_df['Fraud']
summary_df['Balance Ratio'] = summary_df['Balance Ratio'].apply(lambda x: f"1:{x:.1f}")

print("\n" + summary_df.to_string(index=False))

# ============================================================================
# RECOMMENDATIONS
# ============================================================================
print("\n" + "="*80)
print("RECOMMENDATIONS")
print("="*80)

print("""
WHICH TECHNIQUE TO USE?

1. SMOTE (fraud_train_smote.csv)
   ✓ Best for: Most models, especially tree-based (Random Forest, XGBoost)
   ✓ Pros: Creates synthetic samples, maintains all original data
   ✗ Cons: Can create noise, increases training time significantly
   
2. UNDERSAMPLING (fraud_train_undersampled.csv)
   ✓ Best for: When you have LOTS of data and want faster training
   ✓ Pros: Fast training, reduces overfitting on majority class
   ✗ Cons: Loses 98% of non-fraud data, may miss patterns
   
3. CLASS WEIGHTS (use fraud_train_encoded.csv with weights)
   ✓ Best for: Logistic Regression, Neural Networks, when you can't resample
   ✓ Pros: No data modification, preserves original distribution
   ✗ Cons: May not work as well with some algorithms

RECOMMENDATION FOR YOUR CASE:
- Start with SMOTE (fraud_train_smote.csv) for initial models
- Try Class Weights for Logistic Regression
- Compare performance on validation set
""")

print("="*80)
print("CLASS IMBALANCE HANDLING COMPLETE")
print("="*80)
print("\nFiles created:")
print("  1. fraud_train_smote.csv (SMOTE balanced)")
print("  2. fraud_train_undersampled.csv (Undersampled)")
print("  3. class_weights.txt (Weights for model training)")
print("\nNext step: Train models using different balanced datasets!")
print("="*80)
