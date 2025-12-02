import pandas as pd
from sklearn.model_selection import train_test_split

print("="*80)
print("DATA SPLITTING: TRAIN / VALIDATION / TEST")
print("="*80)

# Load the cleaned dataset
print("\nLoading fraud_cleaned.csv...")
df = pd.read_csv('fraud_cleaned.csv')

print(f"Total dataset shape: {df.shape}")
print(f"Total rows: {df.shape[0]:,}")

# Check fraud distribution
fraud_count = df['fraud'].sum()
non_fraud_count = (df['fraud'] == 0).sum()
print(f"\nFraud distribution:")
print(f"  - Non-fraud (0): {non_fraud_count:,} ({non_fraud_count/len(df)*100:.2f}%)")
print(f"  - Fraud (1): {fraud_count:,} ({fraud_count/len(df)*100:.2f}%)")

print("\n" + "-"*80)
print("SPLITTING STRATEGY")
print("-"*80)
print("Train: 70% | Validation: 15% | Test: 15%")
print("Using stratified split to maintain fraud ratio in all sets")

# Separate features and target
X = df.drop('fraud', axis=1)
y = df['fraud']

# First split: 70% train, 30% temp (validation + test)
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, 
    test_size=0.30, 
    random_state=42, 
    stratify=y
)

# Second split: Split temp into 50% validation, 50% test (15% each of total)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, 
    test_size=0.50, 
    random_state=42, 
    stratify=y_temp
)

# Combine features and target back together for each set
train_df = pd.concat([X_train, y_train], axis=1)
val_df = pd.concat([X_val, y_val], axis=1)
test_df = pd.concat([X_test, y_test], axis=1)

print("\n" + "-"*80)
print("SPLIT RESULTS")
print("-"*80)

# Train set
print(f"\nTRAIN SET:")
print(f"  Shape: {train_df.shape}")
print(f"  Rows: {len(train_df):,} ({len(train_df)/len(df)*100:.1f}% of total)")
print(f"  Non-fraud: {(train_df['fraud'] == 0).sum():,} ({(train_df['fraud'] == 0).sum()/len(train_df)*100:.2f}%)")
print(f"  Fraud: {(train_df['fraud'] == 1).sum():,} ({(train_df['fraud'] == 1).sum()/len(train_df)*100:.2f}%)")

# Validation set
print(f"\nVALIDATION SET:")
print(f"  Shape: {val_df.shape}")
print(f"  Rows: {len(val_df):,} ({len(val_df)/len(df)*100:.1f}% of total)")
print(f"  Non-fraud: {(val_df['fraud'] == 0).sum():,} ({(val_df['fraud'] == 0).sum()/len(val_df)*100:.2f}%)")
print(f"  Fraud: {(val_df['fraud'] == 1).sum():,} ({(val_df['fraud'] == 1).sum()/len(val_df)*100:.2f}%)")

# Test set
print(f"\nTEST SET:")
print(f"  Shape: {test_df.shape}")
print(f"  Rows: {len(test_df):,} ({len(test_df)/len(df)*100:.1f}% of total)")
print(f"  Non-fraud: {(test_df['fraud'] == 0).sum():,} ({(test_df['fraud'] == 0).sum()/len(test_df)*100:.2f}%)")
print(f"  Fraud: {(test_df['fraud'] == 1).sum():,} ({(test_df['fraud'] == 1).sum()/len(test_df)*100:.2f}%)")

# Verify total
total_rows = len(train_df) + len(val_df) + len(test_df)
print(f"\nVerification - Total rows across all sets: {total_rows:,}")
print(f"Original dataset rows: {len(df):,}")
print(f"Match: {'YES' if total_rows == len(df) else 'NO'}")

# Save the splits to separate CSV files
print("\n" + "="*80)
print("SAVING SPLITS TO FILES")
print("="*80)

train_file = 'fraud_train.csv'
val_file = 'fraud_validation.csv'
test_file = 'fraud_test.csv'

train_df.to_csv(train_file, index=False)
print(f"\n[SUCCESS] Train set saved to: {train_file}")

val_df.to_csv(val_file, index=False)
print(f"[SUCCESS] Validation set saved to: {val_file}")

test_df.to_csv(test_file, index=False)
print(f"[SUCCESS] Test set saved to: {test_file}")

print("\n" + "="*80)
print("SPLITTING COMPLETE")
print("="*80)
print("\nNext steps:")
print("1. Use fraud_train.csv for training and fitting encoders/scalers")
print("2. Use fraud_validation.csv for hyperparameter tuning")
print("3. Use fraud_test.csv for final model evaluation (DO NOT touch until the end!)")
print("\nRemember: Fit all transformations (encoding, scaling) on TRAIN set only!")
print("="*80)
