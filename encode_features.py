import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("FEATURE ENCODING & PREPROCESSING")
print("="*80)

# Load the split datasets
print("\nLoading split datasets...")
train_df = pd.read_csv('fraud_train.csv')
val_df = pd.read_csv('fraud_validation.csv')
test_df = pd.read_csv('fraud_test.csv')

print(f"Train shape: {train_df.shape}")
print(f"Validation shape: {val_df.shape}")
print(f"Test shape: {test_df.shape}")

# ============================================================================
# STEP 1: DROP ZERO-VARIANCE FEATURES
# ============================================================================
print("\n" + "-"*80)
print("STEP 1: DROPPING ZERO-VARIANCE FEATURES")
print("-"*80)

columns_to_drop = ['zipcodeOri', 'zipMerchant']
print(f"Dropping: {columns_to_drop}")

train_df = train_df.drop(columns=columns_to_drop)
val_df = val_df.drop(columns=columns_to_drop)
test_df = test_df.drop(columns=columns_to_drop)

print(f"New shape: {train_df.shape}")

# ============================================================================
# STEP 2: TARGET ENCODING FOR CUSTOMER (High Cardinality)
# ============================================================================
print("\n" + "-"*80)
print("STEP 2: TARGET ENCODING FOR CUSTOMER")
print("-"*80)

# Calculate fraud rate per customer from TRAINING set only
customer_fraud_rate = train_df.groupby('customer')['fraud'].agg(['mean', 'count']).reset_index()
customer_fraud_rate.columns = ['customer', 'customer_fraud_rate', 'customer_transaction_count']

# Add smoothing to prevent overfitting (using global fraud rate)
global_fraud_rate = train_df['fraud'].mean()
min_samples = 5  # Minimum transactions to trust the customer's fraud rate

# Apply smoothing: if customer has < min_samples, blend with global rate
customer_fraud_rate['customer_fraud_rate_smoothed'] = customer_fraud_rate.apply(
    lambda row: (row['customer_fraud_rate'] * row['customer_transaction_count'] + 
                 global_fraud_rate * min_samples) / 
                (row['customer_transaction_count'] + min_samples),
    axis=1
)

print(f"Unique customers in train: {train_df['customer'].nunique()}")
print(f"Global fraud rate: {global_fraud_rate:.4f}")
print(f"Customer fraud rate range: {customer_fraud_rate['customer_fraud_rate_smoothed'].min():.4f} - {customer_fraud_rate['customer_fraud_rate_smoothed'].max():.4f}")

# Map to all datasets
train_df = train_df.merge(customer_fraud_rate[['customer', 'customer_fraud_rate_smoothed', 'customer_transaction_count']], 
                          on='customer', how='left')
val_df = val_df.merge(customer_fraud_rate[['customer', 'customer_fraud_rate_smoothed', 'customer_transaction_count']], 
                      on='customer', how='left')
test_df = test_df.merge(customer_fraud_rate[['customer', 'customer_fraud_rate_smoothed', 'customer_transaction_count']], 
                        on='customer', how='left')

# Fill unseen customers with global fraud rate
train_df['customer_fraud_rate_smoothed'].fillna(global_fraud_rate, inplace=True)
val_df['customer_fraud_rate_smoothed'].fillna(global_fraud_rate, inplace=True)
test_df['customer_fraud_rate_smoothed'].fillna(global_fraud_rate, inplace=True)

train_df['customer_transaction_count'].fillna(0, inplace=True)
val_df['customer_transaction_count'].fillna(0, inplace=True)
test_df['customer_transaction_count'].fillna(0, inplace=True)

# Drop original customer column
train_df = train_df.drop('customer', axis=1)
val_df = val_df.drop('customer', axis=1)
test_df = test_df.drop('customer', axis=1)

print("Created features: customer_fraud_rate_smoothed, customer_transaction_count")

# ============================================================================
# STEP 3: ORDINAL ENCODING FOR AGE
# ============================================================================
print("\n" + "-"*80)
print("STEP 3: ORDINAL ENCODING FOR AGE")
print("-"*80)

# Age mapping: 0 <= 18, 1: 19-25, 2: 26-35, 3: 36-45, 4: 46-55, 5: 56-65, 6: > 65, U: Unknown
# Remove quotes from age values
train_df['age'] = train_df['age'].str.replace("'", "")
val_df['age'] = val_df['age'].str.replace("'", "")
test_df['age'] = test_df['age'].str.replace("'", "")

age_mapping = {'0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, 'U': 7}
train_df['age'] = train_df['age'].map(age_mapping)
val_df['age'] = val_df['age'].map(age_mapping)
test_df['age'] = test_df['age'].map(age_mapping)

print(f"Age mapping applied: {age_mapping}")
print(f"Age distribution in train:\n{train_df['age'].value_counts().sort_index()}")

# ============================================================================
# STEP 4: ONE-HOT ENCODING FOR GENDER
# ============================================================================
print("\n" + "-"*80)
print("STEP 4: ONE-HOT ENCODING FOR GENDER")
print("-"*80)

# Remove quotes from gender
train_df['gender'] = train_df['gender'].str.replace("'", "")
val_df['gender'] = val_df['gender'].str.replace("'", "")
test_df['gender'] = test_df['gender'].str.replace("'", "")

# One-hot encode gender
train_df = pd.get_dummies(train_df, columns=['gender'], prefix='gender', drop_first=False)
val_df = pd.get_dummies(val_df, columns=['gender'], prefix='gender', drop_first=False)
test_df = pd.get_dummies(test_df, columns=['gender'], prefix='gender', drop_first=False)

# Ensure all datasets have the same gender columns
gender_cols = [col for col in train_df.columns if col.startswith('gender_')]
for col in gender_cols:
    if col not in val_df.columns:
        val_df[col] = 0
    if col not in test_df.columns:
        test_df[col] = 0

print(f"Gender columns created: {gender_cols}")

# ============================================================================
# STEP 5: ONE-HOT ENCODING FOR CATEGORY
# ============================================================================
print("\n" + "-"*80)
print("STEP 5: ONE-HOT ENCODING FOR CATEGORY")
print("-"*80)

# Remove quotes from category
train_df['category'] = train_df['category'].str.replace("'", "")
val_df['category'] = val_df['category'].str.replace("'", "")
test_df['category'] = test_df['category'].str.replace("'", "")

print(f"Unique categories: {train_df['category'].nunique()}")

# One-hot encode category
train_df = pd.get_dummies(train_df, columns=['category'], prefix='category', drop_first=False)
val_df = pd.get_dummies(val_df, columns=['category'], prefix='category', drop_first=False)
test_df = pd.get_dummies(test_df, columns=['category'], prefix='category', drop_first=False)

# Ensure all datasets have the same category columns
category_cols = [col for col in train_df.columns if col.startswith('category_')]
for col in category_cols:
    if col not in val_df.columns:
        val_df[col] = 0
    if col not in test_df.columns:
        test_df[col] = 0

print(f"Category columns created: {len(category_cols)}")

# ============================================================================
# STEP 6: ONE-HOT ENCODING FOR MERCHANT
# ============================================================================
print("\n" + "-"*80)
print("STEP 6: ONE-HOT ENCODING FOR MERCHANT")
print("-"*80)

# Remove quotes from merchant
train_df['merchant'] = train_df['merchant'].str.replace("'", "")
val_df['merchant'] = val_df['merchant'].str.replace("'", "")
test_df['merchant'] = test_df['merchant'].str.replace("'", "")

print(f"Unique merchants: {train_df['merchant'].nunique()}")

# One-hot encode merchant
train_df = pd.get_dummies(train_df, columns=['merchant'], prefix='merchant', drop_first=False)
val_df = pd.get_dummies(val_df, columns=['merchant'], prefix='merchant', drop_first=False)
test_df = pd.get_dummies(test_df, columns=['merchant'], prefix='merchant', drop_first=False)

# Ensure all datasets have the same merchant columns
merchant_cols = [col for col in train_df.columns if col.startswith('merchant_')]
for col in merchant_cols:
    if col not in val_df.columns:
        val_df[col] = 0
    if col not in test_df.columns:
        test_df[col] = 0

print(f"Merchant columns created: {len(merchant_cols)}")

# ============================================================================
# STEP 7: STANDARDIZE NUMERICAL FEATURES
# ============================================================================
print("\n" + "-"*80)
print("STEP 7: STANDARDIZING NUMERICAL FEATURES")
print("-"*80)

numerical_features = ['step', 'amount']
print(f"Numerical features to scale: {numerical_features}")

# Fit scaler on training data only
scaler = StandardScaler()
train_df[numerical_features] = scaler.fit_transform(train_df[numerical_features])
val_df[numerical_features] = scaler.transform(val_df[numerical_features])
test_df[numerical_features] = scaler.transform(test_df[numerical_features])

print("Scaling complete (fitted on train, applied to all)")

# ============================================================================
# FINAL SUMMARY
# ============================================================================
print("\n" + "="*80)
print("ENCODING SUMMARY")
print("="*80)

print(f"\nFinal shapes:")
print(f"  Train: {train_df.shape}")
print(f"  Validation: {val_df.shape}")
print(f"  Test: {test_df.shape}")

print(f"\nFeature count: {train_df.shape[1] - 1} features + 1 target (fraud)")

print(f"\nColumn alignment check:")
print(f"  Train columns: {len(train_df.columns)}")
print(f"  Val columns: {len(val_df.columns)}")
print(f"  Test columns: {len(test_df.columns)}")
print(f"  All match: {len(train_df.columns) == len(val_df.columns) == len(test_df.columns)}")

# Ensure column order is the same
val_df = val_df[train_df.columns]
test_df = test_df[train_df.columns]

# ============================================================================
# SAVE ENCODED DATASETS
# ============================================================================
print("\n" + "="*80)
print("SAVING ENCODED DATASETS")
print("="*80)

train_df.to_csv('fraud_train_encoded.csv', index=False)
print("[SUCCESS] Saved: fraud_train_encoded.csv")

val_df.to_csv('fraud_validation_encoded.csv', index=False)
print("[SUCCESS] Saved: fraud_validation_encoded.csv")

test_df.to_csv('fraud_test_encoded.csv', index=False)
print("[SUCCESS] Saved: fraud_test_encoded.csv")

print("\n" + "="*80)
print("ENCODING COMPLETE")
print("="*80)
print("\nYour data is now ready for model training!")
print("Next step: Handle class imbalance (SMOTE/class weights) and train models")
print("="*80)
