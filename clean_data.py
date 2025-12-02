import pandas as pd

# Load the dataset
print("Loading fraud.csv...")
df = pd.read_csv('fraud.csv')

print("\n" + "="*80)
print("DATA CLEANING: REMOVING ZERO-AMOUNT TRANSACTIONS")
print("="*80)

# Show initial dataset info
print(f"\nOriginal dataset shape: {df.shape}")
print(f"Original number of rows: {df.shape[0]:,}")

# Count zero-amount rows
zero_amount_count = (df['amount'] == 0).sum()
print(f"\nRows with amount = 0: {zero_amount_count}")

# Remove rows where amount = 0
df_clean = df[df['amount'] > 0].copy()

print(f"\nCleaned dataset shape: {df_clean.shape}")
print(f"Cleaned number of rows: {df_clean.shape[0]:,}")
print(f"Rows removed: {df.shape[0] - df_clean.shape[0]}")

# Verify the cleaning
zero_amount_after = (df_clean['amount'] == 0).sum()
print(f"\nVerification - Rows with amount = 0 after cleaning: {zero_amount_after}")

# Show fraud distribution
print("\n" + "-"*80)
print("FRAUD DISTRIBUTION")
print("-"*80)
print(f"\nOriginal dataset:")
print(f"  - Non-fraud: {(df['fraud'] == 0).sum():,} ({(df['fraud'] == 0).sum()/len(df)*100:.2f}%)")
print(f"  - Fraud: {(df['fraud'] == 1).sum():,} ({(df['fraud'] == 1).sum()/len(df)*100:.2f}%)")

print(f"\nCleaned dataset:")
print(f"  - Non-fraud: {(df_clean['fraud'] == 0).sum():,} ({(df_clean['fraud'] == 0).sum()/len(df_clean)*100:.2f}%)")
print(f"  - Fraud: {(df_clean['fraud'] == 1).sum():,} ({(df_clean['fraud'] == 1).sum()/len(df_clean)*100:.2f}%)")

# Save the cleaned dataset
output_file = 'fraud_cleaned.csv'
df_clean.to_csv(output_file, index=False)
print("\n" + "="*80)
print(f"[SUCCESS] Cleaned dataset saved to: {output_file}")
print("="*80)

# Show basic statistics
print("\nAmount statistics (cleaned data):")
print(df_clean['amount'].describe())

print("\n" + "="*80)
print("CLEANING COMPLETE")
print("="*80)
