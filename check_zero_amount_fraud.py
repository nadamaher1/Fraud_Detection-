import pandas as pd

# Load the dataset
print("Loading fraud.csv...")
df = pd.read_csv('fraud.csv')

print("\n" + "="*80)
print("ZERO AMOUNT FRAUD CHECK")
print("="*80)

# Find rows where amount = 0
zero_amount = df[df['amount'] == 0]
print(f"\nTotal rows with amount = 0: {len(zero_amount)}")

# Find rows where amount = 0 AND fraud = 1
zero_amount_fraud = df[(df['amount'] == 0) & (df['fraud'] == 1)]
print(f"Rows with amount = 0 AND fraud = 1: {len(zero_amount_fraud)}")

# Find rows where amount = 0 AND fraud = 0
zero_amount_not_fraud = df[(df['amount'] == 0) & (df['fraud'] == 0)]
print(f"Rows with amount = 0 AND fraud = 0: {len(zero_amount_not_fraud)}")

# Display detailed information if any fraudulent zero-amount transactions exist
if len(zero_amount_fraud) > 0:
    print("\n" + "-"*80)
    print("[WARNING] Found FRAUDULENT transactions with ZERO amount!")
    print("-"*80)
    print("\nDetails of these suspicious transactions:")
    print(zero_amount_fraud.to_string(index=True))
    
    print("\n" + "-"*80)
    print("ANALYSIS:")
    print("-"*80)
    print(f"- This represents {(len(zero_amount_fraud)/len(zero_amount)*100):.2f}% of all zero-amount transactions")
    print(f"- This represents {(len(zero_amount_fraud)/df['fraud'].sum()*100):.2f}% of all fraudulent transactions")
    print("\nRECOMMENDATION: Investigate these cases separately before dropping!")
else:
    print("\n" + "-"*80)
    print("[OK] No fraudulent transactions with zero amount found")
    print("-"*80)
    print("\nRECOMMENDATION: Safe to drop all 52 zero-amount rows")

# Summary statistics for zero-amount transactions
print("\n" + "="*80)
print("ZERO AMOUNT TRANSACTIONS BREAKDOWN")
print("="*80)

print("\nCategory distribution:")
print(zero_amount['category'].value_counts().to_string())

print("\nGender distribution:")
print(zero_amount['gender'].value_counts().to_string())

print("\nAge distribution:")
print(zero_amount['age'].value_counts().to_string())

print("\n" + "="*80)
print("END OF REPORT")
print("="*80)
