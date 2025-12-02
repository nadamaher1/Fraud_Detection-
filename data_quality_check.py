import pandas as pd
import numpy as np

# Load the dataset
print("Loading fraud.csv...")
df = pd.read_csv('fraud.csv')

print("\n" + "="*80)
print("DATA QUALITY CHECK REPORT")
print("="*80)

# Basic dataset info
print(f"\nDataset Shape: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"\nColumns: {list(df.columns)}")

# 1. CHECK FOR NULL VALUES
print("\n" + "-"*80)
print("1. NULL VALUES CHECK")
print("-"*80)

null_counts = df.isnull().sum()
null_percentages = (df.isnull().sum() / len(df)) * 100

null_summary = pd.DataFrame({
    'Column': null_counts.index,
    'Null Count': null_counts.values,
    'Null Percentage': null_percentages.values
})

null_summary = null_summary[null_summary['Null Count'] > 0].sort_values('Null Count', ascending=False)

if len(null_summary) > 0:
    print("\n[WARNING] Found NULL values in the following columns:")
    print(null_summary.to_string(index=False))
else:
    print("\n[OK] No NULL values found in any column.")

# 2. CHECK FOR ZERO VALUES
print("\n" + "-"*80)
print("2. ZERO VALUES CHECK")
print("-"*80)

# Check zero values for numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns
zero_summary = []

for col in numeric_cols:
    zero_count = (df[col] == 0).sum()
    zero_percentage = (zero_count / len(df)) * 100
    if zero_count > 0:
        zero_summary.append({
            'Column': col,
            'Zero Count': zero_count,
            'Zero Percentage': f"{zero_percentage:.2f}%"
        })

if zero_summary:
    print("\nZero values found in numeric columns:")
    zero_df = pd.DataFrame(zero_summary)
    print(zero_df.to_string(index=False))
else:
    print("\n[OK] No zero values found in numeric columns.")

# 3. DATA TYPE CHECK
print("\n" + "-"*80)
print("3. DATA TYPES")
print("-"*80)
print("\n" + df.dtypes.to_string())

# 4. ADDITIONAL CHECKS
print("\n" + "-"*80)
print("4. ADDITIONAL DATA QUALITY CHECKS")
print("-"*80)

# Check for duplicate rows
duplicate_count = df.duplicated().sum()
if duplicate_count > 0:
    print(f"\n[WARNING] Found {duplicate_count} duplicate rows ({(duplicate_count/len(df)*100):.2f}%)")
else:
    print("\n[OK] No duplicate rows found.")

# Check for negative values in numeric columns (where they might not make sense)
print("\nNegative values in numeric columns:")
negative_found = False
for col in numeric_cols:
    negative_count = (df[col] < 0).sum()
    if negative_count > 0:
        print(f"  - {col}: {negative_count} negative values ({(negative_count/len(df)*100):.2f}%)")
        negative_found = True

if not negative_found:
    print("  [OK] No negative values found.")

# 5. SUMMARY STATISTICS
print("\n" + "-"*80)
print("5. SUMMARY STATISTICS FOR NUMERIC COLUMNS")
print("-"*80)
print("\n" + df.describe().to_string())

# 6. CATEGORICAL COLUMNS CHECK
print("\n" + "-"*80)
print("6. CATEGORICAL COLUMNS ANALYSIS")
print("-"*80)

categorical_cols = df.select_dtypes(include=['object']).columns
if len(categorical_cols) > 0:
    for col in categorical_cols:
        unique_count = df[col].nunique()
        print(f"\n{col}:")
        print(f"  - Unique values: {unique_count}")
        print(f"  - Sample values: {df[col].value_counts().head(5).to_dict()}")
else:
    print("\nNo categorical columns found.")

# 7. FINAL SUMMARY
print("\n" + "="*80)
print("FINAL SUMMARY")
print("="*80)

issues_found = []
if len(null_summary) > 0:
    issues_found.append(f"NULL values in {len(null_summary)} column(s)")
if zero_summary:
    issues_found.append(f"Zero values in {len(zero_summary)} numeric column(s)")
if duplicate_count > 0:
    issues_found.append(f"{duplicate_count} duplicate row(s)")

if issues_found:
    print("\n[ISSUES FOUND]:")
    for i, issue in enumerate(issues_found, 1):
        print(f"  {i}. {issue}")
else:
    print("\n[OK] No major data quality issues detected!")

print("\n" + "="*80)
print("END OF REPORT")
print("="*80)
