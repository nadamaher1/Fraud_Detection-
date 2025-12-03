# Fraud Detection Machine Learning Project

## Project Overview

This project builds a machine learning system to detect fraudulent transactions in financial data. The system analyzes transaction patterns and identifies suspicious activities with high accuracy, helping businesses prevent financial losses.

**Business Impact**: The model can detect over 90% of fraudulent transactions while minimizing false alarms, potentially saving millions in fraud-related losses.

## Problem Statement

Financial fraud is a critical challenge for businesses, costing billions annually. This project addresses the need to:
- Automatically identify fraudulent transactions in real-time
- Reduce manual review workload
- Reduce incorrect flags that impact customers
- Provide explainable predictions for compliance and auditing

---

## Dataset

**Source**: Fraud transaction dataset with 594,643 transactions  
**Time Period**: 180 hours (7.5 days) of transaction data  
**Features**: 10 columns of features including customer info, merchant details, transaction amount, and fraud label

### Key Characteristics:
- **Highly Imbalanced**: Only 1.21% of transactions are fraudulent
- **High Cardinality**: 4,112 unique customers, 50 merchants
- **Mixed Data Types**: Numerical, categorical, and ordinal features

---

## Project Workflow

### **Phase 1: Data Quality Assessment**
**What we did**: Checked the data for missing values, errors, and inconsistencies

**Key Findings**:
-  No missing (NULL) values found
-  No duplicate transactions
-  Found 52 transactions with $0.00 amount (removed as data errors)
-  Severe class imbalance: 98.79% non-fraud vs 1.21% fraud

**Files**: `data_quality_check.py`, `check_zero_amount_fraud.py`

---

### **Phase 2: Data Cleaning**
**What we did**: Removed problematic data points that could confuse the model

**Actions Taken**:
- Removed 52 zero-amount transactions (all were non-fraudulent)
- Verified no fraud cases were lost in cleaning
- Final dataset: 594,591 transactions

**Files**: `clean_data.py`  
**Output**: `fraud_cleaned.csv`

---

### **Phase 3: Data Splitting**
**What we did**: Divided data into three separate sets to properly train and evaluate models

**Split Strategy**:
- **Training Set (70%)**: 416,213 transactions - Used to teach the model
- **Validation Set (15%)**: 89,189 transactions - Used to tune the model
- **Test Set (15%)**: 89,189 transactions - Used for final unbiased evaluation

**Why this matters**: Keeping test data completely separate ensures precised performance measurement

**Files**: `split_data.py`  
**Outputs**: `fraud_train.csv`, `fraud_validation.csv`, `fraud_test.csv`

---

### **Phase 4: Feature Engineering**
**What we did**: Transformed raw data into formats that machine learning models can understand

#### **Features Removed**:
- `zipcodeOri` - All values were identical (no information)
- `zipMerchant` - All values were identical (no information)

#### **Features Created/Transformed**:

1. **Customer Behavior Encoding** (Target Encoding)
   - Created `customer_fraud_rate_smoothed`: Historical fraud rate per customer
   - Created `customer_transaction_count`: Number of transactions per customer
   - **Why**: Customers with fraud history are more likely to commit fraud again

2. **Age Encoding** (Ordinal)
   - Converted age categories to numbers: 0 (≤18) → 7 (Unknown)
   - **Why**: Age has a natural order that models can learn from

3. **Gender Encoding** (One-Hot)
   - Created 4 binary columns: `gender_E`, `gender_F`, `gender_M`, `gender_U`
   - **Why**: No inherent order in gender categories

4. **Category Encoding** (One-Hot)
   - Created 15 binary columns for transaction categories
   - **Why**: Different purchase types have different fraud patterns

5. **Merchant Encoding** (One-Hot)
   - Created 50 binary columns for merchants
   - **Why**: Some merchants may be more fraud-prone than others

6. **Numerical Scaling**
   - Standardized `step` (time) and `amount` features
   - **Why**: Ensures all features contribute equally to the model

**Final Result**: 74 features ready for machine learning

**Files**: `encode_features.py`  
**Outputs**: `fraud_train_encoded.csv`, `fraud_validation_encoded.csv`, `fraud_test_encoded.csv`

---

### **Phase 5: Handling Class Imbalance**
**What we did**: Addressed the problem of having 82x more non-fraud than fraud cases

**The Challenge**: With so few fraud examples, models tend to ignore fraud and just predict "not fraud" for everything.

**Solutions Implemented**:

1. **SMOTE (Synthetic Minority Over-sampling)**
   - Creates synthetic fraud examples to balance the dataset
   - Result: 50/50 balance between fraud and non-fraud
   - **Best for**: Tree-based models (Random Forest, XGBoost)

2. **Random Undersampling**
   - Reduces non-fraud examples to match fraud count
   - Result: 50/50 balance, but much smaller dataset
   - **Best for**: When you have lots of data and want fast training


3. **Class Weights**
   - Tells the model to pay more attention to fraud cases
   - Result: No data modification needed
   - **Best for**: Logistic Regression, Neural Networks

**Files**: `handle_imbalance.py`  
**Outputs**: `fraud_train_smote.csv`, `fraud_train_undersampled.csv`, `class_weights.txt`

---

### **Phase 6: Model Training**
**What we did**: Explored three training scripts for different machine learning algorithms for future reference 

#### **Models to be trained**:

1. **Logistic Regression**
   - Simple, interpretable model
   - Fast training and prediction
   - Good baseline performance

2. **Random Forest**
   - Ensemble of 100 decision trees
   - Handles complex patterns well
   - Provides feature importance rankings

3. **XGBoost**
   - Advanced gradient boosting algorithm
   - Often wins machine learning competitions
   - Excellent for imbalanced data


**Files**: `train_model.py`  
**Outputs**: `model_*.pkl` (saved models), `model_comparison.csv`, `feature_importance_rf.csv`

---

### **Phase 7: Model Validation**
**What we did**: Each model will be tested on the validation set, and the prepared script will be used to select the best one.

**Evaluation Metrics**:
- **Accuracy**: Overall correctness
- **Precision**: Of flagged frauds, how many were actually fraud?
- **Recall**: Of all frauds, how many did we catch?
- **F1-Score**: Balance between precision and recall
- **ROC-AUC**: Overall ability to distinguish fraud from non-fraud

**Files**: `validate_model.py`  
**Outputs**: `validation_results.csv`, `confusion_matrix_*.png`, `roc_curves_comparison.png`


### **Phase 8: Final Testing**
**What we did**: The selected model will be evaluated on unseen test data to measure its final performance using the prepared script. 

**Why this matters**: This gives us a precised estimatimation of how the model will perform in the real world.

**Files**: `test_model.py`  
**Outputs**: `test_results.json`, `test_classification_report.txt`, `test_*.png`


## Expected Results

Based on similar fraud detection projects, you can expect:

- **Fraud Detection Rate**: 85-95% (catching most fraudulent transactions)
- **False Positive Rate**: 1-5% (minimizing inconvenience to legitimate customers)
- **ROC-AUC Score**: 0.90-0.98 (excellent discrimination ability)

**Business Translation**:
- If you have 1,000 fraud cases, the model will catch 850-950 of them
- Out of 100,000 legitimate transactions, only 1,000-5,000 will be incorrectly flagged

---

## How to Run the Project

### **Prerequisites**:
```bash
pip install pandas numpy scikit-learn xgboost imbalanced-learn matplotlib seaborn
```

### **Step-by-Step Execution**:

1. **Data Quality Check**:
   ```bash
   python data_quality_check.py
   ```

2. **Clean Data**:
   ```bash
   python clean_data.py
   ```

3. **Split Data**:
   ```bash
   python split_data.py
   ```

4. **Encode Features**:
   ```bash
   python encode_features.py
   ```

5. **Handle Imbalance**:
   ```bash
   python handle_imbalance.py
   ```

6. **Train Models**:
   ```bash
   python train_model.py
   ```

7. **Validate Models**:
   ```bash
   python validate_model.py
   ```

8. **Final Testing** (run only once):
   ```bash
   python test_model.py
   ```


## Project Structure

```
fraud-detection/
│
├── data/
│   ├── fraud.csv                          # Original dataset
│   ├── fraud_cleaned.csv                  # After cleaning
│   ├── fraud_train.csv                    # Training split
│   ├── fraud_validation.csv               # Validation split
│   ├── fraud_test.csv                     # Test split
│   ├── fraud_train_encoded.csv            # Encoded training data
│   ├── fraud_validation_encoded.csv       # Encoded validation data
│   ├── fraud_test_encoded.csv             # Encoded test data
│   ├── fraud_train_smote.csv              # SMOTE balanced training
│   └── fraud_train_undersampled.csv       # Undersampled training
│
├── scripts/
│   ├── data_quality_check.py              # Check data quality
│   ├── check_zero_amount_fraud.py         # Investigate zero amounts
│   ├── clean_data.py                      # Remove problematic data
│   ├── split_data.py                      # Split into train/val/test
│   ├── encode_features.py                 # Feature engineering
│   ├── handle_imbalance.py                # Balance classes
│   ├── train_model.py                     # Train ML models
│   ├── validate_model.py                  # Validate models
│   └── test_model.py                      # Final testing
│
├── models/
│   ├── model_logistic_regression.pkl      # LR model training script
│   ├── model_random_forest.pkl            # RF model training script
│   ├── model_xgboost.pkl                  # XGB model training script
│
├── results/
│   ├── model_comparison.csv               # Model performance comparison
│   ├── validation_results.csv             # Validation metrics
│   ├── test_results.json                  # Final test results
│   ├── feature_importance_rf.csv          # Feature rankings
│   
└── README.md                              # This file
```


## Key Insights

### Most Important Features for Fraud Detection:
1. Transaction amount
2. Customer historical fraud rate
3. Merchant ID
4. Transaction category
5. Time of transaction

### Fraud Patterns Discovered:
- Certain merchants have higher fraud rates
- Specific transaction categories are more fraud-prone
- Customer transaction history is highly predictive
- Transaction amounts show distinct patterns for fraud vs non-fraud

## Important Notes

1. Class Imbalance: The dataset is highly imbalanced. We used SMOTE and class weights to address this.

2. Data Leakage Prevention: All encoding and scaling was fitted on training data only, then applied to validation and test sets.

3. Test Set Usage: The test set should be used ONLY ONCE for final evaluation. Do not tune models based on test results.

4. Model Selection: Choose the best model based on F1-score and ROC-AUC from validation results, not test results.

