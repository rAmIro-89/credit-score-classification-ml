# Credit Score Classification (Supervised Learning Project)

End-to-end machine learning pipeline to classify customer credit scores using structured banking information, rigorous preprocessing, and model comparison.  
Includes business interpretation for financial risk teams.

## 📊 Dataset
- Located in `data/raw/`
- Target variable: `Credit_Score` (`Poor`, `Standard`, `Good`)
- Features include demographics, repayment history, credit utilization, loan portfolio, and monthly balances.

## 🔄 Workflow

### 1. Exploratory Data Analysis
- Class distribution and imbalance review  
- Detection of corrupted categorical tokens  
- Outlier inspection with IQR filters  

### 2. Data Cleaning & Feature Engineering
- Removal of identifiers and low-information columns  
- Text normalization and categorical regrouping  
- Transformation of `Credit_History_Age` to numeric years  
- Ordinal + one-hot encoding  
- Negative-value remediation and numeric quality checks  

### 3. Modeling & Validation
- Stratified train/test split  
- Standardization pipeline for distance-based models  
- `GridSearchCV` tuning for Decision Tree, Random Forest, and KNN  
- Bootstrap stress-testing for robustness  

### 4. Evaluation
- Accuracy, Precision, Recall, F1  
- Confusion matrices and error profiling  
- Business inspection of false positives vs false negatives  
- Comparison across all tuned and bootstrapped models  

---

# 📈 Model Performance (Test Set)

<div align="center">

| Model | CV Accuracy | Test Accuracy | Macro F1 |
|-------|------------:|--------------:|---------:|
| Decision Tree | 0.730 | 0.731 | 0.70 |
| KNN | 0.719 | 0.727 | 0.70 |
| <strong>Random Forest <span style="color:green">🏆 Best Model</span></strong> | <strong>0.785</strong> | <strong>0.788</strong> | <strong>0.78</strong> |

</div>

---

## 📁 Project Structure
```text
credit-score-classification-ml/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
├── reports/
│   └── figures/
├── src/
├── models/
└── README.md
