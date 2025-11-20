# Credit Score Classification (Supervised Learning Project)

End-to-end machine learning pipeline to classify customer credit scores using structured banking information, rigorous preprocessing, and model comparison.

## 📊 Dataset
- Located in `data/raw/`
- Target variable: `Credit_Score` (Poor, Standard, Good)
- Features cover demographics, repayment history, credit utilization, loan portfolio, and monthly balances.

## 🔄 Workflow
1. **Exploratory Data Analysis**
   - Distribution and imbalance review
   - Detection of corrupted categorical tokens
   - Outlier inspection with IQR filters
2. **Data Cleaning & Feature Engineering**
   - Removal of identifiers and low-information columns
   - Text normalization and categorical regrouping
   - Conversion of `Credit_History_Age` to numeric years
   - Ordinal/one-hot encoding plus negative-value remediation
3. **Modeling & Validation**
   - Train/test split with stratification
   - Standardization pipeline for distance-based models
   - GridSearchCV over Decision Tree, Random Forest, and KNN
   - Bootstrap stress-testing for robustness
4. **Evaluation**
   - Accuracy, precision, recall, F1, and confusion matrices
   - Business-focused inspection of false positives vs. false negatives
   - Error profiling to understand risky customer segments

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
```

## 🚀 Getting Started
1. Create/activate a Python 3.10+ environment.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Open the notebooks in `notebooks/` to reproduce the full analysis, or adapt the pipeline inside `src/` for production use.

## ✅ Status
- Data cleaning, feature engineering, and baseline models complete.
- Best-performing model: Random Forest tuned via GridSearchCV (highest accuracy, balanced recall).
- Next steps: automate training script in `src/`, add model persistence, and publish evaluation dashboards in `reports/figures`.
