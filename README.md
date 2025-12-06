# Credit Score Classification – End-to-End ML Project

![Status](https://img.shields.io/badge/status-completed-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/scikit--learn-classification-orange)
![Jupyter](https://img.shields.io/badge/Jupyter-notebooks-F37626?logo=jupyter&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

End-to-end **credit risk classification** project: from data cleaning and EDA to feature engineering, model comparison, and business-oriented evaluation metrics.

The goal is to build a **production-ready baseline** that could be integrated into a real credit approval workflow (loans, credit cards, BNPL, etc.).

---

## 1. Problem Overview

Financial institutions need to decide whether to **approve or reject** new credit applications while controlling:

- Default risk (probability of non-payment)
- Profitability (interest vs expected losses)
- Operational constraints (regulation, fairness, explainability)

This project focuses on predicting a **binary target** (`good` vs `bad` credit) using demographic, behavioral and financial variables.

**Main objectives:**

- Build a robust ML pipeline to classify applicants.
- Handle **class imbalance** and evaluate metrics beyond accuracy.
- Compare different models and select a champion.
- Provide **interpretable insights** (feature importance, thresholds, trade-offs).

---

## 2. Dataset & Target

- **Observations:** credit applicants (historical data)
- **Target:** `default` / `bad` vs `non-default` / `good`
- **Features:** income, age, employment, credit history, loan amount, etc.
- **Task:** **binary classification** (creditworthy vs non-creditworthy)

### Class Balance

![Class Balance](img/01-class-balance.png)  
*Figure 1 – Class distribution: approved vs rejected / good vs bad borrowers.*

---

## 3. Project Structure

```text
credit-score-classification-ml/
├── data/
│   ├── raw/                  # Original dataset
│   └── processed/            # Cleaned / encoded data
├── notebooks/
│   ├── credit-score-eda-preprocessing.ipynb
│   └── credit-score-modeling-and-evaluation.ipynb
├── src/
│   ├── data_prep/            # Cleaning, encoding, train/test split
│   ├── models/               # Model training & evaluation functions
│   └── utils/                # Helpers, config, metrics
├── img/                      # Exported figures for the README
│   ├── 01-class-balance.png
│   ├── 02-feature-importance.png
│   ├── 03-roc-curve.png
│   └── 04-confusion-matrix.png
├── requirements.txt
└── README.md


4. EDA & Preprocessing

Key steps in credit-score-eda-preprocessing.ipynb:

Missing value analysis and imputation strategies.

Encoding of categorical variables (one-hot / target encoding).

Outlier inspection for numeric features.

Train / validation / test split with stratification on the target.

5. Modeling & Evaluation

Models evaluated:

Logistic Regression (baseline, interpretable)

Tree-based models (Random Forest / Gradient Boosting / XGBoost, etc.)

Class imbalance handling (class weights / resampling)

### 5.1 Feature Importance

![Feature Importance](img/02-feature-importance.png)
*Figure 2 – Top features driving the credit risk decision for the final model.*

### 5.2 ROC Curve & Thresholds

![ROC Curve](img/03-roc-curve.png)
*Figure 3 – ROC curve and AUC for the champion model. Used to study trade-offs between TPR and FPR.*

### 5.3 Confusion Matrix (Test Set)

![Confusion Matrix](img/04-confusion-matrix.png)
*Figure 4 – Confusion matrix on the hold-out test set.*

Metrics reported:

AUC-ROC

Precision, Recall, F1-score

Specificity & Sensitivity

Balanced Accuracy

Optional: cost-sensitive metrics (false negative vs false positive cost)

6. How to Run
6.1 Environment Setup
git clone https://github.com/rAmIro-89/credit-score-classification-ml.git
cd credit-score-classification-ml

python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

pip install -r requirements.txt

6.2 Run the Notebooks
jupyter notebook notebooks/credit-score-eda-preprocessing.ipynb
jupyter notebook notebooks/credit-score-modeling-and-evaluation.ipynb

## 7. Skills Demonstrated

- **Data Engineering**: Handling missing values, categorical encoding, stratified splitting
- **Classification Techniques**: Logistic Regression, Tree-based models (Decision Tree, Random Forest, XGBoost)
- **Class Imbalance Management**: Resampling, class weights, cost-sensitive evaluation
- **Hyperparameter Tuning**: Grid Search, cross-validation, bootstrap validation
- **Model Evaluation**: AUC-ROC, Precision-Recall, F1-Score, Confusion Matrices, threshold optimization
- **Business Interpretation**: Cost-benefit analysis, false positive/negative trade-offs, production readiness
- **ML Pipeline**: Clean, reproducible, modular code structure with proper documentation

## 8. License & Author

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

**Ramiro Ottone Villar**  
[![GitHub](https://img.shields.io/badge/GitHub-rAmIro--89-181717?style=flat&logo=github)](https://github.com/rAmIro-89)  
[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-0A66C2?style=flat&logo=linkedin)](https://linkedin.com/in/your-profile)

---

⭐ **If you find this project useful, please consider starring the repository!**


