import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.multiclass import OneVsRestClassifier

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
DATA_PATH = os.path.join(REPO_ROOT, "data", "processed", "set_credit_score_ramior_ottone_villar.csv")
IMG_DIR = os.path.join(REPO_ROOT, "img")
os.makedirs(IMG_DIR, exist_ok=True)

# Load data
df = pd.read_csv(DATA_PATH)
X = df.drop("Credit_Score", axis=1)
y = df["Credit_Score"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 1) Class Balance
plt.figure(figsize=(6, 4))
sns.countplot(x=y)
plt.title("Class Balance")
plt.xlabel("Credit Score")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "01-class-balance.png"), dpi=150)
plt.close()

# 2) Feature Importance (Random Forest)
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
importances = pd.Series(rf.feature_importances_, index=X.columns)
top10 = importances.sort_values(ascending=False).head(10)

plt.figure(figsize=(8, 5))
top10.plot(kind="barh")
plt.gca().invert_yaxis()
plt.title("Top 10 Feature Importances (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "02-feature-importance.png"), dpi=150)
plt.close()

# 3) ROC Curve (One-vs-Rest)
classes = sorted(y.unique())
y_bin = label_binarize(y, classes=classes)
X_train_full, X_val, y_train_full, y_val = train_test_split(
    X, y_bin, test_size=0.2, random_state=42, stratify=y
)

over_rf = OneVsRestClassifier(
    RandomForestClassifier(n_estimators=200, random_state=42)
)
over_rf.fit(X_train_full, y_train_full)
y_score = over_rf.predict_proba(X_val)

# Micro-average ROC
fpr_micro, tpr_micro, _ = roc_curve(y_val.ravel(), y_score.ravel())
micro_auc = auc(fpr_micro, tpr_micro)

# Macro-average ROC
fpr_dict = {}
tpr_dict = {}
roc_auc_dict = {}
for i in range(len(classes)):
    fpr_dict[i], tpr_dict[i], _ = roc_curve(y_val[:, i], y_score[:, i])
    roc_auc_dict[i] = auc(fpr_dict[i], tpr_dict[i])
all_fpr = np.unique(np.concatenate([fpr_dict[i] for i in range(len(classes))]))
mean_tpr = np.zeros_like(all_fpr)
for i in range(len(classes)):
    mean_tpr += np.interp(all_fpr, fpr_dict[i], tpr_dict[i])
mean_tpr /= len(classes)
macro_auc = auc(all_fpr, mean_tpr)

plt.figure(figsize=(7, 5))
plt.plot(fpr_micro, tpr_micro, label=f"Micro-average ROC (AUC = {micro_auc:.3f})", color="blue")
plt.plot(all_fpr, mean_tpr, label=f"Macro-average ROC (AUC = {macro_auc:.3f})", color="green")
plt.plot([0, 1], [0, 1], "k--", lw=1)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve (One-vs-Rest, Random Forest)")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "03-roc-curve.png"), dpi=150)
plt.close()

# 4) Confusion Matrix (Random Forest)
y_pred = rf.predict(X_test)
cm = confusion_matrix(y_test, y_pred, labels=classes)
fig, ax = plt.subplots(figsize=(5, 4))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
disp.plot(ax=ax, cmap=plt.cm.Blues, colorbar=False)
plt.title("Confusion Matrix - Random Forest")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig(os.path.join(IMG_DIR, "04-confusion-matrix.png"), dpi=150)
plt.close()

print("Saved figures:")
for name in [
    "01-class-balance.png",
    "02-feature-importance.png",
    "03-roc-curve.png",
    "04-confusion-matrix.png",
]:
    print(name)
