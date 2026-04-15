import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import ttest_rel

# Load dataset
df = pd.read_csv("sample_data.csv")

# Features and labels
X = df.drop(columns=["label"])
y = df["label"]

# Cross-validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Models
rf = RandomForestClassifier(n_estimators=100, random_state=42)
svm = SVC()
lr = LogisticRegression(max_iter=500)

# CV scores
rf_scores = cross_val_score(rf, X, y, cv=cv)
svm_scores = cross_val_score(svm, X, y, cv=cv)
lr_scores = cross_val_score(lr, X, y, cv=cv)

print("Random Forest CV:", rf_scores)
print("SVM CV:", svm_scores)
print("Logistic Regression CV:", lr_scores)

# Paired t-tests
t_rf_svm, p_rf_svm = ttest_rel(rf_scores, svm_scores)
t_rf_lr, p_rf_lr = ttest_rel(rf_scores, lr_scores)

print("\nPaired T-Test Results:")
print("RF vs SVM p-value:", p_rf_svm)
print("RF vs LR p-value:", p_rf_lr)