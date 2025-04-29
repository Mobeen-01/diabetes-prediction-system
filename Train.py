# # The LightGBM (LGBMClassifier) model is used for training and prediction in the provided code. Various oversampling and 
# # undersampling methods are applied to handle class imbalance, including techniques like SMOTE and AllKNN. In the final 
# # output, AllKNN was selected as the undersampling method, which significantly improved model performance, as shown by the 
# # higher metrics in comparison to the original dataset.


import os
import sys
import warnings
import logging

# Suppress LightGBM C-level warnings printed to stderr
sys.stderr = open(os.devnull, 'w')

# Regular Python warnings/logging suppression
# warnings.filterwarnings('ignore')
# logging.getLogger('lightgbm').setLevel(logging.ERROR)


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from imblearn.over_sampling import SMOTE, ADASYN, SVMSMOTE, RandomOverSampler, BorderlineSMOTE, KMeansSMOTE
from imblearn.under_sampling import EditedNearestNeighbours, AllKNN, InstanceHardnessThreshold, NearMiss, NeighbourhoodCleaningRule, OneSidedSelection, RandomUnderSampler, TomekLinks
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from lightgbm import LGBMClassifier
from prettytable import PrettyTable

# Load Data
csv_file_path = "./data/diabetes.csv"
data = pd.read_csv(csv_file_path)

# Data Overview
tanitim = data.describe()
print(tanitim)

# Features and Labels
X = data.drop(['Outcome'], axis=1).values
y = data['Outcome'].values

# Oversampling and Undersampling techniques
oversamplers = [SMOTE(random_state=42), KMeansSMOTE(random_state=42), ADASYN(random_state=42), SVMSMOTE(random_state=42),
                RandomOverSampler(random_state=42), BorderlineSMOTE(random_state=42)]
undersamplers = [EditedNearestNeighbours(), AllKNN(), InstanceHardnessThreshold(random_state=42),
                 NearMiss(), NeighbourhoodCleaningRule(), OneSidedSelection(random_state=42),
                 RandomUnderSampler(random_state=42), TomekLinks()]

# ---- Function to Train and Compare LightGBM models ----
def plot_metrics_comparison_lightgbm(X, y, samplers):
    metrics_names = ['Accuracy', 'Precision', 'Recall', 'F1 Score']
    metrics_values = {metric: [] for metric in metrics_names}
    sampler_names = ['Original']

    # Original Dataset
    X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42)
    lgbm_model_orig = LGBMClassifier(random_state=42, force_col_wise=True, num_leaves=42, min_child_samples=60)
    lgbm_model_orig.fit(X_train_orig, y_train_orig)
    y_pred_orig = lgbm_model_orig.predict(X_test_orig)

    # Metrics Original
    metrics_values['Accuracy'].append(accuracy_score(y_test_orig, y_pred_orig))
    metrics_values['Precision'].append(precision_score(y_test_orig, y_pred_orig))
    metrics_values['Recall'].append(recall_score(y_test_orig, y_pred_orig))
    metrics_values['F1 Score'].append(f1_score(y_test_orig, y_pred_orig))

    # Sampler Models
    for sampler in samplers:
        X_resampled, y_resampled = sampler.fit_resample(X, y)
        X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

        lgbm_model = LGBMClassifier(random_state=42, force_col_wise=True, verbosity=-1)
        lgbm_model.fit(X_train, y_train)
        y_pred = lgbm_model.predict(X_test)

        metrics_values['Accuracy'].append(accuracy_score(y_test, y_pred))
        metrics_values['Precision'].append(precision_score(y_test, y_pred))
        metrics_values['Recall'].append(recall_score(y_test, y_pred))
        metrics_values['F1 Score'].append(f1_score(y_test, y_pred))

        sampler_names.append(sampler.__class__.__name__)

    # PrettyTable: Only Original and AllKNN
    table = PrettyTable()
    table.field_names = ['Sampler', 'Accuracy', 'Precision', 'Recall', 'F1 Score']

    for name, acc, prec, rec, f1 in zip(sampler_names,
                                        metrics_values['Accuracy'],
                                        metrics_values['Precision'],
                                        metrics_values['Recall'],
                                        metrics_values['F1 Score']):
        if name == 'Original' or name == 'AllKNN':
            table.add_row([name, f"{acc:.4f}", f"{prec:.4f}", f"{rec:.4f}", f"{f1:.4f}"])

    print(table)

# Plot and show metrics
plot_metrics_comparison_lightgbm(X, y, oversamplers + undersamplers)

# ---- Train Final Model on AllKNN and Save ----
sampler = AllKNN()
X_resampled, y_resampled = sampler.fit_resample(X, y)

final_model = LGBMClassifier(random_state=42, force_col_wise=True, verbosity=-1)
final_model.fit(X_resampled, y_resampled)

joblib.dump(final_model, './lgbm_model.pkl')
print("✅ Final Model (AllKNN) saved successfully!")

# ---- Extra: Confusion Matrix and ROC Curve for both models ----

def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"Confusion Matrix: {title}")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.show()

def plot_roc_curve(y_true, y_probs, title):
    fpr, tpr, thresholds = roc_curve(y_true, y_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(6,4))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (area = {roc_auc:.2f})')
    plt.plot([0,1],[0,1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve: {title}')
    plt.legend(loc="lower right")
    plt.show()

# Original dataset model
X_train_orig, X_test_orig, y_train_orig, y_test_orig = train_test_split(X, y, test_size=0.2, random_state=42)
model_orig = LGBMClassifier(random_state=42, force_col_wise=True, num_leaves=42, min_child_samples=60)
model_orig.fit(X_train_orig, y_train_orig)
y_pred_orig = model_orig.predict(X_test_orig)
y_pred_proba_orig = model_orig.predict_proba(X_test_orig)[:, 1]

plot_confusion_matrix(y_test_orig, y_pred_orig, "Original Dataset")
plot_roc_curve(y_test_orig, y_pred_proba_orig, "Original Dataset")

# AllKNN resampled model
X_train_resampled, X_test_resampled, y_train_resampled, y_test_resampled = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
model_resampled = LGBMClassifier(random_state=42, force_col_wise=True, verbosity=-1)
model_resampled.fit(X_train_resampled, y_train_resampled)
y_pred_resampled = model_resampled.predict(X_test_resampled)
y_pred_proba_resampled = model_resampled.predict_proba(X_test_resampled)[:, 1]

plot_confusion_matrix(y_test_resampled, y_pred_resampled, "AllKNN Resampled")
plot_roc_curve(y_test_resampled, y_pred_proba_resampled, "AllKNN Resampled")
