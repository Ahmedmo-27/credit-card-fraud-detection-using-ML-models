# ==============================
# XGBoost ML Model for Fraud Detection
# ==============================

import pandas as pd
import time
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from xgboost import XGBClassifier

# -----------------------------
# Dataset paths (preprocessed)
# -----------------------------
dataset1_train = "C:/Users/pc/Downloads/processed_creditcard/creditcard_train_preprocessed.csv"
dataset1_test = "C:/Users/pc/Downloads/processed_creditcard/creditcard_test_preprocessed.csv"

dataset2_train = "C:/Users/pc/Downloads/processed_data/fraudTrain_preprocessed.csv"
dataset2_test = "C:/Users/pc/Downloads/processed_data/fraudTest_preprocessed.csv"


# -----------------------------
# Load dataset function
# -----------------------------
def load_dataset(train_path, test_path, target_col):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    return X_train, X_test, y_train, y_test


# -----------------------------
# Training & Evaluation function
# -----------------------------
def train_evaluate(X_train, X_test, y_train, y_test):
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

    # Measure training time
    start_train = time.time()
    model.fit(X_train, y_train)
    end_train = time.time()

    # Measure inference time
    start_infer = time.time()
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    end_infer = time.time()

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    results = {
        "Accuracy": acc,
        "Precision": prec,
        "Recall": rec,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "Training Time (s)": end_train - start_train,
        "Inference Time (s)": end_infer - start_infer
    }

    return results


# -----------------------------
# Dataset 1: Credit Card Fraud
# -----------------------------
print("\n=== Dataset 1: Credit Card Fraud ===")
X_train1, X_test1, y_train1, y_test1 = load_dataset(dataset1_train, dataset1_test, "Class")
results1 = train_evaluate(X_train1, X_test1, y_train1, y_test1)
for k, v in results1.items():
    print(f"{k}: {v:.4f}")

# -----------------------------
# Dataset 2: Fraud Detection
# -----------------------------
print("\n=== Dataset 2: Fraud Detection ===")
X_train2, X_test2, y_train2, y_test2 = load_dataset(dataset2_train, dataset2_test, "is_fraud")
results2 = train_evaluate(X_train2, X_test2, y_train2, y_test2)
for k, v in results2.items():
    print(f"{k}: {v:.4f}")