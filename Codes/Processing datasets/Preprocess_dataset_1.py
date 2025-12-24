# ==============================
# Credit Card Fraud Preprocessing DATASET_1
# ==============================

import pandas as pd
import os
from sklearn.preprocessing import StandardScaler

# --------------------------------------------------
# File paths
# --------------------------------------------------
TRAIN_IN = "dataset_1/creditcard_train.csv"
TEST_IN = "dataset_1/creditcard_test.csv"

PROCESSED_DIR = "processed_creditcard/"
os.makedirs(PROCESSED_DIR, exist_ok=True)

# --------------------------------------------------
# Load CSVs
# --------------------------------------------------
df_train = pd.read_csv(TRAIN_IN)
df_test = pd.read_csv(TEST_IN)

print("Loaded pre-split train/test data")

# --------------------------------------------------
# Quick quality check
# --------------------------------------------------
print("Train missing values:\n", df_train.isnull().sum())
print("Test missing values:\n", df_test.isnull().sum())

# --------------------------------------------------
# Separate features and target
# --------------------------------------------------
X_train = df_train.drop(columns=["Class"])
y_train = df_train["Class"]

X_test = df_test.drop(columns=["Class"])
y_test = df_test["Class"]

# --------------------------------------------------
# Feature scaling
# --------------------------------------------------
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# --------------------------------------------------
# Reconstruct scaled dataframes
# --------------------------------------------------
train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
train_scaled["Class"] = y_train.values

test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
test_scaled["Class"] = y_test.values

# --------------------------------------------------
# Export preprocessed files
# --------------------------------------------------
TRAIN_OUT = os.path.join(PROCESSED_DIR, "creditcard_train_preprocessed.csv")
TEST_OUT = os.path.join(PROCESSED_DIR, "creditcard_test_preprocessed.csv")

train_scaled.to_csv(TRAIN_OUT, index=False)
test_scaled.to_csv(TEST_OUT, index=False)

print("Preprocessing completed.")
print("Saved scaled train:", TRAIN_OUT)
print("Saved scaled test:", TEST_OUT)
print("Train shape:", train_scaled.shape)
print("Test shape:", test_scaled.shape)
