# ==============================
# Fraud Dataset Preprocessing DATASET_2
# ==============================

import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import StandardScaler, OrdinalEncoder

# --------------------------------------------------
# Dataset paths
# --------------------------------------------------
RAW_DATA_DIR = "data"
PROCESSED_DATA_DIR = "processed_data"

os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)

TRAIN_FILE = os.path.join(RAW_DATA_DIR, "fraudTrain.csv")
TEST_FILE = os.path.join(RAW_DATA_DIR, "fraudTest.csv")

# --------------------------------------------------
# Load datasets
# --------------------------------------------------
df_train = pd.read_csv(TRAIN_FILE)
df_test = pd.read_csv(TEST_FILE)

print("Datasets loaded successfully.")

# --------------------------------------------------
# Data quality check
# --------------------------------------------------
def data_quality_check(df, name):
    print(f"\n--- {name} Dataset ---")
    print("Shape:", df.shape)
    print("Missing values:\n", df.isnull().sum())
    print("Duplicate rows:", df.duplicated().sum())

data_quality_check(df_train, "Train")
data_quality_check(df_test, "Test")

# --------------------------------------------------
# Remove duplicates
# --------------------------------------------------
df_train = df_train.drop_duplicates().reset_index(drop=True)
df_test = df_test.drop_duplicates().reset_index(drop=True)

# --------------------------------------------------
# Handle missing values
# --------------------------------------------------
num_cols = df_train.select_dtypes(include=["int64", "float64"]).columns
cat_cols = df_train.select_dtypes(include=["object"]).columns

# Numerical → median
for col in num_cols:
    median = df_train[col].median()
    df_train[col] = df_train[col].fillna(median)
    df_test[col] = df_test[col].fillna(median)

# Categorical → mode
for col in cat_cols:
    mode = df_train[col].mode()[0]
    df_train[col] = df_train[col].fillna(mode)
    df_test[col] = df_test[col].fillna(mode)

# --------------------------------------------------
# Date feature engineering
# --------------------------------------------------
df_train["trans_date_trans_time"] = pd.to_datetime(df_train["trans_date_trans_time"])
df_test["trans_date_trans_time"] = pd.to_datetime(df_test["trans_date_trans_time"])

for df in [df_train, df_test]:
    df["trans_hour"] = df["trans_date_trans_time"].dt.hour
    df["trans_day"] = df["trans_date_trans_time"].dt.day
    df["trans_month"] = df["trans_date_trans_time"].dt.month
    df["trans_weekday"] = df["trans_date_trans_time"].dt.weekday

df_train.drop(columns=["trans_date_trans_time"], inplace=True)
df_test.drop(columns=["trans_date_trans_time"], inplace=True)

# --------------------------------------------------
# Drop ID / PII columns
# --------------------------------------------------
DROP_COLS = [
    "Unnamed: 0",
    "trans_num",
    "cc_num",
    "first",
    "last",
    "street",
    "zip",
    "dob"
]

df_train.drop(columns=DROP_COLS, inplace=True, errors="ignore")
df_test.drop(columns=DROP_COLS, inplace=True, errors="ignore")

# --------------------------------------------------
# Encode categorical features (robust to unseen labels)
# --------------------------------------------------
cat_cols = df_train.select_dtypes(include=["object"]).columns

encoder = OrdinalEncoder(
    handle_unknown="use_encoded_value",
    unknown_value=-1
)

df_train[cat_cols] = encoder.fit_transform(df_train[cat_cols])
df_test[cat_cols] = encoder.transform(df_test[cat_cols])

# --------------------------------------------------
# Feature scaling
# --------------------------------------------------
scaler = StandardScaler()

num_cols = df_train.select_dtypes(include=["int64", "float64"]).columns
num_cols = num_cols.drop("is_fraud")

df_train[num_cols] = scaler.fit_transform(df_train[num_cols])
df_test[num_cols] = scaler.transform(df_test[num_cols])

# --------------------------------------------------
# Split features & target
# --------------------------------------------------
X_train = df_train.drop(columns=["is_fraud"])
y_train = df_train["is_fraud"]

X_test = df_test.drop(columns=["is_fraud"])
y_test = df_test["is_fraud"]

# --------------------------------------------------
# Export preprocessed datasets ✅
# --------------------------------------------------
TRAIN_OUT = os.path.join(PROCESSED_DATA_DIR, "fraudTrain_preprocessed.csv")
TEST_OUT = os.path.join(PROCESSED_DATA_DIR, "fraudTest_preprocessed.csv")

df_train.to_csv(TRAIN_OUT, index=False)
df_test.to_csv(TEST_OUT, index=False)

print("\nPreprocessing completed successfully.")
print("Saved files:")
print(TRAIN_OUT)
print(TEST_OUT)
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("Fraud ratio (train):", y_train.mean())
