# ==============================
# Credit Card Fraud Dataset Split
# ==============================

import pandas as pd
from sklearn.model_selection import train_test_split

# --------------------------------------------------
# Load raw dataset
# --------------------------------------------------
DATA_PATH = "data/creditcard.csv"

df = pd.read_csv(DATA_PATH)
print("Raw dataset loaded. Shape:", df.shape)

# --------------------------------------------------
# Separate features and target
# --------------------------------------------------
X = df.drop(columns=["Class"])
y = df["Class"]

# --------------------------------------------------
# Train-test split (stratified — crucial for imbalance)
# --------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.25,
    random_state=42,
    stratify=y
)

# --------------------------------------------------
# Export CSVs for preprocessing
# --------------------------------------------------
train = pd.concat([X_train, y_train], axis=1)
test = pd.concat([X_test, y_test], axis=1)

train.to_csv("data/creditcard_train.csv", index=False)
test.to_csv("data/creditcard_test.csv", index=False)

print("Train and test splits saved:")
print("Train shape:", train.shape)
print("Test shape:", test.shape)
print("Fraud ratio (train):", y_train.mean())
