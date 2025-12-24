import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, \
    f1_score, roc_auc_score
import datetime as dt
import time
import os
from sklearn.model_selection import train_test_split


def calculate_distance(lat1, lon1, lat2, lon2):
    return np.sqrt((lat1 - lat2) * 2 + (lon1 - lon2) * 2) * 111


def preprocess_fraud_data(df):
    df['trans_date_trans_time'] = pd.to_datetime(df['trans_date_trans_time'])
    df['dob'] = pd.to_datetime(df['dob'])
    df['age'] = (df['trans_date_trans_time'] - df['dob']).dt.days // 365
    df['hour'] = df['trans_date_trans_time'].dt.hour
    df['day_of_week'] = df['trans_date_trans_time'].dt.dayofweek
    df['distance'] = calculate_distance(df['lat'], df['long'], df['merch_lat'], df['merch_long'])

    cols_to_drop = ['Unnamed: 0', 'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last',
                    'street', 'city', 'state', 'zip', 'lat', 'long', 'job', 'dob', 'trans_num',
                    'unix_time', 'merch_lat', 'merch_long']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    return df


def print_metrics(y_true, y_pred, y_prob, train_time, infer_time):
    print(f"Accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"Recall: {recall_score(y_true, y_pred):.4f}")
    print(f"F1-Score: {f1_score(y_true, y_pred):.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_true, y_prob):.4f}")
    print(f"Training Time: {train_time:.4f}s")
    print(f"Inference Time: {infer_time:.4f}s")


def run_dataset_1():
    print("\n--- Running SVM on Dataset 1 (Credit Card Fraud Detection) ---")
    train_path = 'data/dataset 1/creditcard_train.csv'
    test_path = 'data/dataset 1/creditcard_test.csv'
    full_path = 'data/dataset 1/creditcard.csv'

    if os.path.exists(train_path) and os.path.exists(test_path):
        print("Using pre-split train and test files for Dataset 1...")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        X_train = train_df.drop('Class', axis=1)
        y_train = train_df['Class']
        X_test = test_df.drop('Class', axis=1)
        y_test = test_df['Class']
    elif os.path.exists(full_path):
        print("Pre-split files not found. Splitting full dataset...")
        df = pd.read_csv(full_path)
        X = df.drop('Class', axis=1)
        y = df['Class']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    else:
        print(f"Error: Dataset 1 not found at {full_path}")
        return

    scaler = StandardScaler()
    clf = LinearSVC(class_weight='balanced', random_state=42, max_iter=2000, dual=False)

    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', clf)
    ])

    print("Training model on Dataset 1...")
    start_train = time.time()
    pipeline.fit(X_train, y_train)
    end_train = time.time()

    start_infer = time.time()
    y_pred = pipeline.predict(X_test)
    end_infer = time.time()

    # LinearSVC doesn't have predict_proba by default, but we can use decision_function
    y_prob = pipeline.decision_function(X_test)

    print("\nMetrics for Dataset 1:")
    print_metrics(y_test, y_pred, y_prob, end_train - start_train, end_infer - start_infer)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


def run_dataset_2():
    print("\n--- Running SVM on Dataset 2 (Simulated Fraud Data) ---")
    train_path = 'data/dataset 2/fraudTrain.csv'
    test_path = 'data/dataset 2/fraudTest.csv'

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("Dataset 2 files not found.")
        return

    print("Loading and preprocessing Dataset 2...")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    train_df = preprocess_fraud_data(train_df)
    test_df = preprocess_fraud_data(test_df)

    X_train = train_df.drop('is_fraud', axis=1)
    y_train = train_df['is_fraud']
    X_test = test_df.drop('is_fraud', axis=1)
    y_test = test_df['is_fraud']

    categorical_cols = ['category', 'gender']
    numerical_cols = ['amt', 'city_pop', 'age', 'hour', 'day_of_week', 'distance']

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_cols),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_cols)
        ])

    clf = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', LinearSVC(class_weight='balanced', random_state=42, max_iter=1000, dual=False))
    ])

    print("Training model on Dataset 2 (this may take a few minutes)...")
    start_train = time.time()
    clf.fit(X_train, y_train)
    end_train = time.time()

    print("Evaluating model...")
    start_infer = time.time()
    y_pred = clf.predict(X_test)
    end_infer = time.time()

    y_prob = clf.decision_function(X_test)

    print("\nMetrics for Dataset 2:")
    print_metrics(y_test, y_pred, y_prob, end_train - start_train, end_infer - start_infer)
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, y_pred))


if _name_ == "_main_":
    run_dataset_1()
    run_dataset_2()