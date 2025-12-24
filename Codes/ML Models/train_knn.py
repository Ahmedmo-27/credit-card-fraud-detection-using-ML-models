# ==========================================
# K-Nearest Neighbors - Fraud Detection
# ==========================================

import pandas as pd
import time
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

# -------------------------------
# Dataset paths
# -------------------------------
dataset_1_train_path = r"D:\MIU\Semester 5\Artificial Inteligence\data cleaned\processed_dataset_1\creditcard_train_preprocessed.csv"
dataset_1_test_path = r"D:\MIU\Semester 5\Artificial Inteligence\data cleaned\processed_dataset_1\creditcard_test_preprocessed.csv"
dataset_2_train_path = r"D:\MIU\Semester 5\Artificial Inteligence\data cleaned\processed_dataset_2\fraudTrain_preprocessed.csv"
dataset_2_test_path = r"D:\MIU\Semester 5\Artificial Inteligence\data cleaned\processed_dataset_2\fraudTest_preprocessed.csv"

# -------------------------------
# Function to train and evaluate KNN
# -------------------------------
def run_knn(train_path, test_path, target_col="is_fraud", dataset_name="Dataset"):
    df_train = pd.read_csv(train_path)
    df_test = pd.read_csv(test_path)

    X_train = df_train.drop(columns=[target_col])
    y_train = df_train[target_col]

    X_test = df_test.drop(columns=[target_col])
    y_test = df_test[target_col]

    knn = KNeighborsClassifier(
        n_neighbors=5,
        weights="distance",
        metric="minkowski",
        algorithm="ball_tree",
        n_jobs=-1
    )

    start_train = time.time()
    knn.fit(X_train, y_train)
    end_train = time.time()
    training_time = end_train - start_train

    start_infer = time.time()
    y_pred = knn.predict(X_test)
    y_proba = knn.predict_proba(X_test)[:, 1]
    end_infer = time.time()
    inference_time = end_infer - start_infer

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)

    results = {
        "Dataset": dataset_name,
        "Training Time": training_time,
        "Inference Time": inference_time,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1-Score": f1,
        "ROC-AUC": roc_auc,
        "Confusion Matrix": confusion_matrix(y_test, y_pred),
        "Classification Report": classification_report(y_test, y_pred)
    }

    # Print results for this dataset
    print(f"\n=== KNN Results for {dataset_name} ===")
    print(f"Training Time: {training_time:.2f} seconds")
    print(f"Inference Time: {inference_time:.2f} seconds\n")
    print(f"Accuracy: {accuracy:.5f}")
    print(f"Precision: {precision:.5f}")
    print(f"Recall: {recall:.5f}")
    print(f"F1-Score: {f1:.5f}")
    print(f"ROC-AUC Score: {roc_auc:.5f}\n")
    print("Confusion Matrix:\n", results["Confusion Matrix"])
    print("\nClassification Report:\n", results["Classification Report"])

    return results

# -------------------------------
# Run KNN on both datasets
# -------------------------------

# Dataset 1 uses 'Class' as target column
results_dataset_1 = run_knn(dataset_1_train_path, dataset_1_test_path, target_col="Class", dataset_name="Dataset 1")

# Dataset 2 uses 'is_fraud' as target column
results_dataset_2 = run_knn(dataset_2_train_path, dataset_2_test_path, target_col="is_fraud", dataset_name="Dataset 2")

combined_results = {
    "Dataset 1": results_dataset_1,
    "Dataset 2": results_dataset_2
}

def calculate_average_metrics(combined_results):
    metrics = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC", "Training Time", "Inference Time"]
    sums = {metric: 0 for metric in metrics}
    n = len(combined_results)

    for dataset in combined_results.values():
        for metric in metrics:
            sums[metric] += dataset[metric]

    averages = {metric: sums[metric] / n for metric in metrics}
    return averages

average_metrics = calculate_average_metrics(combined_results)
print("\n=== Average Metrics Across Datasets ===")
for metric, value in average_metrics.items():
    print(f"{metric}: {value:.6f}")