# Credit Card Fraud Detection Using Machine Learning

A comprehensive machine learning project that implements and compares multiple algorithms for detecting fraudulent credit card transactions. This project evaluates six different machine learning models on two distinct fraud detection datasets to identify the most effective approach for credit card fraud detection.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Datasets](#datasets)
- [Project Structure](#project-structure)
- [Models Implemented](#models-implemented)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Visualizations](#visualizations)
- [Authors](#authors)
- [References](#references)
- [License](#license)

## 🎯 Overview

Credit card fraud is a significant concern in financial transactions, causing billions in losses annually. This project addresses this problem by developing and comparing various machine learning models capable of accurately identifying fraudulent transactions in real-time. The project evaluates multiple algorithms including Logistic Regression, Random Forest, XGBoost, Support Vector Machine (SVM), Decision Tree, and K-Nearest Neighbors (KNN) on two comprehensive fraud detection datasets.

### Key Objectives

- Implement and evaluate multiple machine learning algorithms for fraud detection
- Handle class imbalance issues common in fraud detection datasets
- Preprocess and engineer features from raw transaction data
- Compare model performance using comprehensive evaluation metrics
- Generate visualizations to understand fraud patterns and model behavior

## ✨ Features

- **Multiple ML Models**: Implementation of 6 different machine learning algorithms
- **Two Dataset Support**: Evaluation on two distinct fraud detection datasets
- **Comprehensive Preprocessing**: Feature engineering, scaling, and data cleaning pipelines
- **Class Imbalance Handling**: Techniques to address imbalanced datasets
- **Performance Metrics**: Evaluation using accuracy, precision, recall, F1-score, and ROC-AUC
- **Visualizations**: Extensive visualization of data patterns and model performance
- **Reproducible Results**: Standardized preprocessing and evaluation pipelines

## 📊 Datasets

### Dataset 1: Credit Card Fraud Detection
- **Source**: Credit Card Fraud Detection dataset
- **Features**: 
  - 28 anonymized features (V1-V28) derived from PCA
  - Time and Amount features
  - Target variable: `Class` (0 = Legitimate, 1 = Fraudulent)
- **Characteristics**: Highly imbalanced dataset with significantly fewer fraud cases

### Dataset 2: Fraud Transaction Dataset
- **Source**: Simulated fraud transaction dataset
- **Features**:
  - Transaction details (amount, category, merchant information)
  - Customer demographics (age, gender, location)
  - Temporal features (transaction time, day, month)
  - Geographic features (city, state, distance calculations)
  - Target variable: `is_fraud` (0 = Legitimate, 1 = Fraudulent)

## 📁 Project Structure

```
Project/
│
├── Codes/
│   ├── Processing datasets/
│   │   ├── Preprocess_dataset_1.py      # Preprocessing pipeline for Dataset 1
│   │   ├── Preprocess_dataset_2.py      # Preprocessing pipeline for Dataset 2
│   │   └── split.py                     # Dataset splitting utility
│   │
│   └── ML Models/
│       ├── train_logistic_regression.py # Logistic Regression implementation
│       ├── train_random_forest.py       # Random Forest implementation
│       ├── train_XGBoost.py             # XGBoost implementation
│       ├── train_svm.py                 # Support Vector Machine implementation
│       ├── train_decision_tree.py       # Decision Tree implementation
│       └── train_knn.py                 # K-Nearest Neighbors implementation
│
├── dataset_1/                           # Raw Dataset 1 files
│   ├── creditcard.csv
│   ├── creditcard_train.csv
│   └── creditcard_test.csv
│
├── dataset_2/                           # Raw Dataset 2 files
│   ├── fraudTrain.csv
│   └── fraudTest.csv
│
├── processed_dataset_1/                 # Preprocessed Dataset 1 files
│   ├── creditcard_train_preprocessed.csv
│   └── creditcard_test_preprocessed.csv
│
├── processed_dataset_2/                 # Preprocessed Dataset 2 files
│   ├── fraudTrain_preprocessed.csv
│   └── fraudTest_preprocessed.csv
│
├── Visualizations/                      # Generated visualizations
│   ├── ROC curves for each model
│   ├── Precision-Recall curves
│   ├── Correlation heatmaps
│   ├── Fraud distribution plots
│   └── Model comparison charts
│
├── References/                          # Research papers and references
│   └── [Multiple PDF research papers]
│
├── Team_SW_08_F25_Ahmed_Mostafa.pdf    # Project documentation
├── requirements.txt                     # Python dependencies
└── README.md                           # This file
```

## 🤖 Models Implemented

### 1. Logistic Regression
- **Type**: Linear classification model
- **Features**: Balanced class weights for handling imbalanced data
- **Use Case**: Baseline model for comparison

### 2. Random Forest
- **Type**: Ensemble learning method
- **Configuration**: 200 estimators, balanced class weights
- **Strengths**: Robust to overfitting, feature importance analysis

### 3. XGBoost (Extreme Gradient Boosting)
- **Type**: Gradient boosting framework
- **Configuration**: Optimized for binary classification
- **Strengths**: High performance, handles imbalanced data well

### 4. Support Vector Machine (SVM)
- **Type**: Linear SVC with balanced class weights
- **Configuration**: Optimized for large-scale classification
- **Strengths**: Effective for high-dimensional data

### 5. Decision Tree
- **Type**: Tree-based classifier
- **Configuration**: Max depth=10, balanced class weights
- **Strengths**: Interpretable, handles non-linear relationships

### 6. K-Nearest Neighbors (KNN)
- **Type**: Instance-based learning
- **Configuration**: k=5, distance-weighted
- **Strengths**: Non-parametric, effective for local patterns

## 🚀 Installation

### Prerequisites

- Python 3.7 or higher
- pip (Python package installer)

### Step 1: Clone the Repository

```bash
git clone <repository-url>
cd Project
```

### Step 2: Create a Virtual Environment (Recommended)

```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

The project requires the following Python packages (automatically installed via `requirements.txt`):

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning algorithms and utilities
- `xgboost` - Gradient boosting framework
- `matplotlib` - Data visualization (for visualizations)
- `seaborn` - Statistical data visualization (for visualizations)

## 💻 Usage

### Step 1: Preprocess the Datasets

Before training models, preprocess the raw datasets:

```bash
# Preprocess Dataset 1
cd "Codes/Processing datasets"
python Preprocess_dataset_1.py

# Preprocess Dataset 2
python Preprocess_dataset_2.py
```

### Step 2: Train and Evaluate Models

Navigate to the ML Models directory and run individual model training scripts:

```bash
cd "../ML Models"

# Train Logistic Regression
python train_logistic_regression.py

# Train Random Forest
python train_random_forest.py

# Train XGBoost
python train_XGBoost.py

# Train SVM
python train_svm.py

# Train Decision Tree
python train_decision_tree.py

# Train KNN
python train_knn.py
```

### Step 3: View Results

Each model training script will output:
- **Training Time**: Time taken to train the model
- **Inference Time**: Time taken for predictions
- **Accuracy**: Overall classification accuracy
- **Precision**: Ratio of true positives to predicted positives
- **Recall**: Ratio of true positives to actual positives
- **F1-Score**: Harmonic mean of precision and recall
- **ROC-AUC Score**: Area under the ROC curve
- **Confusion Matrix**: Detailed classification breakdown
- **Classification Report**: Per-class performance metrics

## 📈 Results

The project evaluates all models on both datasets and provides comprehensive performance metrics. Key evaluation metrics include:

- **Accuracy**: Overall correctness of predictions
- **Precision**: Ability to correctly identify fraudulent transactions
- **Recall**: Ability to find all fraudulent transactions
- **F1-Score**: Balanced metric considering both precision and recall
- **ROC-AUC**: Model's ability to distinguish between classes

### Model Comparison

Results are averaged across both datasets to provide a comprehensive comparison. Each model is evaluated on:
- Training time efficiency
- Inference speed
- Classification performance metrics
- Ability to handle class imbalance

## 📊 Visualizations

The project includes comprehensive visualizations located in the `Visualizations/` directory:

- **ROC Curves**: Receiver Operating Characteristic curves for each model
- **Precision-Recall Curves**: PR curves showing precision-recall trade-offs
- **Correlation Heatmaps**: Feature correlation analysis
- **Fraud Distribution Plots**: Visualization of fraud patterns
- **Model Comparison Charts**: Side-by-side performance comparisons
- **Feature Importance**: Visualizations of most important features
- **Temporal Analysis**: Fraud patterns over time

## 👥 Authors

- **Ahmed Mostafa** - Team SW_08_F25

**Institution**: MIU (Misr International University)  
**Course**: Artificial Intelligence - Semester 5  
**Project**: Credit Card Fraud Detection

## 📚 References

This project is based on extensive research in fraud detection and machine learning. Key references are available in the `References/` directory, including:

- Credit Card Fraud Detection using Machine Learning
- Ensemble Methods for Fraud Detection
- Handling Class Imbalance in Fraud Detection
- Feature Engineering for Transaction Data
- And multiple other research papers

## 🔒 License

This project is developed for educational purposes as part of a university course. Please refer to the original dataset licenses for commercial usage restrictions.

## 🤝 Contributing

This is an academic project. For suggestions or improvements, please open an issue or contact the authors.

## 📝 Notes

- **Class Imbalance**: Both datasets exhibit significant class imbalance. All models use techniques like balanced class weights to address this issue.
- **Data Privacy**: Dataset 1 features are anonymized using PCA to protect sensitive information.
- **Reproducibility**: Random seeds are set (random_state=42) to ensure reproducible results.
- **Performance**: Results may vary based on hardware configuration and dataset versions.

## 🐛 Troubleshooting

### Common Issues

1. **File Not Found Errors**: Ensure you're running scripts from the correct directory and that dataset files are in the expected locations.

2. **Memory Errors**: For large datasets, consider using a machine with more RAM or processing data in chunks.

3. **Import Errors**: Make sure all dependencies are installed using `pip install -r requirements.txt`.

4. **Path Issues**: Update file paths in scripts if your directory structure differs from the standard structure.

## 📞 Contact

For questions or issues related to this project, please contact:
- **Email**: [Contact information]
- **Institution**: MIU - Misr International University

---

**Last Updated**: 2025

