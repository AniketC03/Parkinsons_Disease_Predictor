"""
train_logistic.py
Train a Logistic Regression classifier on all datasets.
Scaling is applied (required for LR convergence and fairness).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.linear_model import LogisticRegression
from utils import load_dataset, preprocess, evaluate, save_experiment, list_datasets

MODEL_NAME = "LogisticRegression"


def train_logistic_regression(dataset_name, filepath):
    print(f"\n[LogisticRegression] Training on: {dataset_name}")
    df = load_dataset(filepath)

    # Scaling IS applied for Logistic Regression
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(
        df, use_scaler=True
    )

    model = LogisticRegression(
        max_iter=1000,
        solver="lbfgs",
        class_weight="balanced",
        random_state=42,
        C=1.0,
    )
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    save_experiment(model, scaler, MODEL_NAME, dataset_name, metrics, X_test, y_test)
    return metrics


def main():
    print("=" * 50)
    print("  TRAINING: Logistic Regression on all datasets")
    print("=" * 50)
    for dataset_name, filepath in list_datasets():
        train_logistic_regression(dataset_name, filepath)
    print("\n[LogisticRegression] Done.\n")


if __name__ == "__main__":
    main()
