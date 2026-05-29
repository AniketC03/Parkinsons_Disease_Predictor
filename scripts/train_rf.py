"""
train_rf.py
Train a Random Forest classifier on all datasets.
No scaling applied (tree-based model).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.ensemble import RandomForestClassifier
from utils import load_dataset, preprocess, evaluate, save_experiment, list_datasets

MODEL_NAME = "RandomForest"


def train_random_forest(dataset_name, filepath):
    print(f"\n[RandomForest] Training on: {dataset_name}")
    df = load_dataset(filepath)

    # No scaler for Random Forest
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(
        df, use_scaler=False
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    save_experiment(model, scaler, MODEL_NAME, dataset_name, metrics, X_test, y_test)
    return metrics


def main():
    print("=" * 50)
    print("  TRAINING: Random Forest on all datasets")
    print("=" * 50)
    for dataset_name, filepath in list_datasets():
        train_random_forest(dataset_name, filepath)
    print("\n[RandomForest] Done.\n")


if __name__ == "__main__":
    main()
