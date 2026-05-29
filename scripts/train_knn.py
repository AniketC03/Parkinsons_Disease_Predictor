"""
train_knn.py
Train a K-Nearest Neighbors classifier on all datasets.
Scaling is applied (KNN is distance-based, scaling is essential).
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.neighbors import KNeighborsClassifier
from utils import load_dataset, preprocess, evaluate, save_experiment, list_datasets

MODEL_NAME = "KNN"


def train_knn(dataset_name, filepath):
    print(f"\n[KNN] Training on: {dataset_name}")
    df = load_dataset(filepath)

    # Scaling IS applied for KNN (distance-based algorithm)
    X_train, X_test, y_train, y_test, feature_names, scaler = preprocess(
        df, use_scaler=True
    )

    model = KNeighborsClassifier(
        n_neighbors=7,
        weights="distance",
        metric="euclidean",
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    metrics = evaluate(model, X_test, y_test)
    save_experiment(model, scaler, MODEL_NAME, dataset_name, metrics, X_test, y_test)
    return metrics


def main():
    print("=" * 50)
    print("  TRAINING: KNN on all datasets")
    print("=" * 50)
    for dataset_name, filepath in list_datasets():
        train_knn(dataset_name, filepath)
    print("\n[KNN] Done.\n")


if __name__ == "__main__":
    main()
