"""
train_all.py
Master training script: runs all 3 models on all 3 datasets (9 experiments total),
then automatically selects the best model by F1 score.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
from train_rf       import train_random_forest
from train_logistic import train_logistic_regression
from train_knn      import train_knn
from select_best    import select_best
from utils          import list_datasets, COMPARISON_CSV

TRAINERS = {
    "RandomForest":      train_random_forest,
    "LogisticRegression": train_logistic_regression,
    "KNN":               train_knn,
}


def main():
    print("\n" + "=" * 60)
    print("  PARKINSONS DISEASE PREDICTION — FULL TRAINING PIPELINE")
    print("=" * 60)

    datasets = list_datasets()
    print(f"\nFound {len(datasets)} dataset(s): {[d for d, _ in datasets]}")
    print(f"Training {len(TRAINERS)} model(s) × {len(datasets)} datasets = "
          f"{len(TRAINERS) * len(datasets)} experiments\n")

    results = []
    for model_name, train_fn in TRAINERS.items():
        for dataset_name, filepath in datasets:
            metrics = train_fn(dataset_name, filepath)
            results.append({
                "Model":   model_name,
                "Dataset": dataset_name,
                **metrics
            })

    # Summary table
    print("\n" + "=" * 60)
    print("  EXPERIMENT SUMMARY")
    print("=" * 60)
    df = pd.DataFrame(results)
    print(df.to_string(index=False))

    # Best model selection
    print()
    select_best()
    print("\n All experiments complete. Check experiments/comparison.csv")
    print(" Best model saved to best_model/")


if __name__ == "__main__":
    main()
