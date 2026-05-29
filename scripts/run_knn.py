"""
run_knn.py
══════════
Trains K-Nearest Neighbors on all 3 datasets INDEPENDENTLY.

Results saved to:
  experiments/KNN/<dataset>/model.pkl
  experiments/KNN/<dataset>/scaler.pkl
  experiments/KNN/<dataset>/confusion_matrix.png
  models/KNN/results.csv   ← summary of all 3 datasets

Run:
  cd parkinsons_project/scripts
  python run_knn.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.neighbors import KNeighborsClassifier
from utils import list_datasets, preprocess, evaluate, save_experiment, save_model_summary

MODEL_NAME = "KNN"


def train_one(dataset_name, filepath):
    """Train KNN on a single dataset. Returns metrics dict."""
    import pandas as pd
    df = pd.read_csv(filepath)

    # KNN NEEDS scaling (distance-based — scale changes distances)
    X_tr, X_te, y_tr, y_te, features, scaler = preprocess(df, use_scaler=True)

    model = KNeighborsClassifier(
        n_neighbors = 7,              # odd number avoids ties
        weights     = "distance",     # closer neighbors vote stronger
        metric      = "euclidean",    # standard geometric distance
        n_jobs      = -1,             # use all CPU cores
    )
    model.fit(X_tr, y_tr)

    metrics = evaluate(model, X_te, y_te)
    save_experiment(model, scaler, MODEL_NAME, dataset_name, metrics, X_te, y_te)

    return {"Dataset": dataset_name, **metrics}


def main():
    print()
    print("=" * 55)
    print("  KNN  —  Training on all 3 datasets")
    print("=" * 55)

    results = []
    for dataset_name, filepath in list_datasets():
        print(f"\n  ▸  {dataset_name}")
        results.append(train_one(dataset_name, filepath))

    print("-" * 55)
    save_model_summary(MODEL_NAME, results)
    print()
    print("  Done! To use one of these models in the web app run:")
    print("  python connect_app.py --model KNN --dataset <name>")
    print()


if __name__ == "__main__":
    main()
