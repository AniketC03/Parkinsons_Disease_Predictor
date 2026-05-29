"""
run_rf.py
═════════
Trains Random Forest on all 3 datasets INDEPENDENTLY.

Results saved to:
  experiments/RandomForest/<dataset>/model.pkl
  experiments/RandomForest/<dataset>/confusion_matrix.png
  models/RandomForest/results.csv   ← summary of all 3 datasets

  NOTE: No scaler.pkl — Random Forest does NOT need feature scaling.

Run:
  cd parkinsons_project/scripts
  python run_rf.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.ensemble import RandomForestClassifier
from utils import list_datasets, preprocess, evaluate, save_experiment, save_model_summary

MODEL_NAME = "RandomForest"


def train_one(dataset_name, filepath):
    """Train Random Forest on a single dataset. Returns metrics dict."""
    import pandas as pd
    df = pd.read_csv(filepath)

    # Random Forest does NOT need scaling (threshold-based, not distance-based)
    X_tr, X_te, y_tr, y_te, features, scaler = preprocess(df, use_scaler=False)

    model = RandomForestClassifier(
        n_estimators  = 200,          # 200 trees → stable majority vote
        max_depth     = None,         # trees grow fully (no depth limit)
        class_weight  = "balanced",   # handle unequal PD / non-PD counts
        random_state  = 42,
        n_jobs        = -1,           # use all CPU cores
    )
    model.fit(X_tr, y_tr)

    metrics = evaluate(model, X_te, y_te)
    save_experiment(model, scaler, MODEL_NAME, dataset_name, metrics, X_te, y_te)

    return {"Dataset": dataset_name, **metrics}


def main():
    print()
    print("=" * 55)
    print("  RANDOM FOREST  —  Training on all 3 datasets")
    print("=" * 55)

    results = []
    for dataset_name, filepath in list_datasets():
        print(f"\n  ▸  {dataset_name}")
        results.append(train_one(dataset_name, filepath))

    print("-" * 55)
    save_model_summary(MODEL_NAME, results)
    print()
    print("  Done! To use one of these models in the web app run:")
    print("  python connect_app.py --model RandomForest --dataset <name>")
    print()


if __name__ == "__main__":
    main()
