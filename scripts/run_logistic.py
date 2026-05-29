"""
run_logistic.py
═══════════════
Trains Logistic Regression on all 3 datasets INDEPENDENTLY.

Results saved to:
  experiments/LogisticRegression/<dataset>/model.pkl
  experiments/LogisticRegression/<dataset>/scaler.pkl
  experiments/LogisticRegression/<dataset>/confusion_matrix.png
  models/LogisticRegression/results.csv   ← summary of all 3 datasets

Run:
  cd parkinsons_project/scripts
  python run_logistic.py
"""

import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from sklearn.linear_model import LogisticRegression
from utils import list_datasets, preprocess, evaluate, save_experiment, save_model_summary

MODEL_NAME = "LogisticRegression"


def train_one(dataset_name, filepath):
    """Train Logistic Regression on a single dataset. Returns metrics dict."""
    import pandas as pd
    df = pd.read_csv(filepath)

    # Logistic Regression NEEDS scaling (gradient-based optimizer)
    X_tr, X_te, y_tr, y_te, features, scaler = preprocess(df, use_scaler=True)

    model = LogisticRegression(
        max_iter    = 1000,           # enough iterations to converge
        solver      = "lbfgs",        # works well for small-medium data
        C           = 1.0,            # regularization (1.0 = balanced)
        class_weight= "balanced",     # handle unequal PD / non-PD counts
        random_state= 42,
    )
    model.fit(X_tr, y_tr)

    metrics = evaluate(model, X_te, y_te)
    save_experiment(model, scaler, MODEL_NAME, dataset_name, metrics, X_te, y_te)

    return {"Dataset": dataset_name, **metrics}


def main():
    print()
    print("=" * 55)
    print("  LOGISTIC REGRESSION  —  Training on all 3 datasets")
    print("=" * 55)

    results = []
    for dataset_name, filepath in list_datasets():
        print(f"\n  ▸  {dataset_name}")
        results.append(train_one(dataset_name, filepath))

    print("-" * 55)
    save_model_summary(MODEL_NAME, results)
    print()
    print("  Done! To use one of these models in the web app run:")
    print("  python connect_app.py --model LogisticRegression --dataset <name>")
    print()


if __name__ == "__main__":
    main()
