"""
utils.py  —  Shared helpers used by all three model runners.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix
)

# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT            = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATASETS_DIR    = os.path.join(ROOT, "datasets")
EXPERIMENTS_DIR = os.path.join(ROOT, "experiments")
BEST_MODEL_DIR  = os.path.join(ROOT, "best_model")
COMPARISON_CSV  = os.path.join(EXPERIMENTS_DIR, "comparison.csv")

# ── Config ────────────────────────────────────────────────────────────────────
TARGET_COLUMNS = ["Diagnosis", "parkinson_status"]
DROP_COLUMNS   = ["PatientID", "DoctorInCharge"]
RANDOM_STATE   = 42
TEST_SIZE      = 0.2


# ─────────────────────────────────────────────────────────────────────────────
# Dataset helpers
# ─────────────────────────────────────────────────────────────────────────────

def load_dataset(filepath):
    """Read a CSV file and return a pandas DataFrame."""
    return pd.read_csv(filepath)


def list_datasets():
    """Return [(dataset_name, filepath), ...] for every CSV in datasets/."""
    out = []
    for fname in sorted(os.listdir(DATASETS_DIR)):
        if fname.endswith(".csv"):
            out.append((fname.replace(".csv", ""),
                        os.path.join(DATASETS_DIR, fname)))
    return out


def detect_target(df):
    """Find and return the name of the target column in df."""
    for col in TARGET_COLUMNS:
        if col in df.columns:
            return col
    raise ValueError(f"No target column found. Columns: {list(df.columns)}")


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess(df, use_scaler=False):
    """
    Clean and split the dataset.
    Returns: X_train, X_test, y_train, y_test, feature_names, scaler (or None)
    """
    df = df.copy()

    # Drop admin columns
    df.drop(columns=[c for c in DROP_COLUMNS if c in df.columns], inplace=True)

    # Separate X and y
    target = detect_target(df)
    y = df[target].values
    X = df.drop(columns=[target])

    # Encode any text columns to numbers
    for col in X.select_dtypes(include="object").columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    feature_names = list(X.columns)
    X = X.values.astype(float)

    # Stratified train/test split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # Optional scaling (Logistic Regression + KNN only)
    scaler = None
    if use_scaler:
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_tr)
        X_te  = scaler.transform(X_te)

    return X_tr, X_te, y_tr, y_te, feature_names, scaler


# ─────────────────────────────────────────────────────────────────────────────
# Evaluation
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, X_te, y_te):
    """Return dict of Accuracy, Precision, Recall, F1."""
    y_pred = model.predict(X_te)
    return {
        "Accuracy":  round(accuracy_score(y_te, y_pred), 4),
        "Precision": round(precision_score(y_te, y_pred, zero_division=0), 4),
        "Recall":    round(recall_score(y_te, y_pred, zero_division=0), 4),
        "F1":        round(f1_score(y_te, y_pred, zero_division=0), 4),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Saving
# ─────────────────────────────────────────────────────────────────────────────

def save_confusion_matrix(model, X_te, y_te, path):
    """Draw and save confusion matrix as PNG."""
    y_pred = model.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["No PD", "PD"],
                yticklabels=["No PD", "PD"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close(fig)


def save_experiment(model, scaler, model_name, dataset_name, metrics, X_te, y_te):
    """Save model, scaler, confusion matrix, and append row to comparison.csv."""
    exp_dir = os.path.join(EXPERIMENTS_DIR, model_name, dataset_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Save model
    joblib.dump(model, os.path.join(exp_dir, "model.pkl"))

    # Save scaler if used
    if scaler is not None:
        joblib.dump(scaler, os.path.join(exp_dir, "scaler.pkl"))

    # Save confusion matrix image
    save_confusion_matrix(model, X_te, y_te,
                          os.path.join(exp_dir, "confusion_matrix.png"))

    # Append result to comparison.csv
    _append_comparison(model_name, dataset_name, metrics)

    print(f"  [SAVED] {model_name}/{dataset_name} → "
          f"Acc={metrics['Accuracy']}  F1={metrics['F1']}")


def _append_comparison(model_name, dataset_name, metrics):
    """Add one row to experiments/comparison.csv (creates file if missing)."""
    os.makedirs(EXPERIMENTS_DIR, exist_ok=True)

    row = {
        "Model":     model_name,
        "Dataset":   dataset_name,
        "Accuracy":  metrics["Accuracy"],
        "Precision": metrics["Precision"],
        "Recall":    metrics["Recall"],
        "F1":        metrics["F1"],
        "IsBest":    False,
    }

    if os.path.exists(COMPARISON_CSV):
        df = pd.read_csv(COMPARISON_CSV)
        # Remove duplicate row for this model+dataset (safe to re-run)
        mask = (df["Model"] == model_name) & (df["Dataset"] == dataset_name)
        df = df[~mask]
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    else:
        df = pd.DataFrame([row])

    df.to_csv(COMPARISON_CSV, index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Best model selection
# ─────────────────────────────────────────────────────────────────────────────

def select_and_save_best():
    """
    Read comparison.csv, find the row with the highest F1,
    copy its artifacts to best_model/, mark IsBest=True in CSV.
    Returns: (model_name, dataset_name, best_row)
    """
    if not os.path.exists(COMPARISON_CSV):
        raise FileNotFoundError(
            "comparison.csv not found. Run train_all.py first."
        )

    df = pd.read_csv(COMPARISON_CSV)
    df["IsBest"] = False

    best_idx     = df["F1"].idxmax()
    best_row     = df.loc[best_idx]
    model_name   = str(best_row["Model"])
    dataset_name = str(best_row["Dataset"])

    print(f"\n{'='*50}")
    print(f"  BEST MODEL : {model_name} on {dataset_name}")
    print(f"  F1={best_row['F1']}  Acc={best_row['Accuracy']}")
    print(f"{'='*50}\n")

    # Mark winner in CSV
    df.loc[best_idx, "IsBest"] = True
    df.to_csv(COMPARISON_CSV, index=False)

    # Copy artifacts to best_model/
    src_dir = os.path.join(EXPERIMENTS_DIR, model_name, dataset_name)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)

    joblib.dump(
        joblib.load(os.path.join(src_dir, "model.pkl")),
        os.path.join(BEST_MODEL_DIR, "model.pkl")
    )

    src_scaler = os.path.join(src_dir, "scaler.pkl")
    dst_scaler = os.path.join(BEST_MODEL_DIR, "scaler.pkl")
    if os.path.exists(src_scaler):
        joblib.dump(joblib.load(src_scaler), dst_scaler)
    elif os.path.exists(dst_scaler):
        os.remove(dst_scaler)   # remove stale scaler from previous best

    # Save metadata as CSV (no JSON)
    info_df = pd.DataFrame([{
        "BestModel":   model_name,
        "BestDataset": dataset_name,
        "F1":          best_row["F1"],
        "Accuracy":    best_row["Accuracy"],
        "Precision":   best_row["Precision"],
        "Recall":      best_row["Recall"],
    }])
    info_df.to_csv(os.path.join(BEST_MODEL_DIR, "best_info.csv"), index=False)

    return model_name, dataset_name, best_row
