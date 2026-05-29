"""
connect_app.py
══════════════
Connects the Flask web app to whichever model + dataset you choose.

This script copies the chosen model.pkl (and scaler.pkl if present)
into app_model/ so that app.py always loads from one fixed place.

Usage:
  python connect_app.py --model RandomForest     --dataset parkinsons_disease_data
  python connect_app.py --model LogisticRegression --dataset parkinsons_disease_data2
  python connect_app.py --model KNN              --dataset parkinsons_disease_data3

After running this, start the app with:
  cd ../app
  python app.py
"""

import sys, os, shutil, argparse
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd

ROOT       = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
EXP_DIR    = os.path.join(ROOT, "experiments")
APP_MODEL  = os.path.join(ROOT, "app_model")   # ← app.py always reads from here
MODELS_DIR = os.path.join(ROOT, "models")

VALID_MODELS = ["RandomForest", "LogisticRegression", "KNN"]
VALID_DATA   = ["parkinsons_disease_data", "parkinsons_disease_data2", "parkinsons_disease_data3"]


def connect(model_name, dataset_name):
    src = os.path.join(EXP_DIR, model_name, dataset_name)

    # ── Validate ──────────────────────────────────────────────────────────────
    if model_name not in VALID_MODELS:
        print(f"  ✖  Unknown model '{model_name}'.")
        print(f"     Choose from: {VALID_MODELS}")
        sys.exit(1)

    if dataset_name not in VALID_DATA:
        print(f"  ✖  Unknown dataset '{dataset_name}'.")
        print(f"     Choose from: {VALID_DATA}")
        sys.exit(1)

    model_path = os.path.join(src, "model.pkl")
    if not os.path.exists(model_path):
        print(f"  ✖  model.pkl not found at: {model_path}")
        print(f"     Run the training script first:")
        print(f"       python run_{model_name.lower().replace('logisticregression','logistic')}.py")
        sys.exit(1)

    # ── Copy artifacts ────────────────────────────────────────────────────────
    os.makedirs(APP_MODEL, exist_ok=True)

    # Always copy model.pkl
    shutil.copy2(model_path, os.path.join(APP_MODEL, "model.pkl"))

    # Copy scaler only if it exists (RF has no scaler)
    src_scaler = os.path.join(src, "scaler.pkl")
    dst_scaler = os.path.join(APP_MODEL, "scaler.pkl")
    if os.path.exists(src_scaler):
        shutil.copy2(src_scaler, dst_scaler)
    elif os.path.exists(dst_scaler):
        os.remove(dst_scaler)   # remove stale scaler from previous connection

    # ── Read that model's F1/Accuracy from its results.csv ───────────────────
    results_csv = os.path.join(MODELS_DIR, model_name, "results.csv")
    f1 = acc = prec = rec = "N/A"
    if os.path.exists(results_csv):
        df = pd.read_csv(results_csv)
        row = df[df["Dataset"] == dataset_name]
        if not row.empty:
            r    = row.iloc[0]
            f1   = r["F1"]
            acc  = r["Accuracy"]
            prec = r["Precision"]
            rec  = r["Recall"]

    # ── Write info.csv for app.py header badge ───────────────────────────────
    info = pd.DataFrame([{
        "Model":     model_name,
        "Dataset":   dataset_name,
        "F1":        f1,
        "Accuracy":  acc,
        "Precision": prec,
        "Recall":    rec,
    }])
    info.to_csv(os.path.join(APP_MODEL, "info.csv"), index=False)

    # ── Done ─────────────────────────────────────────────────────────────────
    print()
    print("=" * 55)
    print("  App connected successfully!")
    print("=" * 55)
    print(f"  Model   : {model_name}")
    print(f"  Dataset : {dataset_name}")
    print(f"  F1      : {f1}   Accuracy : {acc}")
    print(f"  Scaler  : {'Yes' if os.path.exists(src_scaler) else 'No (RF does not need it)'}")
    print()
    print("  Start the web app:")
    print("    cd ../app")
    print("    python app.py")
    print()


def list_available():
    """Show which experiments have been trained so far."""
    print()
    print("Available trained experiments:")
    print("-" * 55)
    found = False
    for m in VALID_MODELS:
        for d in VALID_DATA:
            p = os.path.join(EXP_DIR, m, d, "model.pkl")
            if os.path.exists(p):
                # Try to get F1
                f1 = "?"
                rc = os.path.join(MODELS_DIR, m, "results.csv")
                if os.path.exists(rc):
                    df = pd.read_csv(rc)
                    row = df[df["Dataset"] == d]
                    if not row.empty:
                        f1 = row.iloc[0]["F1"]
                print(f"  ✔  {m:<22}  {d:<32}  F1={f1}")
                found = True
    if not found:
        print("  No models trained yet.")
        print("  Run: python run_rf.py / run_knn.py / run_logistic.py first.")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Connect a trained model to the Flask app.")
    parser.add_argument("--model",   type=str, help="Model name: RandomForest | LogisticRegression | KNN")
    parser.add_argument("--dataset", type=str, help="Dataset name: parkinsons_disease_data | parkinsons_disease_data2 | parkinsons_disease_data3")
    parser.add_argument("--list",    action="store_true", help="List all available trained models")
    args = parser.parse_args()

    if args.list:
        list_available()
    elif args.model and args.dataset:
        connect(args.model, args.dataset)
    else:
        parser.print_help()
        print()
        list_available()
