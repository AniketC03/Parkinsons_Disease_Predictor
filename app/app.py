"""
app.py
Flask web application for Parkinson's Disease prediction.

Always loads from:  parkinsons_project/app_model/model.pkl
                    parkinsons_project/app_model/scaler.pkl  (if present)
                    parkinsons_project/app_model/info.csv

To switch which model powers the app, run from scripts/:
  python connect_app.py --model RandomForest --dataset parkinsons_disease_data
Then restart this server.
"""

import os
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request

# ─────────────────────────────────────────────────────────────────────────────
# Paths  (app_model/ lives at the project root, next to app/, scripts/, etc.)
# ─────────────────────────────────────────────────────────────────────────────
BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP_MODEL_DIR = os.path.join(BASE_DIR, "app_model")
MODEL_PATH    = os.path.join(APP_MODEL_DIR, "model.pkl")
SCALER_PATH   = os.path.join(APP_MODEL_DIR, "scaler.pkl")
INFO_CSV      = os.path.join(APP_MODEL_DIR, "info.csv")

app = Flask(__name__)

# ─────────────────────────────────────────────────────────────────────────────
# Load model at startup (once — stays in memory for all requests)
# ─────────────────────────────────────────────────────────────────────────────
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        "\n\n  ✖  No model found at app_model/model.pkl\n"
        "     Run this from the scripts/ folder first:\n"
        "       python connect_app.py --model <ModelName> --dataset <DatasetName>\n"
    )

model  = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None

model_info = {"Model": "Unknown", "Dataset": "Unknown", "F1": "N/A", "Accuracy": "N/A"}
if os.path.exists(INFO_CSV):
    model_info = pd.read_csv(INFO_CSV).iloc[0].to_dict()

# ─────────────────────────────────────────────────────────────────────────────
# Feature definitions
# type = "number"  → plain numeric input with placeholder
# type = "select"  → Yes/No dropdown (value sent to model = 0 or 1)
# ─────────────────────────────────────────────────────────────────────────────
FEATURES = [
    # ── Demographics ─────────────────────────────────────────────────────────
    {"name":"Age",           "label":"Age (years)",          "type":"number", "min":20,  "max":100, "step":1,   "placeholder":"e.g. 65"},
    {"name":"Gender",        "label":"Gender",               "type":"select", "options":[("","-- Select --"),("0","Female"),("1","Male")]},
    {"name":"Ethnicity",     "label":"Ethnicity",            "type":"select", "options":[("","-- Select --"),("0","Caucasian"),("1","African American"),("2","Asian"),("3","Other")]},
    {"name":"EducationLevel","label":"Education Level",      "type":"select", "options":[("","-- Select --"),("0","None / Primary"),("1","High School"),("2","Bachelor's"),("3","Higher Degree")]},
    {"name":"BMI",           "label":"BMI",                  "type":"number", "min":10,  "max":50,  "step":0.1, "placeholder":"e.g. 24.5"},

    # ── Lifestyle ─────────────────────────────────────────────────────────────
    {"name":"Smoking",             "label":"Smoking",                         "type":"select", "options":[("","-- Select --"),("0","No"),("1","Yes")]},
    {"name":"AlcoholConsumption",  "label":"Alcohol Consumption (units/week)","type":"number", "min":0,  "max":20,  "step":0.1, "placeholder":"e.g. 4.0"},
    {"name":"PhysicalActivity",    "label":"Physical Activity (hrs/week)",    "type":"number", "min":0,  "max":10,  "step":0.1, "placeholder":"e.g. 3.5"},
    {"name":"DietQuality",         "label":"Diet Quality (0–10)",             "type":"number", "min":0,  "max":10,  "step":0.1, "placeholder":"e.g. 6.5"},
    {"name":"SleepQuality",        "label":"Sleep Quality (4–10)",            "type":"number", "min":4,  "max":10,  "step":0.1, "placeholder":"e.g. 7.0"},

    # ── Medical History ───────────────────────────────────────────────────────
    {"name":"FamilyHistoryParkinsons","label":"Family History of Parkinson's","type":"select","options":[("","-- Select --"),("0","No"),("1","Yes")]},
    {"name":"TraumaticBrainInjury",   "label":"Traumatic Brain Injury",       "type":"select","options":[("","-- Select --"),("0","No"),("1","Yes")]},
    {"name":"Hypertension",           "label":"Hypertension",                 "type":"select","options":[("","-- Select --"),("0","No"),("1","Yes")]},
    {"name":"Diabetes",               "label":"Diabetes",                     "type":"select","options":[("","-- Select --"),("0","No"),("1","Yes")]},
    {"name":"Depression",             "label":"Depression",                   "type":"select","options":[("","-- Select --"),("0","No"),("1","Yes")]},
    {"name":"Stroke",                 "label":"Stroke",                       "type":"select","options":[("","-- Select --"),("0","No"),("1","Yes")]},

    # ── Clinical Measurements ─────────────────────────────────────────────────
    {"name":"SystolicBP",              "label":"Systolic Blood Pressure (mmHg)", "type":"number","min":90,  "max":200,"step":1,   "placeholder":"e.g. 120"},
    {"name":"DiastolicBP",             "label":"Diastolic Blood Pressure (mmHg)","type":"number","min":60,  "max":130,"step":1,   "placeholder":"e.g. 80"},
    {"name":"CholesterolTotal",        "label":"Total Cholesterol (mg/dL)",      "type":"number","min":100, "max":400,"step":0.1, "placeholder":"e.g. 200"},
    {"name":"CholesterolLDL",          "label":"LDL Cholesterol (mg/dL)",        "type":"number","min":50,  "max":250,"step":0.1, "placeholder":"e.g. 120"},
    {"name":"CholesterolHDL",          "label":"HDL Cholesterol (mg/dL)",        "type":"number","min":20,  "max":100,"step":0.1, "placeholder":"e.g. 55"},
    {"name":"CholesterolTriglycerides","label":"Triglycerides (mg/dL)",          "type":"number","min":50,  "max":400,"step":0.1, "placeholder":"e.g. 150"},
    {"name":"UPDRS",                   "label":"UPDRS Score (0–199)",            "type":"number","min":0,   "max":199,"step":0.1, "placeholder":"e.g. 40"},
    {"name":"MoCA",                    "label":"MoCA Score (0–30)",              "type":"number","min":0,   "max":30, "step":0.1, "placeholder":"e.g. 22"},
    {"name":"FunctionalAssessment",    "label":"Functional Assessment (0–10)",   "type":"number","min":0,   "max":10, "step":0.1, "placeholder":"e.g. 7"},

    # ── Symptoms ──────────────────────────────────────────────────────────────
    {"name":"Tremor",            "label":"Tremor",             "type":"select","options":[("","-- Select --"),("0","No"),("1","Yes")]},
    {"name":"Rigidity",          "label":"Rigidity",           "type":"select","options":[("","-- Select --"),("0","No"),("1","Yes")]},
    {"name":"Bradykinesia",      "label":"Bradykinesia",       "type":"select","options":[("","-- Select --"),("0","No"),("1","Yes")]},
    {"name":"PosturalInstability","label":"Postural Instability","type":"select","options":[("","-- Select --"),("0","No"),("1","Yes")]},
    {"name":"SpeechProblems",    "label":"Speech Problems",    "type":"select","options":[("","-- Select --"),("0","No"),("1","Yes")]},
    {"name":"SleepDisorders",    "label":"Sleep Disorders",    "type":"select","options":[("","-- Select --"),("0","No"),("1","Yes")]},
    {"name":"Constipation",      "label":"Constipation",       "type":"select","options":[("","-- Select --"),("0","No"),("1","Yes")]},
]

SECTIONS = [
    (" Patient Demographics",   FEATURES[0:5]),
    (" Lifestyle Factors",      FEATURES[5:10]),
    (" Medical History",        FEATURES[10:16]),
    (" Clinical Measurements",  FEATURES[16:25]),
    (" Symptoms",               FEATURES[25:]),
]


# ─────────────────────────────────────────────────────────────────────────────
# Routes
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html",
                           sections=SECTIONS,
                           model_info=model_info,
                           result=None)


@app.route("/predict", methods=["POST"])
def predict():
    errors       = []
    input_values = {}

    for feat in FEATURES:
        raw = request.form.get(feat["name"], "").strip()
        if raw == "":
            errors.append(f"Please fill in: {feat['label']}")
            input_values[feat["name"]] = ""
            continue
        try:
            input_values[feat["name"]] = float(raw)
        except ValueError:
            errors.append(f"Invalid value for: {feat['label']}")
            input_values[feat["name"]] = ""

    if errors:
        return render_template("index.html",
                               sections=SECTIONS,
                               model_info=model_info,
                               result=None,
                               errors=errors,
                               form_values=input_values)

    # Build array in exact training column order
    X = np.array([[input_values[f["name"]] for f in FEATURES]])

    if scaler is not None:
        X = scaler.transform(X)

    prediction = int(model.predict(X)[0])
    proba      = model.predict_proba(X)[0]
    confidence = round(float(proba[prediction]) * 100, 1)

    result = {
        "prediction":  prediction,
        "label":       "Parkinson's Disease Detected" if prediction == 1 else "No Parkinson's Disease",
        "confidence":  confidence,
        "risk_class":  "positive" if prediction == 1 else "negative",
        "proba_pd":    round(float(proba[1]) * 100, 1),
        "proba_no_pd": round(float(proba[0]) * 100, 1),
    }

    return render_template("index.html",
                           sections=SECTIONS,
                           model_info=model_info,
                           result=result,
                           form_values=input_values)


if __name__ == "__main__":
    print(f"\n  Model   : {model_info['Model']}")
    print(f"  Dataset : {model_info['Dataset']}")
    print(f"  F1      : {model_info['F1']}   Accuracy : {model_info['Accuracy']}")
    print(f"  Scaler  : {'Yes' if scaler else 'No'}\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
