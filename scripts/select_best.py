"""
select_best.py
Read experiments/comparison.csv and select the best model by F1 score.
Copy model.pkl (and scaler.pkl if present) into best_model/.
Mark the winning row as IsBest=True in comparison.csv.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from utils import select_and_save_best


def select_best():
    model_name, dataset_name, best_row = select_and_save_best()
    print(f"[select_best] Best model saved: {model_name} trained on {dataset_name}")
    print(f"  Accuracy  : {best_row['Accuracy']}")
    print(f"  Precision : {best_row['Precision']}")
    print(f"  Recall    : {best_row['Recall']}")
    print(f"  F1        : {best_row['F1']}")
    return model_name, dataset_name


if __name__ == "__main__":
    select_best()
