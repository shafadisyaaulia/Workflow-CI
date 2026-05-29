"""
modelling.py (MLflow Project version)
======================================
Script modelling yang mendukung argumen CLI untuk dijalankan
melalui MLflow Project dan GitHub Actions CI.

Usage:
    python modelling.py --n_estimators 100 --max_depth 10
"""

import os
import argparse
import json
import warnings
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, classification_report,
)
from sklearn.utils.estimator_html_repr import estimator_html_repr

warnings.filterwarnings("ignore")

# ==============================================================
# ARGUMEN CLI
# ==============================================================
parser = argparse.ArgumentParser(description="Credit Scoring Model Training")
parser.add_argument("--n_estimators",     type=int,   default=100)
parser.add_argument("--max_depth",        type=int,   default=10)
parser.add_argument("--min_samples_split",type=int,   default=5)
parser.add_argument("--data_dir",         type=str,   default="credit_scoring_preprocessing")
args = parser.parse_args()

# ==============================================================
# SETUP
# ==============================================================
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000/")
mlflow.set_tracking_uri(tracking_uri)
mlflow.set_experiment("Credit-Scoring-CI")

# ==============================================================
# LOAD DATA
# ==============================================================
print("=" * 55)
print("  MLflow Project: Credit Scoring Training")
print("=" * 55)
print(f"  Tracking URI : {tracking_uri}")
print(f"  Data dir     : {args.data_dir}")
print(f"  Params       : n_estimators={args.n_estimators}, max_depth={args.max_depth}, min_samples_split={args.min_samples_split}")

X_train = pd.read_csv(os.path.join(args.data_dir, "X_train.csv"))
X_test  = pd.read_csv(os.path.join(args.data_dir, "X_test.csv"))
y_train = pd.read_csv(os.path.join(args.data_dir, "y_train.csv")).squeeze()
y_test  = pd.read_csv(os.path.join(args.data_dir, "y_test.csv")).squeeze()

# ==============================================================
# TRAINING & MANUAL LOGGING
# ==============================================================
with mlflow.start_run(run_name="CI_RandomForest") as run:
    print(f"\n  Run ID: {run.info.run_id}")

    # Log parameters
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("min_samples_split", args.min_samples_split)
    mlflow.log_param("random_state", 42)

    # Train
    model = RandomForestClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc       = accuracy_score(y_test, y_pred)
    f1        = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall    = recall_score(y_test, y_pred)
    roc_auc   = roc_auc_score(y_test, y_prob)

    # Log metrics
    mlflow.log_metric("accuracy",  acc)
    mlflow.log_metric("f1_score",  f1)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall",    recall)
    mlflow.log_metric("roc_auc",   roc_auc)

    print(f"\n  Accuracy  : {acc:.4f}")
    print(f"  F1-Score  : {f1:.4f}")
    print(f"  Precision : {precision:.4f}")
    print(f"  Recall    : {recall:.4f}")
    print(f"  ROC-AUC   : {roc_auc:.4f}")

    # Confusion matrix artifact
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay(cm, display_labels=["No Default", "Default"]).plot(ax=ax, cmap="Blues")
    ax.set_title("Confusion Matrix", fontsize=12, fontweight="bold")
    plt.tight_layout()
    cm_path = "training_confusion_matrix.png"
    plt.savefig(cm_path, dpi=100)
    plt.close()
    mlflow.log_artifact(cm_path)

    # Estimator HTML artifact
    html_path = "estimator.html"
    with open(html_path, "w") as f:
        f.write(estimator_html_repr(model))
    mlflow.log_artifact(html_path)

    # metric_info.json artifact
    metric_dict = {
        "accuracy": acc, "f1_score": f1, "precision": precision,
        "recall": recall, "roc_auc": roc_auc,
    }
    json_path = "metric_info.json"
    with open(json_path, "w") as f:
        json.dump(metric_dict, f, indent=2)
    mlflow.log_artifact(json_path)

    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Simpan run_id untuk CI
    with open("latest_run_id.txt", "w") as f:
        f.write(run.info.run_id)
    print(f"\n  Run ID disimpan: latest_run_id.txt")

print("\n✅ Training CI selesai!")
