"""
=============================================================
  Cold Chain Project — Random Forest Spoilage Classifier
=============================================================
  Trains a Random Forest on the 100,000-row cold chain sensor
  dataset to classify cargo as:
      0 = Fresh
      1 = At Risk
      2 = Spoiled
  (the "Spoiled" column in the dataset is already this 3-class
  label — no derivation needed, unlike the LSTM's target).

  Logged to MLflow:
    - hyperparameters
    - accuracy, macro F1, macro precision, macro recall
    - confusion matrix (image artifact)
    - classification report (text artifact)
    - SHAP summary plot (image artifact)
    - the trained model itself, registered as
      "cold-chain-rf-classifier"

  Also writes a summary row to the Postgres model_metrics table
  so model performance can be tracked/compared over time outside
  of MLflow too.

  Run:
    python train_random_forest.py
=============================================================
"""

import os
import logging
import tempfile
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # no display available in a container
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

import mlflow
import mlflow.sklearn
import shap
import psycopg2

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("TrainRandomForest")

# ── Config ────────────────────────────────────────────────
CSV_FILE     = os.environ.get("CSV_FILE", "augmented_cold_chain_dataset.csv")
MLFLOW_URI   = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT   = "cold_chain_spoilage_classifier"
MODEL_NAME   = "cold-chain-rf-classifier"
CLASS_NAMES  = ["Fresh", "At Risk", "Spoiled"]

NUMERIC_FEATURES = [
    "Temperature", "Humidity", "Methane", "CO2", "Storage_Days",
    "temp_humidity_index", "gas_spoilage_score", "methane_co2_ratio",
    "temp_deviation", "storage_risk_score",
]

PG_CONFIG = dict(
    host=os.environ.get("PG_HOST", "postgres"),
    dbname=os.environ.get("PG_DB", "airflow"),
    user=os.environ.get("PG_USER", "airflow"),
    password=os.environ.get("PG_PASSWORD", "airflow"),
)


def log_metrics_to_postgres(run_id, model_name, f1, accuracy, precision, recall):
    try:
        conn = psycopg2.connect(**PG_CONFIG)
        cur = conn.cursor()
        cur.execute(
            """
            INSERT INTO model_metrics
                (run_id, model_name, f1_score, accuracy, precision_val, recall_val)
            VALUES (%s, %s, %s, %s, %s, %s)
            """,
            (run_id, model_name, f1, accuracy, precision, recall),
        )
        conn.commit()
        cur.close()
        conn.close()
        log.info("Logged metrics to Postgres model_metrics table.")
    except Exception as e:
        log.warning(f"Could not log metrics to Postgres (non-fatal): {e}")


def main():
    log.info(f"Loading {CSV_FILE} ...")
    df = pd.read_csv(CSV_FILE)
    log.info(f"Loaded {len(df):,} rows.")

    # ── Features: numeric sensor readings + one-hot food type ──
    food_dummies = pd.get_dummies(df["Food_Name"], prefix="food")
    X = pd.concat([df[NUMERIC_FEATURES], food_dummies], axis=1)
    y = df["Spoiled"].astype(int)
    feature_names = list(X.columns)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    log.info(f"Train: {len(X_train):,} rows | Test: {len(X_test):,} rows")

    # ── MLflow setup ─────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    params = dict(
        n_estimators=300,
        max_depth=15,
        min_samples_leaf=5,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )

    with mlflow.start_run(run_name=f"rf_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        mlflow.log_params(params)
        mlflow.log_param("n_train_rows", len(X_train))
        mlflow.log_param("n_test_rows", len(X_test))
        mlflow.log_param("n_features", len(feature_names))

        log.info("Training RandomForestClassifier...")
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)

        accuracy  = accuracy_score(y_test, y_pred)
        f1_macro  = f1_score(y_test, y_pred, average="macro")
        prec_macro = precision_score(y_test, y_pred, average="macro")
        rec_macro  = recall_score(y_test, y_pred, average="macro")

        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("f1_macro", f1_macro)
        mlflow.log_metric("precision_macro", prec_macro)
        mlflow.log_metric("recall_macro", rec_macro)

        log.info(f"Accuracy: {accuracy:.4f} | F1 (macro): {f1_macro:.4f} | "
                 f"Precision (macro): {prec_macro:.4f} | Recall (macro): {rec_macro:.4f}")

        with tempfile.TemporaryDirectory() as tmpdir:
            # ── Confusion matrix artifact ──────────────────
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(cm, cmap="Blues")
            ax.set_xticks(range(3)); ax.set_xticklabels(CLASS_NAMES)
            ax.set_yticks(range(3)); ax.set_yticklabels(CLASS_NAMES)
            ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
            ax.set_title("Confusion Matrix — Spoilage Classifier")
            for i in range(3):
                for j in range(3):
                    ax.text(j, i, cm[i, j], ha="center", va="center",
                            color="white" if cm[i, j] > cm.max() / 2 else "black")
            fig.colorbar(im)
            cm_path = os.path.join(tmpdir, "confusion_matrix.png")
            fig.savefig(cm_path, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(cm_path)

            # ── Classification report artifact ─────────────
            report = classification_report(y_test, y_pred, target_names=CLASS_NAMES)
            report_path = os.path.join(tmpdir, "classification_report.txt")
            with open(report_path, "w") as f:
                f.write(report)
            mlflow.log_artifact(report_path)
            log.info(f"\n{report}")

            # ── SHAP explainability ─────────────────────────
            log.info("Computing SHAP values (sampled 500 test rows for speed)...")
            sample = X_test.sample(min(500, len(X_test)), random_state=42)
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(sample)

            fig = plt.figure(figsize=(9, 6))
            shap.summary_plot(
                shap_values, sample, plot_type="bar",
                class_names=CLASS_NAMES, show=False
            )
            shap_path = os.path.join(tmpdir, "shap_summary.png")
            fig.savefig(shap_path, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(shap_path)
            log.info("SHAP summary plot logged.")

        # ── Log + register the model itself ─────────────────
        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
            input_example=X_test.head(3),
        )
        log.info(f"Model registered as '{MODEL_NAME}'.")

        log_metrics_to_postgres(run.info.run_id, MODEL_NAME, f1_macro, accuracy, prec_macro, rec_macro)

    log.info("✅ Random Forest training complete.")


if __name__ == "__main__":
    main()