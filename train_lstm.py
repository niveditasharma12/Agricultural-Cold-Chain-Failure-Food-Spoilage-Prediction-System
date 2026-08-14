"""
=============================================================
  Cold Chain Project — LSTM Shelf-Life Predictor
=============================================================
  Predicts how many hours of shelf life remain for cargo,
  based on a sliding window of recent sensor readings per truck.

  IMPORTANT: see food_shelf_life.py — the target this model
  learns to predict ("remaining_shelf_life_hours") is a derived
  heuristic, not measured ground truth. Read that file's
  docstring before trusting this model's output in any real
  decision-making context.

  Sequence design:
    - Readings are sorted by timestamp per truck_id
    - A sliding window of SEQ_LEN consecutive readings becomes
      one training example
    - The target is the derived remaining-shelf-life value at
      the LAST timestep of each window
    - Trucks are split train/test at the TRUCK level (not row
      level) to avoid leaking a truck's patterns between train
      and test

  Logged to MLflow:
    - hyperparameters
    - MAE, RMSE on the held-out trucks
    - the trained Keras model, registered as
      "cold-chain-lstm-shelf-life"
    - the fitted StandardScaler (needed at inference time)

  Run:
    python train_lstm.py
=============================================================
"""

import os
import logging
import tempfile
from datetime import datetime

import numpy as np
import pandas as pd
import joblib

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

import mlflow
import mlflow.keras

from food_shelf_life import compute_remaining_shelf_life_hours

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("TrainLSTM")

# ── Config ────────────────────────────────────────────────
CSV_FILE   = os.environ.get("CSV_FILE", "augmented_cold_chain_dataset.csv")
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
EXPERIMENT = "cold_chain_shelf_life_lstm"
MODEL_NAME = "cold-chain-lstm-shelf-life"

SEQ_LEN     = 10   # readings per sliding-window sequence
TEST_TRUCKS = 5    # hold these trucks out entirely for testing
EPOCHS      = 15
BATCH_SIZE  = 64

FEATURE_COLS = [
    "Temperature", "Humidity", "Methane", "CO2",
    "temp_humidity_index", "gas_spoilage_score",
    "methane_co2_ratio", "temp_deviation", "storage_risk_score",
]


def build_sequences(df, truck_ids, scaler, fit_scaler=False):
    """Turns a truck-level dataframe into (X, y) sliding-window sequences."""
    if fit_scaler:
        scaler.fit(df.loc[df["truck_id"].isin(truck_ids), FEATURE_COLS])

    X_list, y_list = [], []
    for truck_id in truck_ids:
        truck_df = (
            df[df["truck_id"] == truck_id]
            .sort_values("timestamp")
            .reset_index(drop=True)
        )
        if len(truck_df) < SEQ_LEN + 1:
            continue

        features_scaled = scaler.transform(truck_df[FEATURE_COLS])
        targets = truck_df["remaining_shelf_life_hours"].values

        for i in range(len(truck_df) - SEQ_LEN):
            X_list.append(features_scaled[i:i + SEQ_LEN])
            y_list.append(targets[i + SEQ_LEN - 1])

    return np.array(X_list), np.array(y_list)


def main():
    log.info(f"Loading {CSV_FILE} ...")
    df = pd.read_csv(CSV_FILE)
    log.info(f"Loaded {len(df):,} rows across {df['truck_id'].nunique()} trucks.")

    # ── Derive the regression target (see food_shelf_life.py) ──
    df["remaining_shelf_life_hours"] = compute_remaining_shelf_life_hours(
        df["Food_Name"], df["Storage_Days"], df["storage_risk_score"]
    )

    # ── Truck-level train/test split (avoids leakage) ──────────
    all_trucks = sorted(df["truck_id"].unique())
    test_trucks = all_trucks[-TEST_TRUCKS:]
    train_trucks = all_trucks[:-TEST_TRUCKS]
    log.info(f"Train trucks: {len(train_trucks)} | Test trucks: {len(test_trucks)}")

    scaler = StandardScaler()
    X_train, y_train = build_sequences(df, train_trucks, scaler, fit_scaler=True)
    X_test, y_test = build_sequences(df, test_trucks, scaler, fit_scaler=False)
    log.info(f"Train sequences: {X_train.shape} | Test sequences: {X_test.shape}")

    # ── MLflow setup ────────────────────────────────────────
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT)

    params = dict(
        seq_len=SEQ_LEN,
        lstm_units_1=64,
        lstm_units_2=32,
        dropout=0.2,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        n_features=len(FEATURE_COLS),
    )

    with mlflow.start_run(run_name=f"lstm_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
        mlflow.log_params(params)
        mlflow.log_param("train_trucks", len(train_trucks))
        mlflow.log_param("test_trucks", len(test_trucks))

        log.info("Building LSTM model...")
        model = Sequential([
            LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, len(FEATURE_COLS))),
            Dropout(0.2),
            LSTM(32),
            Dropout(0.2),
            Dense(16, activation="relu"),
            Dense(1),
        ])
        model.compile(optimizer="adam", loss="mse", metrics=["mae"])

        early_stop = EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True)

        log.info(f"Training for up to {EPOCHS} epochs...")
        history = model.fit(
            X_train, y_train,
            validation_split=0.1,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[early_stop],
            verbose=2,
        )

        y_pred = model.predict(X_test).flatten()
        mae  = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))

        mlflow.log_metric("test_mae_hours", mae)
        mlflow.log_metric("test_rmse_hours", rmse)
        mlflow.log_metric("epochs_run", len(history.history["loss"]))

        log.info(f"Test MAE: {mae:.2f} hours | Test RMSE: {rmse:.2f} hours")

        # ── Log scaler + model as artifacts ───────────────────
        with tempfile.TemporaryDirectory() as tmpdir:
            scaler_path = os.path.join(tmpdir, "scaler.joblib")
            joblib.dump(scaler, scaler_path)
            mlflow.log_artifact(scaler_path)

        mlflow.keras.log_model(
            model,
            artifact_path="model",
            registered_model_name=MODEL_NAME,
        )
        log.info(f"Model registered as '{MODEL_NAME}'.")

    log.info("✅ LSTM training complete.")


if __name__ == "__main__":
    main()
