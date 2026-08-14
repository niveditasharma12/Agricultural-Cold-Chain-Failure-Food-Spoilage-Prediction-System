"""
=============================================================
  Cold Chain Project — Model Serving API
=============================================================
  Serves the current Production-stage models from MLflow:
    - RF classifier: Fresh / At Risk / Spoiled
    - LSTM regressor: estimated hours of shelf life remaining

  Both models are loaded once at startup. If promote_best_model.py
  later promotes a new version, restart this service to pick it
  up (or extend it to poll periodically — not done here to keep
  things simple).

  Endpoints:
    GET  /health
    POST /predict/spoilage     — single-reading classification
    POST /predict/shelf_life   — needs a sequence of readings

  Run:
    uvicorn serve_api:app --host 0.0.0.0 --port 8000
=============================================================
"""

import os
import logging
from typing import List, Optional

import numpy as np
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn
import mlflow.keras
from mlflow.tracking import MlflowClient
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from food_shelf_life import FOOD_MAX_SHELF_LIFE_HOURS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("ModelAPI")

MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://mlflow:5000")
mlflow.set_tracking_uri(MLFLOW_URI)

RF_MODEL_NAME   = "cold-chain-rf-classifier"
LSTM_MODEL_NAME = "cold-chain-lstm-shelf-life"
CLASS_NAMES     = ["Fresh", "At Risk", "Spoiled"]

FOOD_TYPES = sorted(FOOD_MAX_SHELF_LIFE_HOURS.keys())
NUMERIC_FEATURES = [
    "Temperature", "Humidity", "Methane", "CO2", "Storage_Days",
    "temp_humidity_index", "gas_spoilage_score", "methane_co2_ratio",
    "temp_deviation", "storage_risk_score",
]

app = FastAPI(
    title="Cold Chain Spoilage Prediction API",
    description="Serves live predictions from the Production-stage RF classifier and LSTM shelf-life model.",
    version="1.0.0",
)

rf_model = None
lstm_model = None
lstm_scaler = None


@app.on_event("startup")
def load_models():
    global rf_model, lstm_model, lstm_scaler

    try:
        rf_model = mlflow.sklearn.load_model(f"models:/{RF_MODEL_NAME}/Production")
        log.info(f"✅ Loaded Production RF classifier ({RF_MODEL_NAME}).")
    except Exception as e:
        log.warning(f"⚠️  Could not load RF classifier from Production stage: {e}")
        log.warning("   /predict/spoilage will return 503 until a model is promoted.")

    try:
        lstm_model = mlflow.keras.load_model(f"models:/{LSTM_MODEL_NAME}/Production")
        log.info(f"✅ Loaded Production LSTM model ({LSTM_MODEL_NAME}).")

        # The LSTM was trained on StandardScaler-normalized inputs. Load the
        # exact scaler fitted at training time (logged as an artifact on the
        # same run) so live inputs are normalized identically — skipping this
        # causes the model's tanh-based gates to saturate on out-of-range raw
        # values, producing near-constant output regardless of input.
        client = MlflowClient()
        versions = client.search_model_versions(f"name='{LSTM_MODEL_NAME}'")
        prod_version = next(v for v in versions if v.current_stage == "Production")
        scaler_path = mlflow.artifacts.download_artifacts(
            run_id=prod_version.run_id, artifact_path="scaler.joblib"
        )
        lstm_scaler = joblib.load(scaler_path)
        log.info("✅ Loaded matching StandardScaler for the LSTM model.")
    except Exception as e:
        log.warning(f"⚠️  Could not load LSTM model or its scaler from Production stage: {e}")
        log.warning("   /predict/shelf_life will return 503 until a model is promoted.")
        lstm_model = None  # don't serve predictions without the matching scaler
        lstm_scaler = None


# ── Request schemas ──────────────────────────────────────────
class SpoilageRequest(BaseModel):
    food_name: str = Field(..., description=f"One of: {FOOD_TYPES}")
    temperature: float
    humidity: float
    methane: float
    co2: float
    storage_days: float
    temp_humidity_index: float
    gas_spoilage_score: float
    methane_co2_ratio: float
    temp_deviation: float
    storage_risk_score: float


class SensorReading(BaseModel):
    temperature: float
    humidity: float
    methane: float
    co2: float
    temp_humidity_index: float
    gas_spoilage_score: float
    methane_co2_ratio: float
    temp_deviation: float
    storage_risk_score: float


class ShelfLifeRequest(BaseModel):
    readings: List[SensorReading] = Field(
        ..., min_items=10, max_items=10,
        description="Exactly 10 consecutive sensor readings, oldest first."
    )


# ── Endpoints ─────────────────────────────────────────────────
@app.get("/health")
def health():
    return {
        "status": "ok",
        "rf_model_loaded": rf_model is not None,
        "lstm_model_loaded": lstm_model is not None,
    }


@app.post("/predict/spoilage")
def predict_spoilage(req: SpoilageRequest):
    if rf_model is None:
        raise HTTPException(status_code=503, detail="RF classifier not loaded — no Production model yet.")

    row = {f: 0.0 for f in NUMERIC_FEATURES}
    row["Temperature"] = req.temperature
    row["Humidity"] = req.humidity
    row["Methane"] = req.methane
    row["CO2"] = req.co2
    row["Storage_Days"] = req.storage_days
    row["temp_humidity_index"] = req.temp_humidity_index
    row["gas_spoilage_score"] = req.gas_spoilage_score
    row["methane_co2_ratio"] = req.methane_co2_ratio
    row["temp_deviation"] = req.temp_deviation
    row["storage_risk_score"] = req.storage_risk_score

    # One-hot food column, matching training-time column naming
    for food in FOOD_TYPES:
        row[f"food_{food}"] = 1.0 if food == req.food_name else 0.0

    # Align to the exact column order the model expects
    expected_cols = list(rf_model.feature_names_in_)
    X = pd.DataFrame([row])
    missing = [c for c in expected_cols if c not in X.columns]
    for c in missing:
        X[c] = 0.0
    X = X[expected_cols]

    pred_class = int(rf_model.predict(X)[0])
    proba = rf_model.predict_proba(X)[0].tolist()

    return {
        "predicted_class": pred_class,
        "predicted_label": CLASS_NAMES[pred_class],
        "probabilities": {CLASS_NAMES[i]: round(p, 4) for i, p in enumerate(proba)},
    }


@app.post("/predict/shelf_life")
def predict_shelf_life(req: ShelfLifeRequest):
    if lstm_model is None or lstm_scaler is None:
        raise HTTPException(status_code=503, detail="LSTM model not loaded — no Production model yet.")

    feature_order = [
        "temperature", "humidity", "methane", "co2",
        "temp_humidity_index", "gas_spoilage_score",
        "methane_co2_ratio", "temp_deviation", "storage_risk_score",
    ]
    # Build a (10, 9) array — 10 timesteps, 9 features, same order as training
    raw_sequence = np.array([[getattr(r, f) for f in feature_order] for r in req.readings])

    # Apply the SAME StandardScaler fitted during training. Without this,
    # raw values sit far outside the range the model learned on, saturating
    # its tanh-based gates and producing near-constant output regardless
    # of input — this was the root cause of identical predictions across
    # very different trucks.
    scaled_sequence = lstm_scaler.transform(raw_sequence)
    scaled_sequence = scaled_sequence.reshape(1, 10, len(feature_order))

    predicted_hours = float(lstm_model.predict(scaled_sequence, verbose=0)[0][0])

    return {
        "predicted_remaining_hours": round(max(0, predicted_hours), 1),
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)