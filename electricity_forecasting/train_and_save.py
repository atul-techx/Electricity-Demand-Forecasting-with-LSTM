"""
train_and_save.py
-----------------
Trains the LSTM model on the dataset and saves it to:
    models/saved_lstm_model.keras

Also computes and exports evaluation metrics and scaler parameters.
"""

import os
import sys
import json
import joblib
import numpy as np

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import prepare_data, LOOKBACK
from models import NaiveBaseline, build_lstm, train_lstm
from evaluate import inverse_scale_predictions, evaluate_all

MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")
os.makedirs(MODELS_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODELS_DIR, "saved_lstm_model.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")


def train_and_persist():
    print("[train_and_save] Loading & preparing dataset...")
    data = prepare_data()

    X_train = data["X_train"]
    y_train = data["y_train"]
    X_test = data["X_test"]
    y_test_mw = data["y_test_raw"]
    scaler = data["scaler"]

    print("[train_and_save] Running Naive Baseline benchmark...")
    baseline = NaiveBaseline(lookback=LOOKBACK)
    baseline_pred_scaled = baseline.predict(X_test)

    n_features = X_train.shape[2]
    print(f"[train_and_save] Building and training multi-feature stacked LSTM (n_features={n_features})...")
    model = build_lstm(lookback=LOOKBACK, n_features=n_features)
    train_lstm(model, X_train, y_train, epochs=25, batch_size=128)

    print(f"[train_and_save] Saving model to {MODEL_PATH}...")
    model.save(MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("[train_and_save] Evaluating model...")
    lstm_pred_scaled = model.predict(X_test, verbose=0).flatten()
    baseline_mw, lstm_mw = inverse_scale_predictions(
        baseline_pred_scaled, lstm_pred_scaled, scaler
    )

    # Save test cache for instant server startup
    np.savez_compressed(
        os.path.join(MODELS_DIR, "test_cache.npz"),
        y_test_mw=y_test_mw,
        baseline_mw=baseline_mw,
        lstm_mw=lstm_mw,
    )

    metrics = evaluate_all(y_test_mw, baseline_mw, lstm_mw)

    metrics_export = {
        "status": "online",
        "model_type": f"Stacked 2-Layer LSTM ({n_features} Features: Demand + Cyclical Diurnal/Weekly Time Signals)",
        "lookback_hours": LOOKBACK,
        "total_records": len(data["train_series"]) + len(data["test_series"]),
        "train_records": len(data["train_series"]),
        "test_records": len(data["test_series"]),
        "baseline": {
            "name": "Naive Persistence Baseline",
            "mae": float(metrics["baseline"]["mae"]),
            "rmse": float(metrics["baseline"]["rmse"]),
        },
        "lstm": {
            "name": "Stacked LSTM (2-Layer)",
            "mae": float(metrics["lstm"]["mae"]),
            "rmse": float(metrics["lstm"]["rmse"]),
        },
        "improvements": {
            "mae_percentage": round(
                (metrics["baseline"]["mae"] - metrics["lstm"]["mae"])
                / metrics["baseline"]["mae"]
                * 100,
                1,
            ),
            "rmse_percentage": round(
                (metrics["baseline"]["rmse"] - metrics["lstm"]["rmse"])
                / metrics["baseline"]["rmse"]
                * 100,
                1,
            ),
        },
        "dataset": {
            "start_date": str(data["train_series"].index[0]),
            "end_date": str(data["test_series"].index[-1]),
            "target_column": "Electricity Demand",
            "unit": "MW",
        },
    }

    with open(METRICS_PATH, "w") as f:
        json.dump(metrics_export, f, indent=2)

    print(f"[train_and_save] Successfully saved model to {MODEL_PATH}")
    print(f"[train_and_save] Successfully saved metrics to {METRICS_PATH}")
    return metrics_export


if __name__ == "__main__":
    train_and_persist()
