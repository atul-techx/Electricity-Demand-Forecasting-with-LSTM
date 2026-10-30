"""
app.py
------
Production-ready FastAPI backend for Electricity Demand Forecasting.
Serves REST API endpoints for metrics, test forecasts, and future out-of-sample projections,
along with static dashboard frontend assets.
"""

import os
import sys
import json
import joblib
import threading
import numpy as np
import pandas as pd
from contextlib import asynccontextmanager
from typing import Dict, Any, List

from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse

# Set environment & paths (prevent matplotlib and tensorflow startup stalls)
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(BASE_DIR, "src")
MODELS_DIR = os.path.join(BASE_DIR, "models")
STATIC_DIR = os.path.join(BASE_DIR, "static")

if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

import keras
from data_loader import prepare_data, scale, LOOKBACK
from models import build_lstm, train_lstm, NaiveBaseline
from evaluate import inverse_scale_predictions, evaluate_all, forecast_future

MODEL_PATH = os.path.join(MODELS_DIR, "saved_lstm_model.keras")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")
TEST_CACHE_PATH = os.path.join(MODELS_DIR, "test_cache.npz")

# In-memory application state
state: Dict[str, Any] = {
    "model": None,
    "scaler": None,
    "data": None,
    "metrics": None,
    "full_series": None,
    "full_scaled": None,
    "full_features": None,
    "test_datetimes": None,
    "baseline_mw": None,
    "lstm_mw": None,
    "y_test_mw": None,
    "forecast_cache": {},
}


def load_or_train_model():
    """Load pre-trained LSTM model from models/ or train and persist it if missing."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    print("[app] Preparing dataset and preprocessing pipeline...")
    data = prepare_data()
    state["data"] = data
    scaler = data["scaler"]
    state["scaler"] = scaler

    # Reconstruct full series and multi-feature array for rolling forecasts
    train_ser = data["train_series"]
    test_ser = data["test_series"]
    full_ser = pd.concat([train_ser, test_ser])
    state["full_series"] = full_ser
    state["full_features"] = data["full_features"]

    train_scaled = scale(train_ser, scaler)
    test_scaled = scale(test_ser, scaler)
    state["full_scaled"] = np.concatenate([train_scaled, test_scaled])

    # Check for saved model
    if os.path.exists(MODEL_PATH):
        print(f"[app] Loading pre-trained LSTM model from {MODEL_PATH}...")
        try:
            model = keras.models.load_model(MODEL_PATH)
            print("[app] Pre-trained model loaded successfully.")
        except Exception as e:
            print(f"[app] Error loading {MODEL_PATH}: {e}. Retraining...")
            model = None
    else:
        model = None

    if model is None:
        n_features = data["X_train"].shape[2]
        print(f"[app] No existing model found. Training multi-feature stacked LSTM (n_features={n_features})...")
        model = build_lstm(lookback=LOOKBACK, n_features=n_features)
        train_lstm(
            model,
            data["X_train"],
            data["y_train"],
            epochs=25,
            batch_size=128,
        )
        print(f"[app] Saving newly trained model to {MODEL_PATH}...")
        model.save(MODEL_PATH)
        joblib.dump(scaler, SCALER_PATH)

    state["model"] = model

    # Evaluate test set predictions (fast-path: load cached predictions)
    y_test_mw = data["y_test_raw"]
    state["y_test_mw"] = y_test_mw
    state["test_datetimes"] = test_ser.index[LOOKBACK : LOOKBACK + len(y_test_mw)]

    cached_test = None
    if os.path.exists(TEST_CACHE_PATH):
        try:
            npz = np.load(TEST_CACHE_PATH)
            baseline_mw = npz["baseline_mw"]
            lstm_mw = npz["lstm_mw"]
            cached_test = True
            print(f"[app] Loaded pre-calculated test predictions from {TEST_CACHE_PATH}.")
        except Exception as e:
            print(f"[app] Could not read test cache: {e}")
            cached_test = False

    if not cached_test:
        print("[app] Generating test-set baseline and LSTM predictions...")
        X_test = data["X_test"]
        baseline = NaiveBaseline(lookback=LOOKBACK)
        baseline_pred_scaled = baseline.predict(X_test)
        lstm_pred_scaled = model.predict(X_test, verbose=0).flatten()

        baseline_mw, lstm_mw = inverse_scale_predictions(
            baseline_pred_scaled, lstm_pred_scaled, scaler
        )
        try:
            np.savez_compressed(
                TEST_CACHE_PATH,
                y_test_mw=y_test_mw,
                baseline_mw=baseline_mw,
                lstm_mw=lstm_mw,
            )
        except Exception:
            pass

    state["baseline_mw"] = baseline_mw
    state["lstm_mw"] = lstm_mw

    # Load or compute metrics
    if os.path.exists(METRICS_PATH):
        try:
            with open(METRICS_PATH, "r") as f:
                metrics = json.load(f)
        except Exception:
            metrics = None
    else:
        metrics = None

    if not metrics:
        eval_res = evaluate_all(y_test_mw, baseline_mw, lstm_mw)
        metrics = {
            "status": "online",
            "model_type": "Stacked 2-Layer LSTM (64-32 units)",
            "lookback_hours": LOOKBACK,
            "total_records": len(full_ser),
            "train_records": len(train_ser),
            "test_records": len(test_ser),
            "baseline": {
                "name": "Naive Persistence Baseline",
                "mae": float(eval_res["baseline"]["mae"]),
                "rmse": float(eval_res["baseline"]["rmse"]),
            },
            "lstm": {
                "name": "Stacked LSTM (2-Layer)",
                "mae": float(eval_res["lstm"]["mae"]),
                "rmse": float(eval_res["lstm"]["rmse"]),
            },
            "improvements": {
                "mae_percentage": round(
                    (eval_res["baseline"]["mae"] - eval_res["lstm"]["mae"])
                    / eval_res["baseline"]["mae"]
                    * 100,
                    1,
                ),
                "rmse_percentage": round(
                    (eval_res["baseline"]["rmse"] - eval_res["lstm"]["rmse"])
                    / eval_res["baseline"]["rmse"]
                    * 100,
                    1,
                ),
            },
            "dataset": {
                "start_date": str(train_ser.index[0]),
                "end_date": str(test_ser.index[-1]),
                "target_column": "Electricity Demand",
                "unit": "MW",
            },
        }
        with open(METRICS_PATH, "w") as f:
            json.dump(metrics, f, indent=2)

    state["metrics"] = metrics

    # Pre-warm common forecast horizons for zero-latency initial page load
    print("[app] Pre-warming forecast cache for instant responses...")
    last_dt = full_ser.index[-1]
    for h in (24, 72, 168):
        if h not in state["forecast_cache"]:
            f_mw, _ = forecast_future(
                model=model,
                full_features=state["full_features"],
                scaler=scaler,
                last_timestamp=last_dt,
                lookback=LOOKBACK,
                steps=h,
            )
            state["forecast_cache"][h] = f_mw

    print("[app] Initialization completed. Ready to serve traffic.")


init_lock = threading.Lock()
is_initializing = False


def ensure_initialized():
    """Ensure model and dataset state are loaded (thread-safe)."""
    global is_initializing
    if state["model"] is not None:
        return
    with init_lock:
        if state["model"] is None:
            is_initializing = True
            try:
                load_or_train_model()
            finally:
                is_initializing = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Lifecycle startup handler:
    Spawns background initialization thread so Uvicorn binds to $PORT in <100ms.
    This guarantees Render/cloud port scanners pass immediately without timing out.
    """
    init_thread = threading.Thread(target=ensure_initialized, name="ModelInitThread", daemon=True)
    init_thread.start()
    yield
    print("[app] Shutting down application.")


app = FastAPI(
    title="Electricity Demand Forecasting API",
    description="Production-ready FastAPI backend for Electricity Demand Forecasting using Stacked LSTM",
    version="1.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health():
    """
    Health check endpoint:
    Returns 200 OK immediately for cloud health checks and port scanners.
    """
    is_ready = state["model"] is not None
    return {
        "status": "online" if is_ready else "initializing",
        "ready": is_ready,
        "initializing": is_initializing,
        "records_loaded": len(state["full_series"]) if state["full_series"] is not None else 0,
    }


@app.get("/api/metrics")
def get_metrics():
    """Return model performance metrics and comparison benchmarks."""
    ensure_initialized()
    if state["metrics"] is None:
        raise HTTPException(status_code=503, detail="Metrics not initialized yet")
    return JSONResponse(content=state["metrics"])


@app.get("/api/forecast")
def get_forecast(
    hours: int = Query(default=168, ge=1, le=720, description="Future forecast hours (e.g. 24 or 168)"),
    history_hours: int = Query(default=168, ge=24, le=1000, description="Trailing historical hours to return"),
):
    """
    Returns historical test demand, baseline prediction, LSTM test prediction,
    and out-of-sample future predictions.
    """
    ensure_initialized()
    if state["model"] is None or state["full_scaled"] is None:
        raise HTTPException(status_code=503, detail="Model is still initializing")

    model = state["model"]
    scaler = state["scaler"]
    full_scaled = state["full_scaled"]
    full_series = state["full_series"]
    test_dt = state["test_datetimes"]
    y_test_mw = state["y_test_mw"]
    baseline_mw = state["baseline_mw"]
    lstm_mw = state["lstm_mw"]

    # 1. Historical Context Slice (last `history_hours` of test data)
    ctx_slice = min(history_hours, len(test_dt))
    h_dates = [dt.strftime("%Y-%m-%d %H:%M") for dt in test_dt[-ctx_slice:]]
    h_actual = [round(float(v), 2) for v in y_test_mw[-ctx_slice:]]
    h_baseline = [round(float(v), 2) for v in baseline_mw[-ctx_slice:]]
    h_lstm = [round(float(v), 2) for v in lstm_mw[-ctx_slice:]]

    # 2. Future Out-of-Sample Forecast (cached for speed)
    cache_key = hours
    if cache_key in state["forecast_cache"]:
        future_mw = state["forecast_cache"][cache_key]
    else:
        future_mw, _ = forecast_future(
            model=model,
            full_features=state["full_features"],
            scaler=scaler,
            last_timestamp=full_series.index[-1],
            lookback=LOOKBACK,
            steps=hours,
        )
        state["forecast_cache"][cache_key] = future_mw

    # Build future timestamps
    last_ts = full_series.index[-1]
    future_range = pd.date_range(
        start=last_ts + pd.Timedelta(hours=1),
        periods=hours,
        freq="h",
    )
    f_dates = [dt.strftime("%Y-%m-%d %H:%M") for dt in future_range]
    f_forecast = [round(float(v), 2) for v in future_mw]

    # 3. Continuous Combined Arrays for seamless Chart.js rendering
    # Connect history and future at the boundary point so the forecast line doesn't float disconnected
    combined_labels = h_dates + f_dates
    combined_actual = h_actual + [None] * len(f_dates)
    combined_baseline = h_baseline + [None] * len(f_dates)
    combined_lstm = h_lstm + [None] * len(f_dates)

    # Future series starts from the last historical point for continuous visualization
    combined_future = [None] * (len(h_dates) - 1) + [h_actual[-1]] + f_forecast

    response_data = {
        "metadata": {
            "requested_forecast_hours": hours,
            "history_hours": ctx_slice,
            "last_known_timestamp": h_dates[-1],
            "forecast_start": f_dates[0],
            "forecast_end": f_dates[-1],
            "unit": "MW",
            "forecast_min_mw": round(float(np.min(future_mw)), 1),
            "forecast_max_mw": round(float(np.max(future_mw)), 1),
            "forecast_avg_mw": round(float(np.mean(future_mw)), 1),
            "peak_hour_timestamp": f_dates[int(np.argmax(future_mw))],
        },
        "historical": {
            "timestamps": h_dates,
            "actual": h_actual,
            "baseline": h_baseline,
            "lstm": h_lstm,
        },
        "future": {
            "timestamps": f_dates,
            "forecast": f_forecast,
        },
        "combined": {
            "labels": combined_labels,
            "actual": combined_actual,
            "baseline": combined_baseline,
            "lstm": combined_lstm,
            "future_forecast": combined_future,
            "split_index": len(h_dates) - 1,
        },
    }
    return JSONResponse(content=response_data)


# Mount static directory and root route
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def serve_dashboard():
    """Serve the single-page application dashboard."""
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return JSONResponse(
        status_code=404,
        content={"message": "Frontend index.html not found in static/"},
    )


if __name__ == "__main__":
    import uvicorn

    print("\n" + "=" * 60)
    print("  EDF-LSTM  |  FastAPI Application Starting")
    print("  Open dashboard at: http://127.0.0.1:8000")
    print("=" * 60 + "\n")
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
