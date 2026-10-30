---
title: Electricity Demand Forecasting LSTM
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# EDF-LSTM — Electricity Demand Forecasting with LSTM, Baseline Comparison & Web Dashboard

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00.svg?logo=tensorflow)](https://tensorflow.org)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.4-FF6384.svg?logo=chartdotjs)](https://www.chartjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready deep learning framework for **Electricity Demand Forecasting** featuring a **Stacked 2-Layer LSTM**, **Naive Persistence Baseline benchmarking**, **Recursive Multi-Step Out-of-Sample Future Projections (7 Days / 168 Hours)**, and a **Modern Glassmorphic Web Application (FastAPI + Chart.js)**.

---

## ⚡ Key Highlights

- **Deep Learning Architecture**: Stacked 2-Layer LSTM (`LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(1)`) trained with Adam optimizer and EarlyStopping.
- **Strict Leakage Prevention**: Chronological 80% train / 20% test split; `MinMaxScaler` fit strictly on the training partition only.
- **Strong Performance Improvement**:
  - **MAE**: **2,158.88 MW** vs. Naive Baseline **4,085.79 MW** (**47.2% reduction**)
  - **RMSE**: **2,827.26 MW** vs. Naive Baseline **5,232.82 MW** (**46.0% reduction**)
- **Out-of-Sample Future Projections**: Autoregressive recursive rolling window extrapolation predicting 24h to 168h (7 days) beyond the latest available timestamp.
- **Full-Stack Web Application**:
  - **FastAPI backend** with CORS and REST API endpoints (`/api/metrics`, `/api/forecast`, `/api/health`).
  - **Interactive Dark-Themed Glassmorphism UI** with real-time KPI cards, horizon slider/pills, and smooth Chart.js visualization.

---

## 📁 Repository Structure

```text
.
├── app.py                            # FastAPI application root entrypoint
├── requirements.txt                  # Full project dependencies
├── README.md                         # Documentation
├── static/                           # Web application assets
│   ├── index.html                    # Dashboard UI
│   ├── style.css                     # Glassmorphic dark design system
│   └── script.js                     # Interactive Chart.js & API controller
├── models/
│   ├── saved_lstm_model.keras        # Saved trained LSTM model
│   ├── scaler.pkl                    # Fitted MinMaxScaler
│   └── metrics.json                  # Cached benchmark evaluation metrics
└── electricity_forecasting/
    ├── main.py                       # CLI pipeline execution script
    ├── app.py                        # FastAPI core server
    ├── data/
    │   ├── hourlyLoadDataIndia.xlsx  # Indian hourly load dataset (2019-2024)
    │   └── electricity.csv           # Alternative CSV dataset
    ├── plots/
    │   ├── future_forecast.png       # 7-day future forecast curve
    │   ├── forecast_comparison.png   # LSTM vs. Baseline on test set
    │   └── training_loss.png         # Training vs. validation loss curve
    └── src/
        ├── data_loader.py            # Ingestion, auto-detection, scaling & windowing
        ├── models.py                 # Naive baseline & Keras LSTM architecture
        └── evaluate.py               # Metrics, plots, and future forecasting engine
```

---

## 🚀 Quick Start

### 1. Clone & Install Dependencies

```bash
git clone https://github.com/atul-techx/Electricity-Demand-Forecasting-with-LSTM.git
cd Electricity-Demand-Forecasting-with-LSTM

# (Optional) Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
```

### 2. Launch the Web Dashboard

Start the FastAPI server:

```bash
python -m uvicorn app:app --reload
```

Or run via Python:

```bash
python app.py
```

Then navigate to: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🖥️ CLI Pipeline Execution

To run the complete data loading, model training, evaluation, and plot generation pipeline:

```bash
cd electricity_forecasting
python main.py
```

Outputs are automatically saved to `plots/`:
- `plots/future_forecast.png` — 7-day forward out-of-sample forecast curve
- `plots/forecast_comparison.png` — Actual vs. LSTM vs. Naive Baseline test comparison
- `plots/training_loss.png` — Training and validation MSE loss curves

---

## 📊 Benchmark Results

| Model | MAE (MW) | RMSE (MW) | Improvement (MAE) | Improvement (RMSE) |
|---|---|---|---|---|
| **Naive Persistence Baseline** | 4,085.79 | 5,232.82 | Benchmark | Benchmark |
| **Stacked LSTM (2-Layer)** | **2,158.88** | **2,827.26** | **+47.2%** | **+46.0%** |

---

## 🌐 API Reference

### `GET /api/metrics`
Returns model metadata, performance metrics, and baseline comparisons.

### `GET /api/forecast?hours=168&history_hours=168`
Returns historical test data, baseline benchmark, LSTM predictions, and recursive future projections.
- `hours`: Future horizon in hours (e.g., `24`, `48`, `72`, `168`).
- `history_hours`: Trailing historical hours for visual context.

### `GET /api/health`
Health check status and loaded dataset sequence count.

---

## 📜 License

This project is licensed under the MIT License.
