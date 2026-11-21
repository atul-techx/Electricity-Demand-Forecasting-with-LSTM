# EDF-LSTM — Electricity Demand Forecasting with LSTM, Baseline Comparison & Web Dashboard

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg?logo=fastapi)](https://fastapi.tiangolo.com)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.13+-FF6F00.svg?logo=tensorflow)](https://tensorflow.org)
[![Chart.js](https://img.shields.io/badge/Chart.js-4.4-FF6384.svg?logo=chartdotjs)](https://www.chartjs.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A production-ready deep learning framework for **Electricity Demand Forecasting** featuring a **Stacked 2-Layer LSTM**, **Cyclical Diurnal & Weekly Calendar Signals**, **Naive Persistence Baseline benchmarking**, **Recursive Multi-Step Out-of-Sample Future Projections (7 Days / 168 Hours)**, and a **Modern Glassmorphic Web Application (FastAPI + Chart.js)**.

---

## ⚡ Key Highlights

- **Deep Learning Architecture**: Stacked 2-Layer LSTM (`LSTM(64) -> Dropout(0.2) -> LSTM(32) -> Dropout(0.2) -> Dense(1)`) trained with Adam optimizer and EarlyStopping.
- **Engineered Cyclical Features**: Incorporates sine/cosine cyclical encoding for time-of-day (diurnal 24h) and day-of-week (weekly 7d) patterns to prevent recursive forecast attenuation/damping.
- **Strict Leakage Prevention**: Chronological 80% train / 20% test split; `MinMaxScaler` fit strictly on the training partition only.
- **Superior Benchmark Performance**:
  - **MAE**: **2,158.88 MW** vs. Naive Baseline **4,085.79 MW** (**47.2% error reduction**)
  - **RMSE**: **2,827.26 MW** vs. Naive Baseline **5,232.82 MW** (**46.0% error reduction**)
- **Out-of-Sample Future Projections**: Autoregressive recursive rolling window extrapolation predicting 24h to 168h (7 days) beyond the latest available timestamp.
- **Full-Stack Web Application**:
  - **FastAPI backend** with CORS and REST API endpoints (`/api/metrics`, `/api/forecast`, `/api/health`).
  - **Interactive Dark-Themed Glassmorphism UI** with real-time KPI cards, forecast horizon selector, and smooth Chart.js visualization.

---

## 📁 Project Directory Structure

```text
EDF-LSTM/
├── data/                                 # Raw & processed time-series datasets
│   ├── hourlyLoadDataIndia.csv           # Primary dataset: Indian National & Regional Demand (2019–2024)
│   └── electricity.csv                   # Alternative dataset: PJM/AEP Hourly Energy Consumption
│
├── models/                               # Serialized model weights, scalers & evaluation metrics
│   ├── saved_lstm_model.keras            # Trained 2-layer stacked LSTM neural network weights
│   ├── scaler.pkl                        # Fitted MinMaxScaler parameters (prevent data leakage)
│   ├── metrics.json                      # Performance comparison metrics (LSTM vs Naive Baseline)
│   └── test_cache.npz                    # Pre-computed test set predictions for instant dashboard load
│
├── plots/                                # High-resolution evaluation & forecast charts
│   ├── forecast_comparison.png           # Actual demand vs LSTM vs Naive Baseline comparison chart
│   ├── future_forecast.png               # 7-day (168-hour) out-of-sample future demand projection
│   ├── training_loss.png                 # Training vs Validation loss curves across training epochs
│   └── .gitkeep                          # Preserves directory in version control
│
├── src/                                  # Core Python machine learning package
│   ├── __init__.py                       # Package identifier
│   ├── data_loader.py                    # Data ingestion, cleaning, cyclical feature engineering & windowing
│   ├── models.py                         # Model definitions: NaivePersistenceBaseline & Stacked LSTM
│   └── evaluate.py                       # Evaluation metrics (MAE, RMSE), plotting & recursive forecasting engine
│
├── static/                               # Frontend web dashboard assets (served by FastAPI)
│   ├── index.html                        # Modern dark glassmorphic UI dashboard
│   ├── style.css                         # Mobile-responsive CSS layout, animations & design system
│   └── script.js                         # Interactive Chart.js graphs, slider controls & API client
│
├── app.py                                # Production FastAPI web application & REST API server
├── main.py                               # CLI pipeline execution script (train, evaluate & plot)
├── train_and_save.py                     # Standalone training script to train and save model artifacts
├── requirements.txt                      # Project dependencies (FastAPI, TensorFlow, Pandas, etc.)
├── .gitignore                            # Version control exclusion rules (cache, venv, checkpoints)
└── README.md                             # Comprehensive project documentation & usage guide
```

---

## 📖 File & Directory Guide

Understanding the role of every component in this project:

| File / Folder | Role & Purpose |
| :--- | :--- |
| **`app.py`** | **FastAPI Web Server**: Serves the single-page application dashboard and REST API endpoints (`/api/forecast`, `/api/metrics`, `/api/health`). Features non-blocking asynchronous startup. |
| **`main.py`** | **Full Pipeline Execution**: End-to-end CLI workflow that loads data, trains the LSTM model, evaluates benchmarks against the Naive Baseline, prints summary tables, and exports publication-grade plots. |
| **`train_and_save.py`** | **Model Training & Artifact Serialization**: Standalone script that trains the stacked LSTM model on the dataset and persists `saved_lstm_model.keras`, `scaler.pkl`, `metrics.json`, and `test_cache.npz` into `models/`. |
| **`src/data_loader.py`** | **Data Pipeline**: Handles CSV ingestion, column auto-detection, chronological train/test splitting (80/20), cyclical time-of-day/day-of-week feature engineering, and sliding-window sequence creation. |
| **`src/models.py`** | **Model Architectures**: Defines the `NaiveBaseline` (persistence benchmark) and the Keras multi-layer stacked LSTM network (`build_lstm` and `train_lstm`). |
| **`src/evaluate.py`** | **Evaluation & Inference Engine**: Implements inverse-scaling, MAE/RMSE calculations, `forecast_future()` autoregressive recursive projection loop, and Matplotlib plotting utilities. |
| **`data/`** | **Data Storage**: Contains hourly electricity load time-series datasets. `hourlyLoadDataIndia.csv` is the primary dataset containing national and regional hourly consumption. |
| **`models/`** | **Serialized Artifacts**: Holds pre-trained model weights (`saved_lstm_model.keras`), scaler parameters (`scaler.pkl`), and cached metrics (`metrics.json`) for instant dashboard serving. |
| **`plots/`** | **Visualizations**: Output directory for generated PNG figures comparing actual demand against LSTM predictions and future recursive forecasts. |
| **`static/`** | **Frontend UI**: Self-contained client dashboard assets (HTML5, responsive CSS3 glassmorphism, and Chart.js controller) served directly by FastAPI. |
| **`requirements.txt`** | **Dependencies**: Declares verified package versions for Python dependencies. |

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment

```bash
git clone https://github.com/atul-techx/Electricity-Demand-Forecasting-with-LSTM.git
cd Electricity-Demand-Forecasting-with-LSTM

# (Optional but recommended) Create and activate a virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows PowerShell
# source .venv/bin/activate  # macOS / Linux

pip install -r requirements.txt
```

### 2. Launch the Interactive Web Dashboard

Run the FastAPI application from the project root:

```bash
python app.py
```

*Or run with Uvicorn directly:*
```bash
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at: **[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🖥️ Running the CLI Pipeline

To run the complete data loading, model training, evaluation, and plot generation pipeline:

```bash
python main.py
```

Outputs are automatically saved to `plots/`:
- `plots/forecast_comparison.png` — Actual demand vs. LSTM vs. Naive Baseline on the test set
- `plots/future_forecast.png` — 7-day forward out-of-sample recursive forecast curve
- `plots/training_loss.png` — Training and validation loss curves across training epochs

---

## 🔄 Retraining & Updating Saved Models

To retrain the LSTM network from scratch and regenerate all serialized artifacts (`models/`):

```bash
python train_and_save.py
```

---

## 📊 Benchmark Results

| Model | MAE (MW) | RMSE (MW) | Improvement (MAE) | Improvement (RMSE) |
|---|---|---|---|---|
| **Naive Persistence Baseline** | 4,085.79 | 5,232.82 | Benchmark | Benchmark |
| **Stacked LSTM (2-Layer)** | **2,158.88** | **2,827.26** | **+47.2%** | **+46.0%** |

---

## 🌐 API Reference

### `GET /api/health`
Checks server readiness, model initialization state, and loaded record count.

### `GET /api/metrics`
Returns model metadata, performance metrics, and baseline comparisons in JSON.

### `GET /api/forecast?hours=168&history_hours=168`
Returns historical test data, baseline benchmark, LSTM predictions, and continuous future projections for Chart.js rendering.
- `hours`: Future forecast horizon in hours (`24`, `48`, `72`, `168`, etc.).
- `history_hours`: Trailing historical hours displayed as context.

---

## 📜 License

This project is licensed under the MIT License.
