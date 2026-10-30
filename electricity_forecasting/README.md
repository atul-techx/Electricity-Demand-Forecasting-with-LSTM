# EDF-LSTM — Electricity Demand Forecasting with LSTM & Baseline Comparison

A clean, modular Python project that trains a **stacked 2-layer LSTM** on
hourly electricity demand data and benchmarks it against a **Naive Persistence
Baseline**. Both models are evaluated with MAE and RMSE and visualised with
publication-quality Matplotlib plots.

---

## Project Structure

```
electricity_forecasting/
├── data/
│   └── electricity.csv         ← place your CSV here (or let the auto-downloader handle it)
├── src/
│   ├── data_loader.py          ← ingestion, chronological split, scaling, windowing
│   ├── models.py               ← Naive Baseline & Keras LSTM architecture
│   └── evaluate.py             ← MAE / RMSE metrics & Matplotlib plots
├── plots/                      ← auto-created; saved PNG figures land here
├── main.py                     ← single entry point → python main.py
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1 — Install dependencies

```bash
# (optional but recommended) create a virtual environment
python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

### 2 — Get the dataset  *(choose ONE option)*

**Option A — Automatic download via Kaggle API** *(recommended)*

Configure your Kaggle credentials (one-time setup):

```bash
# Windows PowerShell
$env:KAGGLE_USERNAME = "your_username"
$env:KAGGLE_KEY      = "your_api_key"

# macOS / Linux
export KAGGLE_USERNAME="your_username"
export KAGGLE_KEY="your_api_key"
```

> Your API key is available at **kaggle.com → Account → Create New API Token**.  
> The pipeline will download and cache the dataset automatically on first run.

**Option B — Manual download**

1. Visit <https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption>
2. Download and extract the archive.
3. Copy `AEP_hourly.csv` (or any hourly MW CSV) to:

```
electricity_forecasting/data/electricity.csv
```

### 3 — Run the pipeline

```bash
cd electricity_forecasting
python main.py
```

The script will:
1. Load & preprocess data (sort, fill gaps, scale)
2. Build LSTM sequences (24-hour lookback → 1-hour forecast)
3. Train the stacked LSTM (EarlyStopping enabled)
4. Evaluate Baseline vs LSTM (MAE, RMSE)
5. Save plots to `electricity_forecasting/plots/`

---

## Technical Details

| Parameter | Value |
|---|---|
| Lookback window | 24 hours |
| Forecast horizon | 1 hour ahead |
| Train / Test split | 80 % / 20 % (chronological) |
| Scaler | MinMaxScaler fitted on **train only** |
| LSTM layers | 2 (64 units → 32 units) |
| Dropout | 0.20 |
| Optimiser | Adam (lr = 1e-3) |
| Loss | MSE |
| Max epochs | 20 (EarlyStopping: patience = 5) |
| Batch size | 64 |

---

## Output

| File | Description |
|---|---|
| `plots/forecast_comparison.png` | Actual vs LSTM vs Naive Baseline (test set) |
| `plots/training_loss.png` | Train / validation MSE loss curves |
| Console | MAE & RMSE summary table for both models |

---

## No Data Leakage Guarantee

- The `MinMaxScaler` is **fit exclusively on the training partition**.
- The test set is transformed using the already-fitted scaler.
- The chronological 80/20 split ensures no future information leaks into training.

---

## Dependencies

| Package | Purpose |
|---|---|
| `tensorflow` | LSTM model (Keras API) |
| `scikit-learn` | MinMaxScaler, MAE/RMSE |
| `pandas` | Time-series DataFrame handling |
| `numpy` | Numerical operations |
| `matplotlib` | Visualisation |
| `kagglehub` | Automatic dataset download *(optional)* |
