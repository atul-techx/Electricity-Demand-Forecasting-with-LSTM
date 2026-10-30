"""
evaluate.py
-----------
Metric computation and visualisation for the electricity demand forecasting
pipeline.

Responsibilities:
    1. Inverse-scale predictions to original MW units.
    2. Compute MAE and RMSE for Baseline and LSTM models.
    3. Generate a publication-quality Matplotlib comparison plot.
    4. Optionally plot LSTM training loss curves.
    5. Run iterative out-of-sample future forecasting (forecast_future).
    6. Plot a future forecast curve extending beyond the historical dataset.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler


@tf.function(reduce_retracing=True)
def _predict_step(model, x_tensor):
    """Pre-compiled graph step for ultra-fast multi-step recurrent forecasting."""
    return model(x_tensor, training=False)


# ---------------------------------------------------------------------------
# Output directory for saved figures
# ---------------------------------------------------------------------------
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Metric Computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "Model",
) -> dict:
    """
    Compute MAE and RMSE and print a formatted summary.

    Parameters
    ----------
    y_true     : Ground-truth values in original MW units.
    y_pred     : Predicted values in original MW units.
    model_name : Label used in the printed output.

    Returns
    -------
    dict  ->  {"mae": float, "rmse": float}
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    print(f"\n{'-'*45}")
    print(f"  {model_name:30s}")
    print(f"{'-'*45}")
    print(f"  MAE  : {mae:>10.2f} MW")
    print(f"  RMSE : {rmse:>10.2f} MW")
    print(f"{'-'*45}")

    return {"mae": mae, "rmse": rmse}


def evaluate_all(
    y_true: np.ndarray,
    baseline_preds: np.ndarray,
    lstm_preds: np.ndarray,
) -> dict:
    """
    Evaluate both models and return a nested metrics dict.

    Parameters
    ----------
    y_true         : Actual MW values (test set).
    baseline_preds : Naive persistence predictions (MW).
    lstm_preds     : LSTM predictions (MW).

    Returns
    -------
    dict  ->  {"baseline": {...}, "lstm": {...}}
    """
    metrics = {
        "baseline": compute_metrics(y_true, baseline_preds, "Naive Baseline (Persistence)"),
        "lstm": compute_metrics(y_true, lstm_preds, "LSTM (Stacked 2-Layer)"),
    }
    return metrics


# ---------------------------------------------------------------------------
# 2. Inverse Scaling Helper
# ---------------------------------------------------------------------------

def inverse_scale_predictions(
    baseline_scaled: np.ndarray,
    lstm_scaled: np.ndarray,
    scaler: MinMaxScaler,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Invert MinMax scaling for both model outputs.

    Parameters
    ----------
    baseline_scaled : Baseline predictions in [0,1] domain.
    lstm_scaled     : LSTM raw predictions in [0,1] domain.
    scaler          : The same MinMaxScaler fitted on training data.

    Returns
    -------
    (baseline_mw, lstm_mw) : Predictions in original MW units.
    """
    baseline_mw = scaler.inverse_transform(
        baseline_scaled.reshape(-1, 1)
    ).flatten()
    lstm_mw = scaler.inverse_transform(
        lstm_scaled.reshape(-1, 1)
    ).flatten()
    return baseline_mw, lstm_mw


# ---------------------------------------------------------------------------
# 3. Comparison Plot
# ---------------------------------------------------------------------------

def plot_forecast_comparison(
    y_true: np.ndarray,
    baseline_preds: np.ndarray,
    lstm_preds: np.ndarray,
    metrics: dict,
    datetimes=None,
    n_plot: int = 500,
    save: bool = True,
) -> None:
    """
    Generate and optionally save a Matplotlib comparison plot.

    Shows the first `n_plot` test observations for clarity.

    Parameters
    ----------
    y_true         : Ground-truth MW values.
    baseline_preds : Naive baseline MW predictions.
    lstm_preds     : LSTM MW predictions.
    metrics        : Dict returned by evaluate_all().
    datetimes      : Optional datetime index for x-axis labels.
    n_plot         : Number of test points to display (default 500).
    save           : Whether to save the figure to disk.
    """
    n = min(n_plot, len(y_true))
    x = datetimes[:n] if datetimes is not None else np.arange(n)

    fig, axes = plt.subplots(2, 1, figsize=(16, 10), facecolor="#0f1117")

    # -- Styling -------------------------------------------------------------
    PRIMARY_BG  = "#0f1117"
    PANEL_BG    = "#1a1d27"
    COLOR_ACTUAL = "#4cc9f0"
    COLOR_NAIVE  = "#f72585"
    COLOR_LSTM   = "#7bed9f"
    GRID_COLOR   = "#2a2d3a"

    for ax in axes:
        ax.set_facecolor(PANEL_BG)
        ax.tick_params(colors="white", labelsize=9)
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.title.set_color("white")
        ax.spines[:].set_color(GRID_COLOR)
        ax.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.7)

    # -- Top panel: Actual vs. LSTM -------------------------------------------
    ax1 = axes[0]
    ax1.plot(x, y_true[:n], color=COLOR_ACTUAL, linewidth=1.2,
             label="Actual", zorder=3)
    ax1.plot(x, lstm_preds[:n], color=COLOR_LSTM, linewidth=1.0,
             alpha=0.85, label=f"LSTM  (MAE={metrics['lstm']['mae']:.1f} MW)", zorder=2)
    ax1.set_title("Electricity Demand Forecast -- LSTM vs Actual", fontsize=13, pad=10)
    ax1.set_ylabel("Demand (MW)", fontsize=10)
    ax1.legend(loc="upper right", fontsize=9, facecolor=PANEL_BG, labelcolor="white",
               edgecolor=GRID_COLOR)
    if datetimes is not None:
        ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax1.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()

    # -- Bottom panel: Actual vs. Naive Baseline ------------------------------
    ax2 = axes[1]
    ax2.plot(x, y_true[:n], color=COLOR_ACTUAL, linewidth=1.2,
             label="Actual", zorder=3)
    ax2.plot(x, baseline_preds[:n], color=COLOR_NAIVE, linewidth=1.0,
             alpha=0.75, linestyle="--",
             label=f"Naive Baseline  (MAE={metrics['baseline']['mae']:.1f} MW)", zorder=2)
    ax2.set_title("Electricity Demand Forecast -- Naive Baseline vs Actual", fontsize=13, pad=10)
    ax2.set_ylabel("Demand (MW)", fontsize=10)
    ax2.set_xlabel("Date / Hour Index", fontsize=10)
    ax2.legend(loc="upper right", fontsize=9, facecolor=PANEL_BG, labelcolor="white",
               edgecolor=GRID_COLOR)
    if datetimes is not None:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        fig.autofmt_xdate()

    # -- Super-title ----------------------------------------------------------
    fig.suptitle(
        "EDF-LSTM  |  Electricity Demand Forecasting Pipeline",
        fontsize=15,
        color="white",
        y=0.98,
        fontweight="bold",
    )
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    if save:
        out_path = os.path.join(PLOTS_DIR, "forecast_comparison.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PRIMARY_BG)
        print(f"\n[evaluate] Comparison plot saved -> {out_path}")
    plt.close('all')


# ---------------------------------------------------------------------------
# 4. Training Loss Plot
# ---------------------------------------------------------------------------

def plot_training_loss(history, save: bool = True) -> None:
    """
    Plot training vs. validation loss curves from a Keras History object.

    Parameters
    ----------
    history : keras.callbacks.History
    save    : Whether to save the figure to disk.
    """
    loss     = history.history.get("loss", [])
    val_loss = history.history.get("val_loss", [])

    if not loss:
        print("[evaluate] No loss history to plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0f1117")
    ax.set_facecolor("#1a1d27")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#2a2d3a")
    ax.grid(True, color="#2a2d3a", linewidth=0.5, linestyle="--", alpha=0.7)

    epochs = range(1, len(loss) + 1)
    ax.plot(epochs, loss, color="#4cc9f0", linewidth=1.5, label="Train Loss (MSE)")
    if val_loss:
        ax.plot(epochs, val_loss, color="#f72585", linewidth=1.5,
                linestyle="--", label="Val Loss (MSE)")

    ax.set_title("LSTM Training Loss Curve", color="white", fontsize=12)
    ax.set_xlabel("Epoch", color="white")
    ax.set_ylabel("MSE Loss", color="white")
    ax.legend(facecolor="#1a1d27", labelcolor="white", edgecolor="#2a2d3a")

    plt.tight_layout()

    if save:
        out_path = os.path.join(PLOTS_DIR, "training_loss.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="#0f1117")
        print(f"[evaluate] Training loss plot saved -> {out_path}")
    plt.close('all')


# ---------------------------------------------------------------------------
# 5. Summary Table
# ---------------------------------------------------------------------------

def print_summary_table(metrics: dict) -> None:
    """
    Print a final ASCII comparison table of model metrics.

    Parameters
    ----------
    metrics : dict returned by evaluate_all().
    """
    b = metrics["baseline"]
    l = metrics["lstm"]

    print("\n" + "=" * 55)
    print("  FINAL MODEL COMPARISON SUMMARY")
    print("=" * 55)
    print(f"  {'Model':<30} {'MAE (MW)':>10} {'RMSE (MW)':>10}")
    print("-" * 55)
    print(f"  {'Naive Baseline (Persistence)':<30} {b['mae']:>10.2f} {b['rmse']:>10.2f}")
    print(f"  {'LSTM (Stacked 2-Layer)':<30} {l['mae']:>10.2f} {l['rmse']:>10.2f}")
    print("-" * 55)
    improvement_mae  = (b["mae"]  - l["mae"])  / b["mae"]  * 100
    improvement_rmse = (b["rmse"] - l["rmse"]) / b["rmse"] * 100
    print(f"  {'LSTM improvement (MAE)':<30} {improvement_mae:>9.1f}%")
    print(f"  {'LSTM improvement (RMSE)':<30} {improvement_rmse:>9.1f}%")
    print("=" * 55 + "\n")


# ---------------------------------------------------------------------------
# 6. Future Out-of-Sample Forecasting
# ---------------------------------------------------------------------------

def forecast_future(
    model,
    full_features: np.ndarray,
    scaler: MinMaxScaler,
    last_timestamp: pd.Timestamp = None,
    lookback: int = 24,
    steps: int = 168,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Autoregressively forecast `steps` time steps beyond the end of the dataset
    using a multi-feature LSTM with exact future calendar signals (diurnal wave preservation).
    """
    print(f"\n[evaluate] Running multi-feature future forecast: {steps} steps ({steps // 24} days) ahead ...")

    if last_timestamp is None:
        last_timestamp = pd.Timestamp("2024-04-30 23:00:00")

    future_dts = pd.date_range(
        start=last_timestamp + pd.Timedelta(hours=1),
        periods=steps,
        freq="h",
    )

    # Multi-feature input check (5 features: demand, sin_h, cos_h, sin_d, cos_d)
    if full_features.ndim == 1:
        hours = np.zeros(len(full_features), dtype=np.float32)
        full_features = np.column_stack([
            full_features.astype(np.float32),
            hours, hours, hours, hours,
        ])

    window = [list(row) for row in full_features[-lookback:]]

    future_preds_scaled = []
    for step in range(steps):
        x_input = tf.constant(window[-lookback:], shape=(1, lookback, 5), dtype=tf.float32)
        # Fast compiled graph execution
        pred = float(_predict_step(model, x_input).numpy()[0, 0])
        future_preds_scaled.append(pred)

        # Inject exact future deterministic time signals for the next hour
        cur_dt = future_dts[step]
        sin_h = float(np.sin(2 * np.pi * cur_dt.hour / 24))
        cos_h = float(np.cos(2 * np.pi * cur_dt.hour / 24))
        sin_d = float(np.sin(2 * np.pi * cur_dt.dayofweek / 7))
        cos_d = float(np.cos(2 * np.pi * cur_dt.dayofweek / 7))

        window.append([pred, sin_h, cos_h, sin_d, cos_d])

    future_preds_scaled = np.array(future_preds_scaled, dtype=np.float32)
    future_preds_mw = scaler.inverse_transform(
        future_preds_scaled.reshape(-1, 1)
    ).flatten()

    print(f"[evaluate] Forecast complete.")
    print(f"  Predicted demand range: {future_preds_mw.min():.0f} - {future_preds_mw.max():.0f} MW")
    return future_preds_mw, future_preds_scaled


# ---------------------------------------------------------------------------
# 7. Future Forecast Plot
# ---------------------------------------------------------------------------

def plot_future_forecast(
    full_series: "pd.Series",          # noqa: F821  (imported lazily to avoid circular)
    future_preds_mw: np.ndarray,
    steps: int = 168,
    context_hours: int = 336,
    save: bool = True,
) -> None:
    """
    Plot the historical demand context and the future LSTM forecast as a
    single continuous timeline extending beyond the last known data point.

    Layout:
        - Left region  (grey shading) : last `context_hours` of actual data.
        - Vertical dashed line        : boundary between history and future.
        - Right region (blue shading) : `steps`-hour LSTM forecast.

    Parameters
    ----------
    full_series     : Complete MW pd.Series with DatetimeIndex (train + test).
    future_preds_mw : np.ndarray of future MW predictions (length = steps).
    steps           : Number of future steps that were forecast (default 168).
    context_hours   : How many trailing historical hours to display for context
                      (default 336 = 2 weeks).
    save            : Whether to save the figure to disk.
    """
    import pandas as pd

    # -- Build datetime index for the forecast region --------------------------
    last_ts   = full_series.index[-1]
    freq      = pd.infer_freq(full_series.index[-50:]) or "h"
    future_idx = pd.date_range(
        start=last_ts + pd.tseries.frequencies.to_offset(freq),
        periods=steps,
        freq=freq,
    )

    # -- Slice the historical context window -----------------------------------
    context = full_series.iloc[-context_hours:]

    # -- Colour palette --------------------------------------------------------
    PRIMARY_BG   = "#0f1117"
    PANEL_BG     = "#1a1d27"
    COLOR_HIST   = "#4cc9f0"     # cyan  - actual history
    COLOR_FUTURE = "#f9c74f"     # amber - future forecast
    COLOR_SHADE_HIST   = "#1e2a3a"
    COLOR_SHADE_FUTURE = "#2a1f10"
    GRID_COLOR   = "#2a2d3a"

    fig, ax = plt.subplots(figsize=(18, 6), facecolor=PRIMARY_BG)
    ax.set_facecolor(PANEL_BG)
    ax.tick_params(colors="white", labelsize=9)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.spines[:].set_color(GRID_COLOR)
    ax.grid(True, color=GRID_COLOR, linewidth=0.5, linestyle="--", alpha=0.7)

    # -- Historical context ----------------------------------------------------
    ax.fill_between(
        context.index, context.values,
        alpha=0.18, color=COLOR_HIST, label="_nolegend_"
    )
    ax.plot(
        context.index, context.values,
        color=COLOR_HIST, linewidth=1.3, label=f"Historical Demand (last {context_hours}h)",
    )

    # -- Future forecast -------------------------------------------------------
    ax.fill_between(
        future_idx, future_preds_mw,
        alpha=0.22, color=COLOR_FUTURE, label="_nolegend_"
    )
    ax.plot(
        future_idx, future_preds_mw,
        color=COLOR_FUTURE, linewidth=1.8,
        label=f"LSTM Forecast (next {steps}h / {steps // 24} days)",
    )

    # -- Vertical boundary line ------------------------------------------------
    ax.axvline(
        x=last_ts, color="#ff6b6b", linewidth=1.5,
        linestyle="--", alpha=0.9, label="Forecast Start",
    )

    # -- Background region shading ---------------------------------------------
    ax.axvspan(context.index[0],  last_ts,          alpha=0.06, color=COLOR_SHADE_HIST)
    ax.axvspan(last_ts,           future_idx[-1],   alpha=0.10, color=COLOR_SHADE_FUTURE)

    # -- Annotation on the forecast region ------------------------------------
    ax.annotate(
        f"  {steps}-Hour Forecast\n  ({steps // 24} Days Ahead)",
        xy=(future_idx[steps // 6], future_preds_mw[steps // 6]),
        xytext=(future_idx[steps // 4], future_preds_mw.max() * 0.98),
        color=COLOR_FUTURE,
        fontsize=9,
        arrowprops=dict(arrowstyle="->", color=COLOR_FUTURE, lw=1.2),
    )

    # -- Labels & title --------------------------------------------------------
    ax.set_title(
        "EDF-LSTM  |  Out-of-Sample Future Demand Forecast",
        color="white", fontsize=14, fontweight="bold", pad=12,
    )
    ax.set_xlabel("Date / Time", color="white", fontsize=10)
    ax.set_ylabel("Electricity Demand (MW)", color="white", fontsize=10)
    ax.legend(
        loc="lower left", fontsize=9,
        facecolor=PANEL_BG, labelcolor="white", edgecolor=GRID_COLOR,
    )
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%d %b %Y"))
    ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    fig.autofmt_xdate(rotation=30, ha="right")

    plt.tight_layout()

    if save:
        out_path = os.path.join(PLOTS_DIR, "future_forecast.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=PRIMARY_BG)
        print(f"[evaluate] Future forecast plot saved -> {out_path}")
    plt.close('all')

