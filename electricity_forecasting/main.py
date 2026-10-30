"""
main.py
-------
Single entry-point for the Electricity Demand Forecasting pipeline.

Run the full workflow with:
    python main.py

Workflow stages executed in order:
    1.  Load & preprocess data       (data_loader.prepare_data)
    2.  Run Naive Baseline            (models.NaiveBaseline)
    3.  Build & train LSTM            (models.build_lstm / train_lstm)
    4.  Inverse-scale outputs         (evaluate.inverse_scale_predictions)
    5.  Compute MAE / RMSE            (evaluate.evaluate_all)
    6.  Plot results                  (evaluate.plot_forecast_comparison)
    7.  Print summary table           (evaluate.print_summary_table)
    8.  Future out-of-sample forecast (evaluate.forecast_future + plot_future_forecast)
"""

import os
import sys
import numpy as np

# -- Suppress verbose TensorFlow logs (keep warnings & errors) ---------------
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

# Allow running from the project root without installing the package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from data_loader import prepare_data, LOOKBACK, inverse_scale
from models import NaiveBaseline, build_lstm, train_lstm
from evaluate import (
    inverse_scale_predictions,
    evaluate_all,
    plot_forecast_comparison,
    plot_training_loss,
    print_summary_table,
    forecast_future,
    plot_future_forecast,
)


# -----------------------------------------------------------------------------
# Hyperparameters  (adjust here - no need to edit other files)
# -----------------------------------------------------------------------------
EPOCHS        = 50    # maximum training epochs (EarlyStopping may stop earlier)
BATCH_SIZE    = 128
N_PLOT        = 1000  # points to display in the comparison chart
FUTURE_STEPS  = 168  # steps to forecast into the future (168 = 7 days x 24 hours)
CONTEXT_HOURS = 336  # historical hours shown in the future forecast plot (2 weeks)


def main() -> None:
    print("\n" + "=" * 60)
    print("  EDF-LSTM  |  Electricity Demand Forecasting Pipeline")
    print("=" * 60 + "\n")

    # ---------------------------------------------
    # Stage 1: Data Preparation
    # ---------------------------------------------
    print("-- Stage 1: Data Preparation ------------------------------")
    data = prepare_data()

    X_train      = data["X_train"]
    y_train      = data["y_train"]
    X_test       = data["X_test"]
    y_test       = data["y_test"]          # scaled
    y_test_mw    = data["y_test_raw"]      # original MW units
    scaler       = data["scaler"]
    train_ser    = data["train_series"]
    test_ser     = data["test_series"]

    # Full MW series (train + test) needed to seed the future forecast
    import pandas as pd
    full_series = pd.concat([train_ser, test_ser])

    # Datetime index for x-axis labels (aligned with windowed test sequences)
    try:
        # The windowed sequences start LOOKBACK steps into the test set
        test_datetimes = test_ser.index[LOOKBACK:]
        # Trim to match sequence length (horizon offset)
        test_datetimes = test_datetimes[: len(y_test_mw)]
    except Exception:
        test_datetimes = None

    # ---------------------------------------------
    # Stage 2: Naive Baseline
    # ---------------------------------------------
    print("\n-- Stage 2: Naive Persistence Baseline --------------------")
    baseline = NaiveBaseline(lookback=LOOKBACK)
    baseline_pred_scaled = baseline.predict(X_test)

    # ---------------------------------------------
    # Stage 3: Build & Train LSTM
    # ---------------------------------------------
    print("\n-- Stage 3: LSTM Training ----------------------------------")
    model = build_lstm(lookback=LOOKBACK)
    model.summary()
    history = train_lstm(
        model, X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
    )

    # ---------------------------------------------
    # Stage 4: Generate Predictions
    # ---------------------------------------------
    print("\n-- Stage 4: Generating Predictions ------------------------")
    lstm_pred_scaled = model.predict(X_test, verbose=0).flatten()
    print(f"[main] Predictions generated for {len(lstm_pred_scaled):,} test samples.")

    # ---------------------------------------------
    # Stage 5: Inverse Scale -> MW Units
    # ---------------------------------------------
    print("\n-- Stage 5: Inverse Scaling --------------------------------")
    baseline_mw, lstm_mw = inverse_scale_predictions(
        baseline_pred_scaled, lstm_pred_scaled, scaler
    )

    # ---------------------------------------------
    # Stage 6: Evaluation Metrics
    # ---------------------------------------------
    print("\n-- Stage 6: Metrics ----------------------------------------")
    metrics = evaluate_all(y_test_mw, baseline_mw, lstm_mw)

    # ---------------------------------------------
    # Stage 7: Plots
    # ---------------------------------------------
    print("\n-- Stage 7: Plotting ---------------------------------------")
    plot_training_loss(history, save=True)
    plot_forecast_comparison(
        y_true=y_test_mw,
        baseline_preds=baseline_mw,
        lstm_preds=lstm_mw,
        metrics=metrics,
        datetimes=test_datetimes,
        n_plot=N_PLOT,
        save=True,
    )

    # ---------------------------------------------
    # Stage 8: Summary
    # ---------------------------------------------
    print_summary_table(metrics)

    # ---------------------------------------------
    # Stage 9: Future Out-of-Sample Forecast
    # ---------------------------------------------
    print("\n-- Stage 9: Future Forecast --------------------------------")

    # Build the full scaled series (train + test concatenated) to seed the window
    from data_loader import scale
    train_scaled_full = scale(train_ser, scaler)
    test_scaled_full  = scale(test_ser,  scaler)
    full_scaled = np.concatenate([train_scaled_full, test_scaled_full])

    future_mw, _ = forecast_future(
        model=model,
        full_series_scaled=full_scaled,
        scaler=scaler,
        lookback=LOOKBACK,
        steps=FUTURE_STEPS,
    )

    plot_future_forecast(
        full_series=full_series,
        future_preds_mw=future_mw,
        steps=FUTURE_STEPS,
        context_hours=CONTEXT_HOURS,
        save=True,
    )

    print("Pipeline complete.  Plots saved to:  electricity_forecasting/plots/")
    print("  - plots/training_loss.png")
    print("  - plots/forecast_comparison.png")
    print("  - plots/future_forecast.png  <-- NEW: 7-day out-of-sample forecast\n")


if __name__ == "__main__":
    main()
