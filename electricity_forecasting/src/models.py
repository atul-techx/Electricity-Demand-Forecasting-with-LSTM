"""
models.py
---------
Contains two forecasting models:

    1. NaiveBaseline  - Persistence / last-value forecast:  ?_t = y_{t-1}
    2. build_lstm     - Two-layer stacked LSTM with Dropout + Dense output.

Both models operate on the scaled domain; inverse-scaling is handled
in evaluate.py to keep responsibilities cleanly separated.
"""

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


# ---------------------------------------------------------------------------
# 1. Naive Persistence Baseline
# ---------------------------------------------------------------------------

class NaiveBaseline:
    """
    Persistence (last-value) baseline forecaster.

    For a horizon of 1 step, the prediction at time t is simply the
    observation at time t-1  ->  ?_t = y_{t-1}.

    Parameters
    ----------
    lookback : int
        The same lookback window used for LSTM sequence construction.
        The last value in each X window is used as the naive prediction.
    """

    def __init__(self, lookback: int = 24) -> None:
        self.lookback = lookback

    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, lookback, 1)
            LSTM-formatted test sequences (scaled).

        Returns
        -------
        np.ndarray, shape (n_samples,)
            Naive predictions: the last timestep of every window.
        """
        # X[:, -1, 0]  ->  last hour of each input window
        return X[:, -1, 0]


# ---------------------------------------------------------------------------
# 2. Stacked LSTM Model
# ---------------------------------------------------------------------------

def build_lstm(
    lookback: int = 24,
    lstm_units_1: int = 64,
    lstm_units_2: int = 32,
    dropout_rate: float = 0.20,
    learning_rate: float = 1e-3,
) -> keras.Sequential:
    """
    Build and compile a two-layer stacked LSTM model.

    Architecture
    ------------
    Input  ->  LSTM(64, return_sequences=True)  ->  Dropout(0.2)
           ->  LSTM(32, return_sequences=False) ->  Dropout(0.2)
           ->  Dense(1)

    Parameters
    ----------
    lookback        : number of timesteps in each input sequence.
    lstm_units_1    : units in the first LSTM layer.
    lstm_units_2    : units in the second LSTM layer.
    dropout_rate    : fraction of units to drop for regularisation.
    learning_rate   : Adam optimiser learning rate.

    Returns
    -------
    keras.Sequential
        Compiled model ready for training.
    """
    model = keras.Sequential(
        [
            # -- First LSTM layer ------------------------------------------
            # return_sequences=True passes the full output sequence to
            # the second LSTM layer.
            layers.LSTM(
                lstm_units_1,
                return_sequences=True,
                input_shape=(lookback, 1),
                name="lstm_1",
            ),
            layers.Dropout(dropout_rate, name="dropout_1"),

            # -- Second LSTM layer -----------------------------------------
            # return_sequences=False collapses to a single context vector.
            layers.LSTM(lstm_units_2, return_sequences=False, name="lstm_2"),
            layers.Dropout(dropout_rate, name="dropout_2"),

            # -- Dense output: single MW value -----------------------------
            layers.Dense(1, name="output"),
        ],
        name="stacked_lstm",
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mean_squared_error",
        metrics=["mae"],
    )

    return model


# ---------------------------------------------------------------------------
# 3. Training Helper
# ---------------------------------------------------------------------------

def train_lstm(
    model: keras.Sequential,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 20,
    batch_size: int = 64,
    validation_split: float = 0.10,
) -> keras.callbacks.History:
    """
    Train the LSTM model with early stopping.

    Parameters
    ----------
    model            : compiled Keras Sequential model.
    X_train          : training sequences, shape (n, lookback, 1).
    y_train          : training targets, shape (n,).
    epochs           : maximum number of training epochs.
    batch_size       : mini-batch size.
    validation_split : fraction of training data used for validation.

    Returns
    -------
    keras.callbacks.History
        Training history object (loss curves).
    """
    early_stop = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=5,            # stop if val_loss doesn't improve for 5 epochs
        restore_best_weights=True,
        verbose=1,
    )

    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-6,
        verbose=1,
    )

    print(f"\n[models] Training LSTM for up to {epochs} epochs ?")
    history = model.fit(
        X_train,
        y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=validation_split,
        callbacks=[early_stop, reduce_lr],
        verbose=1,
    )
    print("[models] Training complete.")
    return history
