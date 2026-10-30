"""
data_loader.py
--------------
Handles all data ingestion, preprocessing, scaling, and windowing for the
Electricity Demand Forecasting pipeline.

Key responsibilities:
    1. Load data from the data/ directory (CSV or Excel) with auto-detection,
       or fall back to Kaggle download via kagglehub.
    2. Auto-detect Datetime and Demand columns regardless of naming convention.
    3. Sort chronologically and fill missing values.
    4. Split into train/test sets (80/20) with NO data leakage.
    5. Fit MinMaxScaler exclusively on the training partition.
    6. Generate sliding-window sequences for LSTM input.

Supported local files (checked in priority order):
    data/hourlyLoadDataIndia.xlsx  ? Indian national hourly demand (2019-2024)
    data/electricity.csv           ? AEP / any hourly MW CSV
    data/<any>.xlsx or <any>.csv   ? generic fallback scan

Column auto-detection rules:
    Datetime  -> column whose name contains "datetime", "date", or "time"
    Demand    -> column whose name contains "national", "demand", "mw", "load",
                "consumption", or "energy" (first match wins)
              -> if no keyword matches, the first non-datetime numeric column is used
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR   = os.path.join(os.path.dirname(__file__), "..", "data")
TRAIN_RATIO = 0.80
LOOKBACK    = 24        # hours of history fed to the LSTM
HORIZON     = 1         # steps ahead to forecast

# Preferred local files, checked in order
_PREFERRED_FILES = [
    "hourlyLoadDataIndia.csv",    # Pre-exported CSV for blazing fast loading (<0.1s)
    "hourlyLoadDataIndia.xlsx",   # Indian national demand Excel (fallback)
    "electricity.csv",            # AEP / generic MW CSV (secondary)
]

# Keywords for auto-detecting the demand column (checked in order)
_DEMAND_KEYWORDS = ["national", "demand", "mw", "load", "consumption", "energy"]


# ---------------------------------------------------------------------------
# 1. Local File Discovery & Loading
# ---------------------------------------------------------------------------

def _read_file(filepath: str) -> pd.DataFrame:
    """
    Read a CSV or Excel file into a DataFrame based on its extension.

    Parameters
    ----------
    filepath : str  - absolute path to a .csv or .xlsx / .xls file.

    Returns
    -------
    pd.DataFrame
    """
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".csv":
        df = pd.read_csv(filepath)
    elif ext in (".xlsx", ".xls"):
        df = pd.read_excel(filepath, engine="openpyxl")
    else:
        raise ValueError(f"[data_loader] Unsupported file format: '{ext}'  ({filepath})")
    print(f"[data_loader] Loaded '{os.path.basename(filepath)}'  ->  {len(df):,} rows, {df.shape[1]} columns")
    return df


def _scan_data_dir() -> str | None:
    """
    Scan data/ for any CSV or Excel file and return the first one found.
    Used as a last-resort local fallback after preferred files are not found.
    """
    if not os.path.isdir(DATA_DIR):
        return None
    for fname in sorted(os.listdir(DATA_DIR)):
        if fname.lower().endswith((".csv", ".xlsx", ".xls")):
            return os.path.join(DATA_DIR, fname)
    return None


def _load_local() -> pd.DataFrame | None:
    """
    Attempt to load a dataset from the local data/ directory.

    Priority:
        1. Files listed in _PREFERRED_FILES  (in order)
        2. First CSV / Excel file found in data/  (alphabetical scan)

    Returns
    -------
    pd.DataFrame or None
    """
    # -- Priority files --------------------------------------------------------
    for fname in _PREFERRED_FILES:
        fpath = os.path.join(DATA_DIR, fname)
        if os.path.exists(fpath):
            print(f"[data_loader] Found preferred local file: {fpath}")
            return _read_file(fpath)

    # -- Generic scan ---------------------------------------------------------
    fpath = _scan_data_dir()
    if fpath:
        print(f"[data_loader] Found local file via scan: {fpath}")
        return _read_file(fpath)

    return None


def _load_from_kaggle() -> pd.DataFrame | None:
    """
    Attempt to download the AEP hourly energy consumption dataset
    from Kaggle using kagglehub.  Falls back gracefully if unavailable.
    """
    try:
        import kagglehub
        print("[data_loader] Downloading dataset via kagglehub ?")
        path = kagglehub.dataset_download("robikscube/hourly-energy-consumption")
        candidates = [
            os.path.join(path, "AEP_hourly.csv"),
            os.path.join(path, "PJME_hourly.csv"),
        ]
        for candidate in candidates:
            if os.path.exists(candidate):
                print(f"[data_loader] Using file: {candidate}")
                return pd.read_csv(candidate)
        csvs = [f for f in os.listdir(path) if f.endswith(".csv")]
        if csvs:
            fp = os.path.join(path, csvs[0])
            print(f"[data_loader] Using file: {fp}")
            return pd.read_csv(fp)
    except Exception as exc:
        print(f"[data_loader] kagglehub download failed ({exc}).")
    return None


def load_raw_data() -> pd.DataFrame:
    """
    Load the raw hourly electricity demand dataset.

    Priority order:
        1. data/hourlyLoadDataIndia.xlsx  (Indian national demand)
        2. data/electricity.csv           (AEP / generic CSV)
        3. Any CSV or Excel in data/      (generic scan)
        4. Kaggle download via kagglehub  (requires API key)

    Returns
    -------
    pd.DataFrame

    Raises
    ------
    FileNotFoundError
        If no data source is found.
    """
    # -- 1-3. Local files ------------------------------------------------------
    df = _load_local()
    if df is not None:
        return df

    # -- 4. Kaggle remote download ---------------------------------------------
    df = _load_from_kaggle()
    if df is not None:
        os.makedirs(DATA_DIR, exist_ok=True)
        out = os.path.join(DATA_DIR, "electricity.csv")
        df.to_csv(out, index=False)
        print(f"[data_loader] Dataset cached at: {out}")
        return df

    # -- Nothing worked --------------------------------------------------------
    raise FileNotFoundError(
        "\n[ERROR] No dataset found.\n"
        "Please do ONE of the following:\n"
        "  Option A - Place your file in:  electricity_forecasting/data/\n"
        "             Supported formats: .csv, .xlsx, .xls\n"
        "             Recommended: data/hourlyLoadDataIndia.xlsx  (already supported)\n"
        "  Option B - Install kagglehub and set your Kaggle API credentials:\n"
        "             pip install kagglehub\n"
        "             $env:KAGGLE_USERNAME = 'your_username'   # PowerShell\n"
        "             $env:KAGGLE_KEY      = 'your_api_key'\n"
        "  Option C - Download AEP_hourly.csv manually from:\n"
        "             https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption\n"
        "             and save it to:  data/electricity.csv\n"
    )


# ---------------------------------------------------------------------------
# 2. Smart Column Detection
# ---------------------------------------------------------------------------

def _detect_datetime_column(df: pd.DataFrame) -> str:
    """
    Return the name of the datetime column using keyword matching.

    Matching order (case-insensitive):
        1. Column already parsed as datetime dtype by pandas
        2. Column name containing 'datetime', 'date', or 'time'

    Raises
    ------
    ValueError  if no datetime column can be found.
    """
    # Already a datetime dtype (e.g. when read from Excel with parse_dates)
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col

    # Keyword search in column names
    keywords = ["datetime", "date", "time"]
    for keyword in keywords:
        for col in df.columns:
            if keyword in col.lower():
                return col

    raise ValueError(
        f"[data_loader] Cannot detect a datetime column.\n"
        f"  Columns found: {df.columns.tolist()}\n"
        f"  Hint: Rename your timestamp column to 'Datetime' or 'datetime'."
    )


def _detect_demand_column(df: pd.DataFrame, dt_col: str) -> str:
    """
    Return the name of the electricity demand column using keyword matching.

    Matching order (case-insensitive):
        1. Column name containing any keyword in _DEMAND_KEYWORDS
        2. First numeric column that is not the datetime column

    Parameters
    ----------
    df     : Raw DataFrame.
    dt_col : Name of the already-detected datetime column (excluded).

    Raises
    ------
    ValueError  if no suitable demand column can be found.
    """
    non_dt = [c for c in df.columns if c != dt_col]

    # Keyword match
    for keyword in _DEMAND_KEYWORDS:
        for col in non_dt:
            if keyword in col.lower():
                return col

    # Fallback: first numeric column
    for col in non_dt:
        if pd.api.types.is_numeric_dtype(df[col]):
            return col

    raise ValueError(
        f"[data_loader] Cannot detect a demand/consumption column.\n"
        f"  Non-datetime columns found: {non_dt}\n"
        f"  Hint: Ensure your demand column name contains one of: {_DEMAND_KEYWORDS}"
    )


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Auto-detect and rename Datetime & Demand columns to canonical names
    ('Datetime', 'MW') for uniform downstream processing.

    Also logs which columns were selected so the user can verify.
    """
    df = df.copy()
    df.columns = df.columns.str.strip()     # remove accidental whitespace

    dt_col  = _detect_datetime_column(df)
    mw_col  = _detect_demand_column(df, dt_col)

    print(f"[data_loader] Datetime column  detected -> '{dt_col}'")
    print(f"[data_loader] Demand column    detected -> '{mw_col}'")

    result = df[[dt_col, mw_col]].copy()
    result.columns = ["Datetime", "MW"]
    return result


# ---------------------------------------------------------------------------
# 3. Preprocessing
# ---------------------------------------------------------------------------

def preprocess(df: pd.DataFrame) -> pd.Series:
    """
    Coerce types, sort chronologically, deduplicate, and fill gaps.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataframe returned by load_raw_data().

    Returns
    -------
    pd.Series
        Clean hourly MW time series indexed by Datetime.
    """
    df = _normalise_columns(df)

    # Parse datetimes; coerce bad values to NaT
    df["Datetime"] = pd.to_datetime(df["Datetime"], errors="coerce")
    n_bad_dt = df["Datetime"].isna().sum()
    if n_bad_dt > 0:
        print(f"[data_loader] WARNING: {n_bad_dt} unparseable datetime rows dropped.")
    df.dropna(subset=["Datetime"], inplace=True)

    # Chronological sort
    df.sort_values("Datetime", inplace=True)

    # Remove duplicate timestamps (keep last)
    n_dupes = df["Datetime"].duplicated().sum()
    if n_dupes > 0:
        print(f"[data_loader] WARNING: {n_dupes} duplicate timestamps removed (kept last).")
        df.drop_duplicates(subset=["Datetime"], keep="last", inplace=True)

    df.set_index("Datetime", inplace=True)

    # Ensure numeric demand values
    df["MW"] = pd.to_numeric(df["MW"], errors="coerce")

    # Fill missing values: forward-fill first, then backward-fill any leading NaNs
    df["MW"] = df["MW"].ffill().bfill()

    n_still_null = df["MW"].isna().sum()
    if n_still_null > 0:
        raise ValueError(
            f"[data_loader] {n_still_null} demand values remain null after filling. "
            "Check your dataset for large contiguous gaps."
        )

    print(
        f"[data_loader] Preprocessed: {len(df):,} hourly records  "
        f"| range: {df.index.min().date()} -> {df.index.max().date()}  "
        f"| demand: {df['MW'].min():.0f} - {df['MW'].max():.0f} MW"
    )
    return df["MW"]


# ---------------------------------------------------------------------------
# 4. Train / Test Split  (strictly chronological - NO shuffling)
# ---------------------------------------------------------------------------

def train_test_split_ts(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """
    Split time series chronologically into training and test partitions.

    Parameters
    ----------
    series : pd.Series
        Clean hourly MW series.

    Returns
    -------
    train, test : pd.Series
        80 % / 20 % chronological partitions.
    """
    n = len(series)
    split_idx = int(n * TRAIN_RATIO)
    train = series.iloc[:split_idx]
    test  = series.iloc[split_idx:]
    print(
        f"[data_loader] Train: {len(train):,} samples  "
        f"({train.index.min().date()} -> {train.index.max().date()})  |  "
        f"Test: {len(test):,} samples  "
        f"({test.index.min().date()} -> {test.index.max().date()})"
    )
    return train, test


# ---------------------------------------------------------------------------
# 5. Scaling  (fit ONLY on train - zero leakage guarantee)
# ---------------------------------------------------------------------------

def fit_scaler(train: pd.Series) -> MinMaxScaler:
    """
    Fit a MinMaxScaler on the training partition only.

    Parameters
    ----------
    train : pd.Series

    Returns
    -------
    MinMaxScaler - fitted, to be reused for test transform and inverse transform.
    """
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train.values.reshape(-1, 1))
    return scaler


def scale(series: pd.Series, scaler: MinMaxScaler) -> np.ndarray:
    """Transform a series using a pre-fitted scaler."""
    return scaler.transform(series.values.reshape(-1, 1)).flatten()


def inverse_scale(array: np.ndarray, scaler: MinMaxScaler) -> np.ndarray:
    """Invert scaling to recover original MW units."""
    return scaler.inverse_transform(array.reshape(-1, 1)).flatten()


def extract_time_features(datetimes: pd.DatetimeIndex) -> np.ndarray:
    """
    Generate deterministic cyclical time features from datetime index:
      - sin_hour, cos_hour (24-hour diurnal cycle)
      - sin_dow, cos_dow   (7-day weekly cycle)
    Returns: np.ndarray of shape (len(datetimes), 4)
    """
    hours = datetimes.hour.values
    dows = datetimes.dayofweek.values

    sin_h = np.sin(2 * np.pi * hours / 24).astype(np.float32)
    cos_h = np.cos(2 * np.pi * hours / 24).astype(np.float32)
    sin_d = np.sin(2 * np.pi * dows / 7).astype(np.float32)
    cos_d = np.cos(2 * np.pi * dows / 7).astype(np.float32)

    return np.column_stack([sin_h, cos_h, sin_d, cos_d])


def build_feature_matrix(series_scaled: np.ndarray, datetimes: pd.DatetimeIndex) -> np.ndarray:
    """
    Combine scaled demand (column 0) with 4 cyclical time signals (columns 1-4).
    Returns: np.ndarray of shape (N, 5)
    """
    time_feats = extract_time_features(datetimes)
    return np.column_stack([series_scaled.astype(np.float32), time_feats])


# ---------------------------------------------------------------------------
# 6. Sliding-Window Sequence Generation
# ---------------------------------------------------------------------------

def create_sequences(
    feature_matrix: np.ndarray,
    lookback: int = LOOKBACK,
    horizon: int = HORIZON,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build overlapping (X, y) pairs from a 2-D multi-feature array of shape (N, 5).

    Each X window has shape (lookback, 5) and each y is the scaled demand (column 0)
    `horizon` steps ahead.

    Parameters
    ----------
    feature_matrix : 2-D numpy array of shape (N, 5).
    lookback       : number of past timesteps used as features (default 24).
    horizon        : forecast horizon (default = 1 step ahead).

    Returns
    -------
    X : np.ndarray, shape (n_samples, lookback, 5)
    y : np.ndarray, shape (n_samples,)
    """
    if feature_matrix.ndim == 1:
        feature_matrix = feature_matrix.reshape(-1, 1)

    X, y = [], []
    end = len(feature_matrix) - horizon
    for i in range(lookback, end + 1):
        X.append(feature_matrix[i - lookback : i])
        y.append(feature_matrix[i + horizon - 1, 0])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ---------------------------------------------------------------------------
# 7. Public Helper - full pipeline in one call
# ---------------------------------------------------------------------------

def prepare_data() -> dict:
    """
    End-to-end data preparation convenience function with multi-feature engineering.

    Returns
    -------
    dict with keys:
        "train_series"   - raw MW training pd.Series
        "test_series"    - raw MW test pd.Series
        "scaler"         - MinMaxScaler fitted on train only
        "train_features" - multi-feature array for train (n, 5)
        "test_features"  - multi-feature array for test (m, 5)
        "full_features"  - concatenated multi-feature array (n+m, 5)
        "X_train"        - LSTM training sequences  (n, lookback, 5)
        "y_train"        - LSTM training targets    (n,)
        "X_test"         - LSTM test sequences      (m, lookback, 5)
        "y_test"         - LSTM test targets scaled (m,)
        "y_test_raw"     - test targets in original MW units (m,)
    """
    raw_df = load_raw_data()
    series = preprocess(raw_df)
    train_series, test_series = train_test_split_ts(series)

    scaler       = fit_scaler(train_series)
    train_scaled = scale(train_series, scaler)
    test_scaled  = scale(test_series,  scaler)

    train_features = build_feature_matrix(train_scaled, train_series.index)
    test_features  = build_feature_matrix(test_scaled, test_series.index)
    full_features  = np.vstack([train_features, test_features])

    X_train, y_train = create_sequences(train_features)
    X_test,  y_test  = create_sequences(test_features)

    # Ground-truth in original MW for plots and metrics
    y_test_raw = inverse_scale(y_test, scaler)

    print(
        f"[data_loader] Multi-feature sequence shapes -> "
        f"X_train: {X_train.shape}  X_test: {X_test.shape}"
    )

    return {
        "train_series":   train_series,
        "test_series":    test_series,
        "scaler":         scaler,
        "train_features": train_features,
        "test_features":  test_features,
        "full_features":  full_features,
        "X_train":        X_train,
        "y_train":        y_train,
        "X_test":         X_test,
        "y_test":         y_test,
        "y_test_raw":     y_test_raw,
    }
