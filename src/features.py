"""Construcción de features y ventanas temporales para el LSTM Autoencoder."""

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from src.data import load_data, TICKERS


WINDOW      = 30    # días por ventana temporal
TRAIN_END   = "2017-12-31"
VAL_END     = "2018-12-31"


def build_features(close: pd.DataFrame, volume: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Construye 8 features para un ticker dado.

    Args:
        close:  DataFrame de precios de cierre.
        volume: DataFrame de volúmenes.
        ticker: Ticker a procesar (ej: 'GGAL.BA').

    Returns:
        DataFrame con 8 columnas de features, sin NaN.
    """
    c = close[ticker]
    v = volume[ticker]
    m = close["^MERV"]
    m_ret = m.pct_change()

    df = pd.DataFrame(index=close.index)
    df["ret_1d"]      = c.pct_change()
    df["log_ret"]     = np.log(c / c.shift(1))
    df["vol_5d"]      = df["ret_1d"].rolling(5).std()
    df["vol_20d"]     = df["ret_1d"].rolling(20).std()
    df["vol_ratio"]   = df["vol_5d"] / (m_ret.rolling(5).std() + 1e-8)
    df["corr_30d"]    = df["ret_1d"].rolling(30).corr(m_ret)
    df["zscore"]      = (c - c.rolling(90).mean()) / (c.rolling(90).std() + 1e-8)
    df["vol_ratio_v"] = v / (v.rolling(20).mean() + 1)

    return df.dropna()


def make_windows(df: pd.DataFrame, window: int = WINDOW) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """
    Convierte un DataFrame en ventanas deslizantes para LSTM.

    Args:
        df:     DataFrame de features ya escaladas.
        window: Tamaño de la ventana en días.

    Returns:
        Tupla (X, dates) donde X tiene shape (N, window, n_features)
        y dates contiene la fecha del último día de cada ventana.
    """
    X, dates = [], []
    for i in range(len(df) - window):
        X.append(df.values[i : i + window])
        dates.append(df.index[i + window - 1])
    return np.array(X), pd.DatetimeIndex(dates)


def split_and_scale(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray,
           pd.DatetimeIndex, pd.DatetimeIndex, pd.DatetimeIndex,
           StandardScaler]:
    """
    Divide en train/val/test y escala con StandardScaler ajustado solo en train.

    Returns:
        X_train, X_val, X_test, dates_train, dates_val, dates_test, scaler
    """
    train_df = df[df.index <= TRAIN_END]
    val_df   = df[(df.index > TRAIN_END) & (df.index <= VAL_END)]
    test_df  = df[df.index > VAL_END]

    # Ajustar scaler SOLO en train
    scaler = StandardScaler()

    def scale(subset: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        flat = subset.values
        if fit:
            scaler.fit(flat)
        return pd.DataFrame(scaler.transform(flat), index=subset.index, columns=subset.columns)

    train_s = scale(train_df, fit=True)
    val_s   = scale(val_df)
    test_s  = scale(test_df)

    X_train, d_train = make_windows(train_s)
    X_val,   d_val   = make_windows(val_s)
    X_test,  d_test  = make_windows(test_s)

    return X_train, X_val, X_test, d_train, d_val, d_test, scaler


if __name__ == "__main__":
    close, volume = load_data()

    # Procesar GGAL.BA como ejemplo
    ticker = "GGAL.BA"
    df = build_features(close, volume, ticker)

    Path("data/processed").mkdir(parents=True, exist_ok=True)
    df.to_csv(f"data/processed/features_{ticker.replace('.', '_')}.csv")

    X_train, X_val, X_test, d_train, d_val, d_test, scaler = split_and_scale(df)
    print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")
