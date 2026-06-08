"""Descarga de datos históricos del MERVAL desde Yahoo Finance."""

import time
import yfinance as yf
import pandas as pd
from pathlib import Path


TICKERS = ["GGAL.BA", "BMA.BA", "YPFD.BA", "PAMP.BA", "BBAR.BA", "ALUA.BA", "^MERV"]
START = "2015-01-01"
END   = "2025-12-31"


def download_data(start: str = START, end: str = END) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Descarga precios de cierre y volumen para todos los tickers del MERVAL.

    Args:
        start: Fecha de inicio en formato YYYY-MM-DD.
        end:   Fecha de fin en formato YYYY-MM-DD.

    Returns:
        Tupla (close, volume) con DataFrames indexados por fecha.
    """
    print(f"Descargando {len(TICKERS)} tickers desde {start} hasta {end}...")
    raw = yf.download(TICKERS, start=start, end=end, auto_adjust=True, progress=False)

    close  = raw["Close"].copy()
    volume = raw["Volume"].copy()

    # Rellenar gaps de feriados argentinos (máximo 3 días consecutivos)
    close  = close.ffill(limit=3)
    volume = volume.ffill(limit=3)

    # Eliminar filas donde todos los valores sean NaN
    close  = close.dropna(how="all")
    volume = volume.dropna(how="all")

    Path("data/raw").mkdir(parents=True, exist_ok=True)
    close.to_csv("data/raw/close.csv")
    volume.to_csv("data/raw/volume.csv")

    print(f"Guardado: {len(close)} días x {len(close.columns)} tickers")
    return close, volume


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Carga los CSV ya descargados desde data/raw/."""
    close  = pd.read_csv("data/raw/close.csv",  index_col=0, parse_dates=True)
    volume = pd.read_csv("data/raw/volume.csv", index_col=0, parse_dates=True)
    return close, volume


if __name__ == "__main__":
    close, volume = download_data()
    print(close.tail(3))
