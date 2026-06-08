# MERVAL Anomaly Detection — Plan de Implementación

Proyecto de detección de anomalías en precios del MERVAL argentino usando LSTM Autoencoder.
Trabajo integrador de Inteligencia Artificial.

Implementar las 5 fases en orden. Crear cada archivo exactamente como se especifica.
No avanzar a la siguiente fase hasta que la actual corra sin errores.

---

## Fase 1 — Entorno y dependencias

**Objetivo**: dejar el entorno listo para correr el proyecto.

### 1.1 Crear estructura de directorios

```bash
mkdir -p merval_anomaly/{data/raw,data/processed,models,notebooks,results,src}
cd merval_anomaly
```

### 1.2 Crear `requirements.txt`

```
yfinance==0.2.38
pandas==2.2.0
numpy==1.26.4
scikit-learn==1.4.0
tensorflow==2.15.0
matplotlib==3.8.2
seaborn==0.13.2
```

### 1.3 Instalar dependencias

```bash
pip install -r requirements.txt
```

### 1.4 Verificar instalación

```bash
python -c "import yfinance, tensorflow, sklearn; print('OK')"
```

**Resultado esperado**: `OK` sin errores.

---

## Fase 2 — Descarga de datos

**Objetivo**: descargar precios históricos del MERVAL desde Yahoo Finance y guardarlos en `data/raw/`.

### 2.1 Crear `src/data.py`

```python
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
```

### 2.2 Ejecutar

```bash
python src/data.py
```

**Resultado esperado**: `data/raw/close.csv` y `data/raw/volume.csv` con ~2500 filas cada uno.

---

## Fase 3 — Ingeniería de features

**Objetivo**: construir 8 features por ticker y generar ventanas temporales de 30 días para el modelo.

### 3.1 Crear `src/features.py`

```python
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
    n_feat = df.shape[1]

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
```

### 3.2 Ejecutar

```bash
python src/features.py
```

**Resultado esperado**: `data/processed/features_GGAL_BA.csv` y línea con shapes tipo
`Train: (700, 30, 8) | Val: (220, 30, 8) | Test: (1500, 30, 8)`.

---

## Fase 4 — Modelo LSTM Autoencoder

**Objetivo**: definir la arquitectura y entrenar el modelo solo con datos normales (train 2015–2017).

### 4.1 Crear `src/model.py`

```python
"""Arquitectura del LSTM Autoencoder para detección de anomalías."""

from tensorflow.keras import Sequential, layers


def build_lstm_ae(time_steps: int = 30, n_features: int = 8) -> Sequential:
    """
    Construye el LSTM Autoencoder.

    Arquitectura:
        Encoder: LSTM(64) → Dropout(0.2)
        Bottleneck: RepeatVector(time_steps)
        Decoder: LSTM(64, return_seq=True) → Dropout(0.2) → Dense(n_features)

    Args:
        time_steps: Longitud de la ventana temporal.
        n_features: Cantidad de features por paso de tiempo.

    Returns:
        Modelo Keras compilado con loss MAE y optimizer Adam.
    """
    model = Sequential([
        layers.LSTM(64, input_shape=(time_steps, n_features),
                    return_sequences=False, name="encoder"),
        layers.Dropout(0.2),
        layers.RepeatVector(time_steps, name="bottleneck"),
        layers.LSTM(64, return_sequences=True, name="decoder"),
        layers.Dropout(0.2),
        layers.TimeDistributed(layers.Dense(n_features), name="output"),
    ], name="lstm_autoencoder")

    model.compile(optimizer="adam", loss="mae")
    return model


if __name__ == "__main__":
    model = build_lstm_ae()
    model.summary()
```

### 4.2 Crear `src/train.py`

```python
"""Entrenamiento del LSTM Autoencoder con datos normales (2015–2017)."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tensorflow.keras import callbacks

from src.model import build_lstm_ae
from src.features import build_features, split_and_scale
from src.data import load_data


TICKER  = "GGAL.BA"
EPOCHS  = 50
BATCH   = 32


def train(
    X_train: np.ndarray,
    X_val: np.ndarray,
    model_path: str = "models/best.keras",
    epochs: int = EPOCHS,
    batch_size: int = BATCH,
):
    """
    Entrena el autoencoder usando solo datos normales.

    Args:
        X_train:    Tensor de entrenamiento (N, 30, 8).
        X_val:      Tensor de validación (N, 30, 8).
        model_path: Ruta donde guardar el mejor modelo.
        epochs:     Máximo de épocas.
        batch_size: Tamaño del batch.

    Returns:
        Tupla (model, history).
    """
    Path("models").mkdir(exist_ok=True)

    model = build_lstm_ae(
        time_steps=X_train.shape[1],
        n_features=X_train.shape[2],
    )

    cbs = [
        callbacks.EarlyStopping(monitor="val_loss", patience=7,
                                restore_best_weights=True, verbose=1),
        callbacks.ModelCheckpoint(model_path, monitor="val_loss",
                                  save_best_only=True, verbose=0),
    ]

    history = model.fit(
        X_train, X_train,
        validation_data=(X_val, X_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=cbs,
        verbose=1,
    )
    return model, history


def plot_loss(history, save_path: str = "results/loss_curve.png") -> None:
    """Grafica la curva de pérdida de entrenamiento y validación."""
    Path("results").mkdir(exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(history.history["loss"],     label="Train loss")
    plt.plot(history.history["val_loss"], label="Val loss")
    plt.xlabel("Época")
    plt.ylabel("MAE")
    plt.title("Curva de pérdida — LSTM Autoencoder")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Guardado: {save_path}")


if __name__ == "__main__":
    close, volume = load_data()
    df = build_features(close, volume, TICKER)
    X_train, X_val, X_test, _, _, _, _ = split_and_scale(df)

    model, history = train(X_train, X_val)
    plot_loss(history)
    print(f"Mejor val_loss: {min(history.history['val_loss']):.6f}")
```

### 4.3 Ejecutar

```bash
python src/train.py
```

**Resultado esperado**: `models/best.keras` y `results/loss_curve.png`.
El val_loss debe decrecer y estabilizarse en las primeras 20–35 épocas.

---

## Fase 5 — Evaluación y visualización

**Objetivo**: calcular el error de reconstrucción sobre el test set, determinar el umbral 3σ,
detectar anomalías y verificar que coincidan con los eventos históricos conocidos.

### 5.1 Crear `src/evaluate.py`

```python
"""Evaluación del modelo: umbral 3σ, detección de anomalías y métricas."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path
from tensorflow.keras.models import load_model
from sklearn.metrics import precision_score, recall_score, f1_score

from src.features import build_features, split_and_scale
from src.data import load_data


TICKER = "GGAL.BA"

EVENTOS = {
    "2019-08-12": "PASO 2019\n(-38%)",
    "2020-03-13": "COVID-19",
    "2023-08-14": "PASO 2023",
    "2023-12-13": "Shock Milei",
}


def reconstruction_error(model, X: np.ndarray) -> np.ndarray:
    """
    Calcula el error de reconstrucción MAE por ventana.

    Args:
        model: Modelo Keras entrenado.
        X:     Tensor de entrada (N, 30, 8).

    Returns:
        Array 1D con el MAE de cada ventana.
    """
    X_hat = model.predict(X, verbose=0)
    return np.mean(np.abs(X - X_hat), axis=(1, 2))


def compute_threshold(train_errors: np.ndarray, sigma: float = 3.0) -> float:
    """
    Calcula el umbral de anomalía con la regla N-sigma.

    Args:
        train_errors: Errores de reconstrucción en datos de train.
        sigma:        Multiplicador de desviación estándar (default 3).

    Returns:
        Valor del umbral.
    """
    return float(train_errors.mean() + sigma * train_errors.std())


def plot_anomalias(
    errors_test: np.ndarray,
    dates_test: pd.DatetimeIndex,
    threshold: float,
    sigma: float,
    save_path: str = "results/anomalias.png",
) -> pd.DatetimeIndex:
    """
    Grafica el error de reconstrucción en el test set con el umbral y los eventos históricos.

    Returns:
        Fechas detectadas como anomalías.
    """
    Path("results").mkdir(exist_ok=True)

    anomaly_mask = errors_test > threshold
    anomaly_dates = dates_test[anomaly_mask]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates_test, errors_test, lw=0.8, color="steelblue",
            alpha=0.85, label="Error de reconstrucción")
    ax.fill_between(dates_test, errors_test, alpha=0.15, color="steelblue")
    ax.axhline(threshold, color="crimson", ls="--", lw=1.5,
               label=f"Umbral {sigma}σ = {threshold:.4f}")

    # Marcar anomalías detectadas
    ax.scatter(anomaly_dates,
               errors_test[anomaly_mask],
               color="crimson", s=18, zorder=5, label="Anomalía detectada")

    # Marcar eventos históricos conocidos
    ylim = ax.get_ylim()
    for fecha_str, label in EVENTOS.items():
        fecha = pd.Timestamp(fecha_str)
        if dates_test.min() <= fecha <= dates_test.max():
            ax.axvline(fecha, color="orange", lw=1.2, alpha=0.9, ls="-.")
            ax.text(fecha, ylim[1] * 0.92, label,
                    fontsize=7.5, ha="left", va="top",
                    rotation=90, color="darkorange")

    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.set_xlabel("Fecha")
    ax.set_ylabel("MAE de reconstrucción")
    ax.set_title(f"Anomalías detectadas — {TICKER} | Umbral {sigma}σ")
    ax.legend(loc="upper left", fontsize=9)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"Guardado: {save_path}")
    return anomaly_dates


def run_ablation_experiment(
    close: pd.DataFrame,
    volume: pd.DataFrame,
    model_path: str = "models/best.keras",
) -> None:
    """
    Experimento 3: compara el modelo con las 8 features vs solo retornos.
    Grafica los errores de reconstrucción de ambas versiones.
    """
    from src.train import train as train_model

    results = {}
    feature_sets = {
        "solo_retornos": ["ret_1d", "log_ret"],
        "8_features":    ["ret_1d", "log_ret", "vol_5d", "vol_20d",
                          "vol_ratio", "corr_30d", "zscore", "vol_ratio_v"],
    }

    for name, cols in feature_sets.items():
        df = build_features(close, volume, TICKER)[cols]
        X_tr, X_val, X_te, _, _, d_te, _ = split_and_scale(df)

        model, _ = train_model(
            X_tr, X_val,
            model_path=f"models/ablation_{name}.keras",
            epochs=50,
        )
        err = reconstruction_error(model, X_te)
        thr = compute_threshold(reconstruction_error(model, X_tr))
        results[name] = {"errors": err, "dates": d_te, "threshold": thr}
        print(f"[{name}] threshold={thr:.5f} | anomalías={( err > thr).sum()}")

    # Gráfico comparativo
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, (name, data) in zip(axes, results.items()):
        ax.plot(data["dates"], data["errors"], lw=0.8)
        ax.axhline(data["threshold"], color="crimson", ls="--", lw=1.2,
                   label=f"Umbral 3σ")
        ax.set_title(f"Ablation — {name}")
        ax.set_ylabel("MAE")
        ax.legend()
    plt.tight_layout()
    plt.savefig("results/ablation_study.png", dpi=150)
    plt.close()
    print("Guardado: results/ablation_study.png")


def sigma_sweep(
    train_errors: np.ndarray,
    test_errors: np.ndarray,
    dates_test: pd.DatetimeIndex,
) -> pd.DataFrame:
    """
    Experimento 1: barre sigma en {2, 3, 4} y reporta cantidad de anomalías por valor.

    Returns:
        DataFrame con sigma, threshold y total de anomalías detectadas.
    """
    rows = []
    for s in [2.0, 3.0, 4.0]:
        thr = compute_threshold(train_errors, sigma=s)
        n   = int((test_errors > thr).sum())
        rows.append({"sigma": s, "threshold": round(thr, 5), "anomalias_detectadas": n})
        print(f"σ={s} → umbral={thr:.5f} | anomalías={n}")
    return pd.DataFrame(rows)


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)

    close, volume = load_data()
    df = build_features(close, volume, TICKER)
    X_train, X_val, X_test, d_train, d_val, d_test, scaler = split_and_scale(df)

    model = load_model("models/best.keras")

    train_errors = reconstruction_error(model, X_train)
    test_errors  = reconstruction_error(model, X_test)

    # ── Experimento 1: variación de sigma ──────────────────────────────────
    print("\n── Experimento 1: variación de sigma ──")
    sweep_df = sigma_sweep(train_errors, test_errors, d_test)
    sweep_df.to_csv("results/sigma_sweep.csv", index=False)

    # ── Experimento 2: gráfico principal con sigma=3 ───────────────────────
    print("\n── Experimento 2: detección con sigma=3 ──")
    threshold = compute_threshold(train_errors, sigma=3.0)
    anomaly_dates = plot_anomalias(test_errors, d_test, threshold, sigma=3.0)
    print(f"Anomalías detectadas: {len(anomaly_dates)}")
    print(anomaly_dates.strftime("%Y-%m-%d").tolist()[:10], "...")

    # ── Experimento 3: ablation study ─────────────────────────────────────
    print("\n── Experimento 3: ablation study ──")
    run_ablation_experiment(close, volume)

    print("\nPipeline completo. Revisar carpeta results/")
```

### 5.2 Ejecutar evaluación completa

```bash
python src/evaluate.py
```

**Resultado esperado**: tres archivos en `results/`:
- `anomalias.png` — serie del error con umbral y eventos marcados
- `sigma_sweep.csv` — tabla con anomalías detectadas para σ ∈ {2, 3, 4}
- `ablation_study.png` — comparativo 2 features vs 8 features

---

## Ejecución completa del pipeline

Una vez que todas las fases están implementadas:

```bash
python src/data.py && \
python src/features.py && \
python src/train.py && \
python src/evaluate.py
```

---

## Convenciones

- Type hints en todas las funciones públicas
- Docstrings en español con sección Args y Returns
- Scaler: `fit` solo en train, `transform` en val y test
- Modelos en `models/` con nombre descriptivo
- Gráficos en `results/` como PNG a 150 DPI
- Feriados argentinos: `ffill(limit=3)`, nunca rellenar más de 3 días
