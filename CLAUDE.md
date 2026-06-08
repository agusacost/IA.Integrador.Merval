# MERVAL Anomaly Detection — LSTM Autoencoder

Proyecto de detección de anomalías en precios de acciones del MERVAL argentino usando un LSTM Autoencoder. Trabajo integrador de Inteligencia Artificial.

## Comandos

```bash
# Instalar dependencias
pip install -r requirements.txt

# Descargar datos
python src/data.py

# Construir features
python src/features.py

# Entrenar modelo
python src/train.py

# Evaluar y graficar resultados
python src/evaluate.py

# Ejecutar todo el pipeline
python src/data.py && python src/features.py && python src/train.py && python src/evaluate.py
```

## Estructura del proyecto

```
merval_anomaly/
├── CLAUDE.md
├── requirements.txt
├── data/
│   ├── raw/           # CSV de yfinance (close.csv, volume.csv)
│   └── processed/     # features escaladas (features_TICKER.csv)
├── models/            # pesos guardados (.keras)
├── notebooks/         # exploración EDA
├── results/           # gráficos y métricas exportadas
└── src/
    ├── data.py        # descarga de yfinance
    ├── features.py    # ingeniería de features (8 features por ticker)
    ├── model.py       # arquitectura LSTM Autoencoder
    ├── train.py       # entrenamiento con EarlyStopping
    └── evaluate.py    # umbral 3σ, plot de anomalías, métricas
```

## Stack técnico

- Python 3.10+
- yfinance 0.2.38 — datos históricos de Yahoo Finance
- TensorFlow 2.15 / Keras — modelo LSTM Autoencoder
- pandas, numpy — procesamiento de datos
- scikit-learn — StandardScaler, métricas
- matplotlib, seaborn — visualización

## Tickers del MERVAL

```python
TICKERS = ["GGAL.BA", "BMA.BA", "YPFD.BA", "PAMP.BA", "BBAR.BA", "ALUA.BA", "^MERV"]
```

Sufijo `.BA` = Bolsa de Buenos Aires. `^MERV` es el índice completo.

## División temporal del dataset

| Split | Período | Propósito |
|-------|---------|-----------|
| Train | 2015-01-01 → 2017-12-31 | Solo datos "normales" — aprende la norma |
| Validation | 2018-01-01 → 2018-12-31 | Ajuste de umbral |
| Test | 2019-01-01 → 2025-12-31 | Contiene anomalías reales conocidas |

**Regla crítica**: el `StandardScaler` se hace `fit` SOLO sobre train. Nunca sobre todo el dataset.

## Features del modelo (8 por ticker)

| Feature | Cálculo |
|---------|---------|
| `ret_1d` | `close.pct_change()` |
| `log_ret` | `np.log(close / close.shift(1))` |
| `vol_5d` | `ret_1d.rolling(5).std()` |
| `vol_20d` | `ret_1d.rolling(20).std()` |
| `vol_ratio` | `vol_5d / merval_vol_5d` |
| `corr_30d` | `rolling_corr(ticker, merval, 30)` |
| `zscore` | `(close - mean_90d) / std_90d` |
| `vol_ratio_v` | `volume / volume.rolling(20).mean()` |

## Arquitectura del modelo

```
Input (30, 8) → LSTM(64) → Dropout(0.2) → RepeatVector(30)
             → LSTM(64, return_seq=True) → Dropout(0.2) → Dense(8)
Loss: MAE | Optimizer: Adam | EarlyStopping patience=7
```

Tensor de entrada: `(N_ventanas, 30_dias, 8_features)` — ventanas deslizantes de 30 días.

## Detección de anomalías

Umbral: `threshold = mean(train_errors) + 3 * std(train_errors)`

Para el experimento de variación de umbral: probar con `sigma ∈ {2, 3, 4}`.

## Eventos históricos etiquetados (ground truth)

```python
EVENTOS = {
    "2019-08-12": "PASO 2019 — MERVAL -38% en un día",
    "2020-03-13": "COVID-19 — caída global sincronizada",
    "2023-08-14": "PASO 2023 — devaluación post-primarias",
    "2023-12-13": "Shock Milei — devaluación +54% tipo oficial",
}
```

El modelo debe marcar estas fechas con alto error de reconstrucción.

## Experimentos del trabajo integrador

1. **Variación de umbral**: mismo modelo, `sigma ∈ {2, 3, 4}` — graficar precisión/recall
2. **Acciones vs índice**: modelo entrenado con `^MERV` solo vs con tickers individuales
3. **Ablation study**: modelo con solo retornos (`ret_1d`, `log_ret`) vs modelo con las 8 features

## Convenciones de código

- Type hints en todas las funciones públicas
- Docstrings en español con parámetros y retornos
- Guardar modelos en `models/` con nombre descriptivo: `lstm_ae_GGAL_sigma3.keras`
- Guardar gráficos en `results/` como PNG a 150 DPI
- No hacer `fit` del scaler sobre datos de test — esto es data leakage

## Notas importantes

- Los feriados argentinos generan gaps en los datos — usar `ffill()` con límite de 3 días
- El índice `^MERV` está en ARS nominales; normalizar con el scaler antes de comparar con acciones
- Algunas acciones tienen NaN los primeros años — `dropna()` por ticker, no global
- `yfinance` puede fallar en descargas masivas — agregar `time.sleep(1)` entre tickers si ocurre
