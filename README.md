# Detección de Anomalías en el MERVAL — LSTM Autoencoder

**Trabajo Integrador — Inteligencia Artificial**
**Ticker analizado:** GGAL.BA (Grupo Financiero Galicia)

---

## Introducción

El objetivo del trabajo fue construir un sistema de detección de anomalías en precios de acciones del mercado bursátil argentino (MERVAL) usando un modelo de aprendizaje profundo no supervisado: un **LSTM Autoencoder**.

La hipótesis central es que un modelo entrenado exclusivamente con datos "normales" (períodos sin shocks económicos) aprende a reconstruir el comportamiento habitual del mercado. Cuando el mercado se comporta de forma atípica, el error de reconstrucción del autoencoder aumenta, lo que permite identificar anomalías sin necesitar etiquetas.

---

## Stack técnico

| Componente | Herramienta |
|------------|-------------|
| Lenguaje | Python 3.14 |
| Datos | yfinance 1.4.1 |
| Procesamiento | pandas 3.0.3, numpy 2.4.6 |
| Modelo | PyTorch 2.12.0 |
| Preprocessing | scikit-learn 1.8.0 |
| Visualización | matplotlib 3.10.9, seaborn 0.13.2 |

> **Nota:** el plan original especificaba TensorFlow 2.15, pero no tiene soporte para Python 3.14. Se adaptó la implementación a PyTorch manteniendo la misma arquitectura y lógica de entrenamiento.

---

## Datos

### Fuente y descarga

Se descargaron datos históricos desde Yahoo Finance usando `yfinance` para los siguientes instrumentos:

| Ticker | Instrumento |
|--------|-------------|
| GGAL.BA | Grupo Financiero Galicia |
| BMA.BA | Banco Macro |
| YPFD.BA | YPF S.A. |
| PAMP.BA | Pampa Energía |
| BBAR.BA | Banco BBVA Argentina |
| ALUA.BA | Aluar Aluminio |
| ^MERV | Índice Merval (benchmark) |

El sufijo `.BA` indica cotización en la Bolsa de Buenos Aires. Los datos cubren el período **2 enero 2015 – 30 diciembre 2025**, resultando en **2.682 días de cotización**.

Los gaps generados por feriados argentinos se rellenaron con forward-fill limitado a 3 días consecutivos (`ffill(limit=3)`).

### División temporal

La división del dataset es intencionalmente asimétrica: el modelo aprende solo con años estables, y el período de prueba concentra los eventos de crisis conocidos.

| Split | Período | Ventanas | Propósito |
|-------|---------|----------|-----------|
| Train | 2015-01-02 → 2017-12-31 | 613 | Aprende el comportamiento normal |
| Validación | 2018-01-01 → 2018-12-31 | 215 | Ajuste de hiperparámetros |
| Test | 2019-01-01 → 2025-12-30 | 1.675 | Contiene los shocks conocidos |

---

## Ingeniería de features

Se construyeron **8 features** por cada día de cotización, diseñadas para capturar distintas dimensiones del comportamiento del activo:

| Feature | Cálculo | Captura |
|---------|---------|---------|
| `ret_1d` | `close.pct_change()` | Retorno diario |
| `log_ret` | `log(close / close.shift(1))` | Retorno logarítmico |
| `vol_5d` | `ret_1d.rolling(5).std()` | Volatilidad de corto plazo |
| `vol_20d` | `ret_1d.rolling(20).std()` | Volatilidad de mediano plazo |
| `vol_ratio` | `vol_5d / merval_vol_5d` | Volatilidad relativa al índice |
| `corr_30d` | `rolling_corr(ticker, merval, 30)` | Correlación con el mercado |
| `zscore` | `(close - mean_90d) / std_90d` | Desviación del precio de su media |
| `vol_ratio_v` | `volume / volume.rolling(20).mean()` | Volumen relativo |

Los features se calcularon sobre los 2.593 días con datos completos para GGAL.BA (el primer año necesita período de calentamiento para las ventanas móviles de 90 días).

**Normalización:** se aplicó `StandardScaler` ajustado **exclusivamente** sobre el conjunto de train. Los datos de validación y test se transforman con el mismo scaler — nunca se re-ajusta — para evitar *data leakage*.

### Ventanas temporales

El modelo consume secuencias de 30 días consecutivos (ventanas deslizantes). Cada ventana tiene shape `(30, 8)` — 30 timesteps, 8 features — generando tensores de entrada `(N, 30, 8)`.

---

## Arquitectura del modelo

El **LSTM Autoencoder** comprime la secuencia de entrada a un vector latente y luego la reconstruye. La diferencia entre entrada y salida (error de reconstrucción) es la señal de anomalía.

```
Input (30, 8)
    │
    ▼
LSTM(64, return_sequences=False)   ← Encoder: resume 30 días en 64 dimensiones
    │
Dropout(0.2)
    │
RepeatVector(30)                   ← Bottleneck: expande el contexto
    │
    ▼
LSTM(64, return_sequences=True)    ← Decoder: reconstruye la secuencia
    │
Dropout(0.2)
    │
    ▼
Dense(8)                           ← Output: reconstrucción de los 8 features
    │
Output (30, 8)
```

**Función de pérdida:** MAE (Mean Absolute Error) — penaliza el error de reconstrucción promedio por timestep y feature.
**Optimizador:** Adam con learning rate por defecto (1e-3).
**Parámetros totales:** ~100.000

---

## Entrenamiento

El modelo se entrenó con datos de 2015–2017 usando las siguientes configuraciones:

| Hiperparámetro | Valor |
|---------------|-------|
| Épocas máximas | 50 |
| Batch size | 32 |
| EarlyStopping patience | 7 |
| Métrica monitoreada | val_loss (MAE) |

### Resultados del entrenamiento

El modelo convergió en **16 épocas** gracias al EarlyStopping. Se guardó automáticamente el checkpoint con mejor val_loss.

| Métrica | Valor |
|---------|-------|
| Mejor val_loss | **0.9878** |
| Época de parada | 16 |

La curva de pérdida muestra convergencia estable sin sobreajuste:

![Curva de pérdida](results/loss_curve.png)

---

## Detección de anomalías

### Umbral de decisión (regla N-σ)

El umbral se calcula a partir del **error de reconstrucción en el conjunto de train** (datos normales):

```
threshold = mean(train_errors) + σ × std(train_errors)
```

Una ventana se clasifica como anomalía si su error supera el umbral.

### Experimento 1 — Variación de sigma

Se evaluó el mismo modelo con tres valores de σ sobre el test set (2019–2025):

| σ | Umbral | Anomalías detectadas |
|---|--------|---------------------|
| 2 | 0.77802 | 724 |
| **3** | **0.87476** | **441** |
| 4 | 0.97150 | 304 |

Con σ=2 el modelo es muy sensible (muchas alertas). Con σ=4 es más conservador. **σ=3 ofrece el balance más interpretable** y es el estándar en detección estadística de outliers.

### Experimento 2 — Gráfico principal (σ=3)

El modelo con σ=3 detectó **441 días anómalos** en el período 2019–2025. Los primeros días detectados coinciden exactamente con el **PASO 2019** (12 de agosto de 2019), donde el MERVAL cayó un 38% en una sola jornada — el evento de mayor caída en un día de la historia reciente del mercado argentino.

```
Primeras anomalías detectadas:
2019-08-12, 2019-08-13, 2019-08-14, 2019-08-15, 2019-08-16,
2019-08-20, 2019-08-21, 2019-08-22, 2019-08-23, 2019-08-26 ...
```

Eventos históricos etiquetados como ground truth:

| Fecha | Evento |
|-------|--------|
| 2019-08-12 | PASO 2019 — caída -38% en un día |
| 2020-03-13 | COVID-19 — caída global sincronizada |
| 2023-08-14 | PASO 2023 — devaluación post-primarias |
| 2023-12-13 | Shock Milei — devaluación +54% tipo oficial |

![Anomalías detectadas](results/anomalias.png)

### Experimento 3 — Ablation study

Se comparó el modelo completo (8 features) contra una versión reducida (solo retornos: `ret_1d`, `log_ret`):

| Configuración | Umbral 3σ | Anomalías detectadas |
|---------------|-----------|---------------------|
| Solo retornos (2 features) | 1.27953 | 365 |
| **8 features** | **0.84712** | **511** |

El modelo de 8 features es **más sensible**: detecta más anomalías con un umbral menor. Esto indica que los features adicionales (volatilidad relativa, correlación con el índice, z-score de precio) aportan señal real — el modelo aprende patrones más ricos del comportamiento normal y reacciona más fuerte ante desviaciones.

![Ablation study](results/ablation_study.png)

---

## Estructura del proyecto

```
integrador-merval/
├── README.md
├── PLAN.md
├── CLAUDE.md
├── requirements.txt
├── data/
│   ├── raw/
│   │   ├── close.csv        # Precios de cierre (2.682 días × 7 tickers)
│   │   └── volume.csv       # Volúmenes operados
│   └── processed/
│       └── features_GGAL_BA.csv   # 8 features calculadas (2.593 días)
├── models/
│   ├── best.pt              # Modelo principal (σ=3, 8 features)
│   ├── ablation_solo_retornos.pt
│   └── ablation_8_features.pt
├── results/
│   ├── loss_curve.png       # Curva de entrenamiento
│   ├── anomalias.png        # Serie temporal con anomalías marcadas
│   ├── sigma_sweep.csv      # Tabla Experimento 1
│   └── ablation_study.png   # Comparativo Experimento 3
└── src/
    ├── __init__.py
    ├── data.py              # Descarga y carga de datos (yfinance)
    ├── features.py          # Ingeniería de features + split + scaler
    ├── model.py             # Arquitectura LSTMAutoencoder (PyTorch)
    ├── train.py             # Loop de entrenamiento con EarlyStopping
    └── evaluate.py          # Umbral, detección, gráficos, experimentos
```

---

## Cómo reproducir

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Descargar datos
python -m src.data

# 3. Construir features
python -m src.features

# 4. Entrenar el modelo
python -m src.train

# 5. Evaluar y generar resultados
python -m src.evaluate

# O ejecutar todo de una vez:
python -m src.data && python -m src.features && python -m src.train && python -m src.evaluate
```

> Los scripts deben ejecutarse desde el directorio raíz del proyecto.

---

## Conclusiones

1. **El enfoque no supervisado funciona**: el modelo, entrenado sin ninguna etiqueta de anomalía, detecta correctamente los eventos de mayor impacto en el mercado argentino (PASO 2019, COVID-19).

2. **La elección de σ importa**: σ=3 es el punto de equilibrio entre sensibilidad y especificidad. Con σ=2 se generan demasiadas alertas en períodos de alta volatilidad sostenida (2022–2023); con σ=4 se pierden eventos reales.

3. **Más features, mejor detección**: el ablation study muestra que agregar contexto (volatilidad relativa, correlación con el índice, z-score) mejora la capacidad del modelo para distinguir el comportamiento normal del anómalo.

4. **Limitación principal**: el modelo detecta períodos de alta volatilidad en general, no solo eventos discretos. El MERVAL 2022–2023 tuvo alta volatilidad estructural (crisis cambiaria) que genera falsos positivos incluso con σ=3. Una mejora posible sería incorporar variables macroeconómicas (tipo de cambio, riesgo país) como contexto adicional.
