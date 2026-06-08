"""Evaluación del modelo: umbral 3σ, detección de anomalías y métricas."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from pathlib import Path

import torch

from src.model import build_lstm_ae, LSTMAutoencoder
from src.features import build_features, split_and_scale
from src.data import load_data


TICKER = "GGAL.BA"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

EVENTOS = {
    "2019-08-12": "PASO 2019\n(-38%)",
    "2020-03-13": "COVID-19",
    "2023-08-14": "PASO 2023",
    "2023-12-13": "Shock Milei",
}


def load_model(model_path: str, time_steps: int = 30, n_features: int = 8) -> LSTMAutoencoder:
    """
    Carga el modelo desde disco.

    Args:
        model_path: Ruta al archivo .pt con los pesos.
        time_steps: Longitud de la ventana temporal.
        n_features: Cantidad de features.

    Returns:
        Modelo LSTMAutoencoder en modo eval.
    """
    model = build_lstm_ae(time_steps, n_features).to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


def reconstruction_error(model: LSTMAutoencoder, X: np.ndarray) -> np.ndarray:
    """
    Calcula el error de reconstrucción MAE por ventana.

    Args:
        model: Modelo PyTorch entrenado.
        X:     Tensor de entrada (N, 30, n_features).

    Returns:
        Array 1D con el MAE de cada ventana.
    """
    model.eval()
    with torch.no_grad():
        X_t   = torch.FloatTensor(X).to(DEVICE)
        X_hat = model(X_t).cpu().numpy()
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

    anomaly_mask  = errors_test > threshold
    anomaly_dates = dates_test[anomaly_mask]

    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(dates_test, errors_test, lw=0.8, color="steelblue",
            alpha=0.85, label="Error de reconstrucción")
    ax.fill_between(dates_test, errors_test, alpha=0.15, color="steelblue")
    ax.axhline(threshold, color="crimson", ls="--", lw=1.5,
               label=f"Umbral {sigma}σ = {threshold:.4f}")

    ax.scatter(anomaly_dates,
               errors_test[anomaly_mask],
               color="crimson", s=18, zorder=5, label="Anomalía detectada")

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
        print(f"sigma={s} -> umbral={thr:.5f} | anomalias={n}")
    return pd.DataFrame(rows)


def run_ablation_experiment(
    close: pd.DataFrame,
    volume: pd.DataFrame,
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
            model_path=f"models/ablation_{name}.pt",
            epochs=50,
        )
        err = reconstruction_error(model, X_te)
        thr = compute_threshold(reconstruction_error(model, X_tr))
        results[name] = {"errors": err, "dates": d_te, "threshold": thr}
        print(f"[{name}] threshold={thr:.5f} | anomalías={(err > thr).sum()}")

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for ax, (name, data) in zip(axes, results.items()):
        ax.plot(data["dates"], data["errors"], lw=0.8)
        ax.axhline(data["threshold"], color="crimson", ls="--", lw=1.2,
                   label="Umbral 3σ")
        ax.set_title(f"Ablation — {name}")
        ax.set_ylabel("MAE")
        ax.legend()
    plt.tight_layout()
    plt.savefig("results/ablation_study.png", dpi=150)
    plt.close()
    print("Guardado: results/ablation_study.png")


if __name__ == "__main__":
    Path("results").mkdir(exist_ok=True)

    close, volume = load_data()
    df = build_features(close, volume, TICKER)
    X_train, X_val, X_test, d_train, d_val, d_test, scaler = split_and_scale(df)

    model = load_model("models/best.pt")

    train_errors = reconstruction_error(model, X_train)
    test_errors  = reconstruction_error(model, X_test)

    # -- Experimento 1: variacion de sigma ------------------------------------
    print("\n-- Experimento 1: variacion de sigma --")
    sweep_df = sigma_sweep(train_errors, test_errors, d_test)
    sweep_df.to_csv("results/sigma_sweep.csv", index=False)

    # -- Experimento 2: grafico principal con sigma=3 -------------------------
    print("\n-- Experimento 2: deteccion con sigma=3 --")
    threshold     = compute_threshold(train_errors, sigma=3.0)
    anomaly_dates = plot_anomalias(test_errors, d_test, threshold, sigma=3.0)
    print(f"Anomalias detectadas: {len(anomaly_dates)}")
    print(anomaly_dates.strftime("%Y-%m-%d").tolist()[:10], "...")

    # -- Experimento 3: ablation study ----------------------------------------
    print("\n-- Experimento 3: ablation study --")
    run_ablation_experiment(close, volume)

    print("\nPipeline completo. Revisar carpeta results/")
