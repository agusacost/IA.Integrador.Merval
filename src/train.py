"""Entrenamiento del LSTM Autoencoder con datos normales (2015–2017)."""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from src.model import build_lstm_ae, LSTMAutoencoder
from src.features import build_features, split_and_scale
from src.data import load_data


TICKER     = "GGAL.BA"
EPOCHS     = 50
BATCH      = 32
PATIENCE   = 7
DEVICE     = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(
    X_train: np.ndarray,
    X_val: np.ndarray,
    model_path: str = "models/best.pt",
    epochs: int = EPOCHS,
    batch_size: int = BATCH,
) -> tuple[LSTMAutoencoder, dict]:
    """
    Entrena el autoencoder usando solo datos normales.

    Args:
        X_train:    Tensor de entrenamiento (N, 30, n_features).
        X_val:      Tensor de validación (N, 30, n_features).
        model_path: Ruta donde guardar el mejor modelo.
        epochs:     Máximo de épocas.
        batch_size: Tamaño del batch.

    Returns:
        Tupla (model, history) donde history tiene keys 'loss' y 'val_loss'.
    """
    Path("models").mkdir(exist_ok=True)

    model = build_lstm_ae(
        time_steps=X_train.shape[1],
        n_features=X_train.shape[2],
    ).to(DEVICE)

    optimizer = torch.optim.Adam(model.parameters())

    X_tr = torch.FloatTensor(X_train).to(DEVICE)
    X_v  = torch.FloatTensor(X_val).to(DEVICE)

    loader = DataLoader(TensorDataset(X_tr, X_tr), batch_size=batch_size, shuffle=True)

    best_val_loss   = float("inf")
    patience_counter = 0
    history         = {"loss": [], "val_loss": []}

    for epoch in range(1, epochs + 1):
        model.train()
        batch_losses = []
        for xb, _ in loader:
            optimizer.zero_grad()
            pred = model(xb)
            loss = torch.mean(torch.abs(pred - xb))
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(X_v)
            val_loss = torch.mean(torch.abs(val_pred - X_v)).item()

        avg_train = sum(batch_losses) / len(batch_losses)
        history["loss"].append(avg_train)
        history["val_loss"].append(val_loss)

        print(f"Epoch {epoch:3d}/{epochs} — loss: {avg_train:.6f} — val_loss: {val_loss:.6f}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), model_path)
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"EarlyStopping en época {epoch}. Mejor val_loss: {best_val_loss:.6f}")
                model.load_state_dict(torch.load(model_path, weights_only=True))
                break

    return model, history


def plot_loss(history: dict, save_path: str = "results/loss_curve.png") -> None:
    """Grafica la curva de pérdida de entrenamiento y validación."""
    Path("results").mkdir(exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.plot(history["loss"],     label="Train loss")
    plt.plot(history["val_loss"], label="Val loss")
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
    print(f"Mejor val_loss: {min(history['val_loss']):.6f}")
