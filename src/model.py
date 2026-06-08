"""Arquitectura del LSTM Autoencoder para detección de anomalías (PyTorch)."""

import torch
import torch.nn as nn


class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder para detección de anomalías en series temporales.

    Arquitectura:
        Encoder: LSTM(64) → Dropout(0.2)
        Bottleneck: RepeatVector(time_steps)
        Decoder: LSTM(64, return_seq=True) → Dropout(0.2) → Linear(n_features)

    Args:
        time_steps: Longitud de la ventana temporal.
        n_features: Cantidad de features por paso de tiempo.
    """

    def __init__(self, time_steps: int = 30, n_features: int = 8) -> None:
        super().__init__()
        self.time_steps = time_steps
        self.encoder_lstm = nn.LSTM(n_features, 64, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.decoder_lstm = nn.LSTM(64, 64, batch_first=True)
        self.dropout2 = nn.Dropout(0.2)
        self.output_layer = nn.Linear(64, n_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Encoder: tomar el último hidden state
        _, (h_n, _) = self.encoder_lstm(x)       # h_n: (1, batch, 64)
        encoded = self.dropout1(h_n[-1])           # (batch, 64)

        # RepeatVector: repetir el contexto time_steps veces
        repeated = encoded.unsqueeze(1).expand(-1, self.time_steps, -1)  # (batch, time_steps, 64)

        # Decoder
        decoded, _ = self.decoder_lstm(repeated)  # (batch, time_steps, 64)
        decoded = self.dropout2(decoded)
        return self.output_layer(decoded)          # (batch, time_steps, n_features)


def build_lstm_ae(time_steps: int = 30, n_features: int = 8) -> LSTMAutoencoder:
    """
    Construye el LSTM Autoencoder.

    Args:
        time_steps: Longitud de la ventana temporal.
        n_features: Cantidad de features por paso de tiempo.

    Returns:
        Modelo LSTMAutoencoder listo para entrenar.
    """
    return LSTMAutoencoder(time_steps, n_features)


if __name__ == "__main__":
    model = build_lstm_ae()
    x = torch.randn(4, 30, 8)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    total = sum(p.numel() for p in model.parameters())
    print(f"Parámetros totales: {total:,}")
