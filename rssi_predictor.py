"""
PSO-LSTM Position Predictor (Inference Module)
================================================

Loads the trained PSO-LSTM model and runs real-time inference.

Model trained with:
    Input  : (1, SEQ_LEN=8, N_FEATURES=70) — temporal sliding window
    Output : (1, 2) — normalized (x, y) floor coordinates in [0, 1]

predict() denormalises to actual metres and returns a confidence label.
Z is intentionally excluded — only the 2D floor plan is tracked and plotted.
"""

import torch
import numpy as np
from LSTM import LSTMModel
from config import (
    N_FEATURES, N_OUTPUTS, HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    MODEL_PATH, RSSI_MIN, RSSI_MAX,
    ROOM_W, ROOM_H,
)


class PositionPredictor:
    """
    Inference wrapper for the PSO-LSTM 2D positioning model.

    Input:
        sequence : np.ndarray shape (SEQ_LEN, N_FEATURES)
                   Normalised, gateway-corrected RSSI feature vectors.

    Output (predict() → dict):
        x          : estimated X coordinate in metres
        y          : estimated Y coordinate in metres
        confidence : signal quality label from mean RSSI
        zone       : NORTH/SOUTH-EAST/WEST quadrant label
    """

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = LSTMModel(
            input_size  = N_FEATURES,
            hidden_size = HIDDEN_SIZE,
            num_layers  = NUM_LAYERS,
            output_size = N_OUTPUTS,   # 2 — (x, y)
            dropout     = DROPOUT
        ).to(self.device)

        self._load_weights()
        self.model.eval()
        print(f"[MODEL] PositionPredictor ready on {self.device}")

    def _load_weights(self):
        try:
            self.model.load_state_dict(
                torch.load(MODEL_PATH, map_location=self.device,
                           weights_only=True)
            )
            print(f"[MODEL] Weights loaded from {MODEL_PATH}")
        except FileNotFoundError:
            print(f"[MODEL] WARNING: {MODEL_PATH} not found — "
                  f"using random weights (run train_model.py first)")
        except Exception as e:
            print(f"[MODEL] WARNING: Could not load weights — {e}")

    def predict(self, sequence: np.ndarray) -> dict:
        """
        Run LSTM inference on a temporal RSSI sequence.

        Args:
            sequence : np.ndarray shape (SEQ_LEN, N_FEATURES)

        Returns:
            dict with keys: x, y, confidence, zone
        """
        x_t = torch.tensor(sequence, dtype=torch.float32) \
                    .unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x_t)   # (1, 2)

        pred = output.squeeze().cpu().numpy()
        x_m  = float(np.clip(pred[0], 0.0, 1.0)) * ROOM_W
        y_m  = float(np.clip(pred[1], 0.0, 1.0)) * ROOM_H

        # Signal quality from mean RSSI features in the last timestep
        last_step     = sequence[-1]
        mean_features = [last_step[i * 5] for i in range(N_FEATURES // 5)]
        active        = [f for f in mean_features if f > 0]
        avg_rssi_dbm  = (
            float(np.mean(active)) * (RSSI_MAX - RSSI_MIN) + RSSI_MIN
            if active else -100.0
        )

        return {
            "x":          round(x_m, 3),
            "y":          round(y_m, 3),
            "confidence": self._confidence_label(avg_rssi_dbm),
            "zone":       self._zone_label(x_m, y_m),
        }

    def _confidence_label(self, rssi_dbm: float) -> str:
        if rssi_dbm >= -60: return "EXCELLENT"
        if rssi_dbm >= -70: return "GOOD"
        if rssi_dbm >= -80: return "FAIR"
        if rssi_dbm >= -90: return "WEAK"
        return "VERY_WEAK"

    def _zone_label(self, x: float, y: float) -> str:
        ns = "NORTH" if y >= ROOM_H / 2.0 else "SOUTH"
        ew = "EAST"  if x >= ROOM_W / 2.0 else "WEST"
        return f"{ns}-{ew}"
