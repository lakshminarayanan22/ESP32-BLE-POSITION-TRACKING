"""
PSO-LSTM Position Predictor (Inference Module)
================================================

Loads the trained PSO-LSTM model and runs real-time inference.

The model was trained with:
    - Input : (1, SEQ_LEN=8, N_FEATURES=20) — temporal sliding window
    - Output: (1, 3) — normalized (x, y, z) coordinates in [0, 1]

predict() denormalizes the output to actual room coordinates in meters
and returns a confidence label based on the average RSSI quality of the
most recent feature window.
"""

import torch
import numpy as np
from LSTM import LSTMModel
from config import (
    N_FEATURES, HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    MODEL_PATH, RSSI_MIN, RSSI_MAX,
    ROOM_W, ROOM_H, ROOM_Z
)


class PositionPredictor:
    """
    Inference wrapper for the PSO-LSTM 3D positioning model.

    Input:
        sequence : np.ndarray shape (SEQ_LEN, N_FEATURES=20)
                   Normalized, gateway-corrected RSSI feature vectors
                   representing a temporal sliding window of RSSI history.

    Output (predict() return dict):
        x          : estimated X coordinate in meters
        y          : estimated Y coordinate in meters
        z          : estimated Z coordinate in meters
        confidence : signal quality label derived from mean RSSI feature
        zone       : spatial zone label based on (x, y) position
    """

    def __init__(self):
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.model = LSTMModel(
            input_size  = N_FEATURES,
            hidden_size = HIDDEN_SIZE,
            num_layers  = NUM_LAYERS,
            output_size = 3,          # predict (x, y, z)
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
            sequence : np.ndarray of shape (SEQ_LEN, N_FEATURES)
                       Each row is a 20-dim feature vector
                       [mean, std, min, max, count] × 4 stations,
                       computed from gateway-corrected RSSI readings.

        Returns:
            dict with keys: x, y, z, confidence, zone
        """
        # Add batch dimension: (1, SEQ_LEN, N_FEATURES)
        x = torch.tensor(sequence, dtype=torch.float32) \
                  .unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x)   # (1, 3) — normalized coordinates

        # Denormalize from [0, 1] to actual room meters
        pred = output.squeeze().cpu().numpy()
        x_m  = float(np.clip(pred[0], 0.0, 1.0)) * ROOM_W
        y_m  = float(np.clip(pred[1], 0.0, 1.0)) * ROOM_H
        z_m  = float(np.clip(pred[2], 0.0, 1.0)) * ROOM_Z

        # Estimate signal quality from mean RSSI features in the last timestep
        # The mean features are at indices 0, 5, 10, 15 (first of each 5-stat block)
        last_step    = sequence[-1]   # most recent time step
        mean_features = [last_step[i * 5] for i in range(4)]
        # Denormalize from [0,1] back to dBm for quality label
        active = [f for f in mean_features if f > 0]
        if active:
            avg_norm_rssi = float(np.mean(active))
            avg_rssi_dbm  = avg_norm_rssi * (RSSI_MAX - RSSI_MIN) + RSSI_MIN
        else:
            avg_rssi_dbm = -100.0

        return {
            "x":          round(x_m, 3),
            "y":          round(y_m, 3),
            "z":          round(z_m, 3),
            "confidence": self._confidence_label(avg_rssi_dbm),
            "zone":       self._zone_label(x_m, y_m)
        }

    # ── Label Helpers ─────────────────────────────────────────────────────

    def _confidence_label(self, rssi_dbm: float) -> str:
        """Map average RSSI to a signal quality label."""
        if rssi_dbm >= -60:  return "EXCELLENT"
        if rssi_dbm >= -70:  return "GOOD"
        if rssi_dbm >= -80:  return "FAIR"
        if rssi_dbm >= -90:  return "WEAK"
        return "VERY_WEAK"

    def _zone_label(self, x: float, y: float) -> str:
        """
        Assign a spatial zone label from (x, y) position.
        Room is divided into a 2×2 grid.
        """
        mid_x = ROOM_W / 2.0
        mid_y = ROOM_H / 2.0

        ns = "NORTH" if y >= mid_y else "SOUTH"
        ew = "EAST"  if x >= mid_x else "WEST"
        return f"{ns}-{ew}"
