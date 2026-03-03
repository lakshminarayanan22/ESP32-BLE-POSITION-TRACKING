import torch
import numpy as np
from LSTM  import LSTMModel
from config import (N_FEATURES, HIDDEN_SIZE, NUM_LAYERS,
                    DROPOUT, MODEL_PATH, RSSI_MIN, RSSI_MAX)

class RSSIPredictor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = LSTMModel(
            input_size  = N_FEATURES,    # 1 (rssi only)
            hidden_size = HIDDEN_SIZE,
            num_layers  = NUM_LAYERS,
            output_size = N_FEATURES,    # predict next rssi
            dropout     = DROPOUT
        ).to(self.device)

        self._load_weights()
        self.model.eval()
        print(f"[MODEL] Ready on {self.device}")

    def _load_weights(self):
        try:
            self.model.load_state_dict(
                torch.load(MODEL_PATH, map_location=self.device)
            )
            print(f"[MODEL] Weights loaded from {MODEL_PATH}")
        except FileNotFoundError:
            print(f"[MODEL] WARNING: {MODEL_PATH} not found — using untrained weights")

    def predict(self, sequence: np.ndarray) -> dict:
        """
        sequence : np.array of shape (SEQ_LEN, 1) — normalized RSSI values
        returns  : dict with predicted RSSI and derived info
        """
        # (1, SEQ_LEN, 1)
        x = torch.tensor(sequence, dtype=torch.float32) \
                  .unsqueeze(0).to(self.device)

        with torch.no_grad():
            output = self.model(x)          # (1, 1)

        pred_normalized = output.squeeze().item()
        pred_rssi       = pred_normalized * (RSSI_MAX - RSSI_MIN) + RSSI_MIN
        pred_rssi       = max(RSSI_MIN, min(RSSI_MAX, pred_rssi))  # clamp

        # Simple distance estimate from predicted RSSI
        tx_power = -59
        distance = 10 ** ((tx_power - pred_rssi) / 20.0)

        return {
            "predicted_rssi":     round(pred_rssi, 2),
            "predicted_distance": round(distance,  2),
            "signal_quality":     self._quality_label(pred_rssi)
        }

    def _quality_label(self, rssi):
        if rssi >= -60:  return "EXCELLENT"
        if rssi >= -70:  return "GOOD"
        if rssi >= -80:  return "FAIR"
        if rssi >= -90:  return "WEAK"
        return "VERY_WEAK"