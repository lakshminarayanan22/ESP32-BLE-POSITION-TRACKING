"""
SVM Position Predictor — RBF Kernel (Inference Module)
=======================================================

Loads the trained MultiOutputRegressor(SVR) model and runs real-time
inference on a single 70-dim RSSI feature vector.

Model trained in train_svm_rf.py using GridSearchCV over:
    C, gamma, epsilon  with RBF kernel.

Unlike the LSTM, the SVM does not require a temporal sequence — it
predicts from a single snapshot feature vector, making it available
immediately after the first RSSI reading (no warm-up delay).

Input:
    feature_vec : np.ndarray of shape (N_FEATURES=70,)
                  Normalized [mean, std, min, max, count] × 14 stations
                  Gateway-corrected, built by DataProcessor.

Output (predict() → dict):
    x, y, z     : estimated coordinates in meters
"""

import numpy as np
import joblib
from config import SVM_MODEL_PATH, ROOM_W, ROOM_H


class SVMPositionPredictor:
    """
    Inference wrapper for the GridSearchCV-optimised SVR (RBF kernel).

    Uses sklearn's MultiOutputRegressor to regress (x, y, z) coordinates
    simultaneously from the 70-dim gateway-corrected RSSI feature vector.
    """

    def __init__(self):
        self.model  = None
        self.ready  = False
        self._load_model()

    def _load_model(self):
        try:
            self.model = joblib.load(SVM_MODEL_PATH)
            self.ready = True
            print(f"[SVM] Model loaded from {SVM_MODEL_PATH}")
        except FileNotFoundError:
            print(f"[SVM] WARNING: {SVM_MODEL_PATH} not found — "
                  f"run train_svm_rf.py first")
        except Exception as e:
            print(f"[SVM] WARNING: Could not load model — {e}")

    def predict(self, feature_vec: np.ndarray) -> dict:
        """
        Predict 3D position from a single RSSI feature snapshot.

        Args:
            feature_vec : 1D np.ndarray of shape (N_FEATURES,)
                          Normalized [0, 1] values from DataProcessor.

        Returns:
            dict with keys: x, y, z  (float, meters)
        """
        if not self.ready or self.model is None:
            return {"x": 0.0, "y": 0.0, "z": 0.0}

        x_in  = np.array(feature_vec, dtype=np.float64).reshape(1, -1)
        pred  = self.model.predict(x_in)[0]   # shape (2,) — normalised (x, y)

        x_m = float(np.clip(pred[0], 0.0, 1.0)) * ROOM_W
        y_m = float(np.clip(pred[1], 0.0, 1.0)) * ROOM_H

        return {
            "x": round(x_m, 3),
            "y": round(y_m, 3),
        }
