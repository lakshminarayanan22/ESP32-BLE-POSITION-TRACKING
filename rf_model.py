"""
Random Forest Position Predictor (Inference Module)
=====================================================

Loads the trained RandomForestRegressor model and runs real-time inference
on a single 70-dim RSSI feature vector.

Model trained in train_svm_rf.py using GridSearchCV over:
    n_estimators, max_depth, min_samples_split, min_samples_leaf,
    max_features, min_impurity_decrease, bootstrap.

Random Forest naturally supports multi-output regression (predicts x, y, z
simultaneously without needing MultiOutputRegressor wrapping).

Like SVM, it predicts from a single snapshot — no temporal warm-up delay.

Input:
    feature_vec : np.ndarray of shape (N_FEATURES=70,)
                  Normalized [mean, std, min, max, count] × 14 stations
                  Gateway-corrected, built by DataProcessor.

Output (predict() → dict):
    x, y, z     : estimated coordinates in meters
"""

import numpy as np
import joblib
from config import RF_MODEL_PATH, ROOM_W, ROOM_H


class RFPositionPredictor:
    """
    Inference wrapper for the GridSearchCV-optimised RandomForestRegressor.

    RandomForestRegressor natively handles multi-output targets, predicting
    (x, y, z) in a single call. Each tree votes independently, and the
    ensemble average reduces variance from noisy RSSI inputs.
    """

    def __init__(self):
        self.model  = None
        self.ready  = False
        self._load_model()

    def _load_model(self):
        try:
            self.model = joblib.load(RF_MODEL_PATH)
            self.ready = True
            print(f"[RF]  Model loaded from {RF_MODEL_PATH}")
        except FileNotFoundError:
            print(f"[RF]  WARNING: {RF_MODEL_PATH} not found — "
                  f"run train_svm_rf.py first")
        except Exception as e:
            print(f"[RF]  WARNING: Could not load model — {e}")

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
