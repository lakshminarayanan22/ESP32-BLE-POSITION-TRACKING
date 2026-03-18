"""
Bluetooth Gateway Real-Time RSSI Correction
============================================

Mitigates BLE transmit power instability using a fixed-position Bluetooth
gateway as a reference signal monitor.

Problem:
    BLE beacon transmit power is not perfectly constant. A beacon can
    experience up to ±10 dB fluctuations due to hardware temperature drift,
    battery voltage changes, or firmware behavior. A static distance model
    interprets these fluctuations as physical movement, causing ghost
    position jumps of several meters.

Solution (from research):
    A "Bluetooth Gateway" is placed at a fixed, known location in the room.
    Since the distance between the gateway and each beacon is constant, any
    change in the RSSI measured by the gateway must be due to the beacon's
    transmit power fluctuation — not physical movement.

    The server computes a real-time correction offset per beacon:

        ΔA_l = R_Ml - R̄_Ml

    where:
        R_Ml   = current RSSI from beacon l measured at the gateway
        R̄_Ml  = historical rolling average of gateway RSSI from beacon l
        ΔA_l   = correction offset (+ means beacon transmitting stronger)

    The mobile node then applies this correction to its own measurements:

        corrected_rssi = mobile_rssi - ΔA_l

    This effectively "normalizes" the transmit power before the corrected
    RSSI is fed into the PSO-LSTM distance model.

Usage:
    corrector = GatewayCorrection()

    # When gateway receives a beacon broadcast:
    corrector.update_gateway_rssi("STATION1", gateway_rssi=-62.0)

    # When mobile node receives the same beacon:
    clean_rssi = corrector.apply_correction("STATION1", mobile_rssi=-71.0)

    # The clean_rssi is then fed to the Kalman filter and LSTM pipeline.
"""

import numpy as np
from collections import defaultdict, deque
from config import GATEWAY_HISTORY_LEN


class GatewayCorrection:
    """
    Real-time RSSI correction using a fixed-position Bluetooth gateway.

    Maintains a rolling history of RSSI readings per beacon as observed
    by the gateway. Computes and applies correction offsets to mobile-node
    RSSI measurements before they enter the distance estimation pipeline.
    """

    def __init__(self, history_len: int = None):
        """
        Args:
            history_len: Rolling window size for gateway RSSI history.
                         Defaults to GATEWAY_HISTORY_LEN from config.
        """
        self.history_len = history_len or GATEWAY_HISTORY_LEN
        # gateway_history[beacon_id] = deque of recent gateway RSSI readings
        self.gateway_history: dict = defaultdict(
            lambda: deque(maxlen=self.history_len)
        )

    # ── Gateway Data Ingestion ────────────────────────────────────────────

    def update_gateway_rssi(self, beacon_id: str, gateway_rssi: float):
        """
        Record a new gateway RSSI measurement for a beacon.

        Called each time the fixed gateway detects a beacon broadcast.
        The rolling deque automatically discards readings older than
        history_len to keep the average representative of current conditions.

        Args:
            beacon_id   : station / beacon identifier (e.g. "STATION1")
            gateway_rssi: RSSI measured at the fixed gateway (dBm)
        """
        self.gateway_history[beacon_id].append(float(gateway_rssi))

    # ── Correction Offset Computation ────────────────────────────────────

    def get_historical_average(self, beacon_id: str) -> float | None:
        """
        Return R̄_Ml — the historical mean gateway RSSI for this beacon.

        Returns None if no history has been collected yet.
        """
        history = self.gateway_history[beacon_id]
        if not history:
            return None
        return float(np.mean(history))

    def get_correction_offset(self, beacon_id: str,
                               current_gateway_rssi: float) -> float:
        """
        Compute ΔA_l = R_Ml - R̄_Ml.

        Positive offset: beacon transmitting stronger than historical average.
            → mobile node RSSI is inflated → subtract offset to compensate.
        Negative offset: beacon transmitting weaker than historical average.
            → mobile node RSSI is deflated → subtract offset (adds back).

        Returns 0.0 if no historical data is available yet (no correction).
        """
        avg = self.get_historical_average(beacon_id)
        if avg is None:
            return 0.0
        return current_gateway_rssi - avg

    # ── Correction Application ────────────────────────────────────────────

    def apply_correction(self, beacon_id: str, mobile_rssi: float,
                          current_gateway_rssi: float = None) -> float:
        """
        Apply transmit-power correction to a mobile node's RSSI reading.

        Steps:
          1. If current_gateway_rssi provided, record it in history.
          2. Compute offset ΔA_l using the latest gateway reading.
          3. Return: corrected_rssi = mobile_rssi - ΔA_l

        Args:
            beacon_id             : station / beacon identifier
            mobile_rssi           : raw RSSI from mobile node (dBm)
            current_gateway_rssi  : optional, latest reading from gateway

        Returns:
            Corrected RSSI value clamped to valid range [-120, 0] dBm.
        """
        if current_gateway_rssi is not None:
            self.update_gateway_rssi(beacon_id, current_gateway_rssi)

        # Use the most recent gateway reading to compute offset
        history = self.gateway_history[beacon_id]
        if not history:
            return float(np.clip(mobile_rssi, -120.0, 0.0))

        latest_gateway = history[-1]
        offset         = self.get_correction_offset(beacon_id, latest_gateway)
        corrected      = mobile_rssi - offset
        return float(np.clip(corrected, -120.0, 0.0))

    # ── Utility ───────────────────────────────────────────────────────────

    def has_calibration(self, beacon_id: str, min_samples: int = 5) -> bool:
        """True if enough gateway readings exist for reliable correction."""
        return len(self.gateway_history[beacon_id]) >= min_samples

    def get_status(self) -> str:
        """Return calibration status summary for all monitored beacons."""
        if not self.gateway_history:
            return "No gateway data collected"
        parts = []
        for bid, hist in self.gateway_history.items():
            if hist:
                avg = float(np.mean(hist))
                parts.append(f"{bid}: {len(hist)} pts, avg={avg:.1f} dBm")
        return " | ".join(parts) if parts else "No gateway data collected"

    def reset(self, beacon_id: str = None):
        """
        Reset gateway history.
        If beacon_id is given, resets only that beacon.
        Otherwise resets all beacons.
        """
        if beacon_id:
            self.gateway_history[beacon_id].clear()
        else:
            self.gateway_history.clear()
