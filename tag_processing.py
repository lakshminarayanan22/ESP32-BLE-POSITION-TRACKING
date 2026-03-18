import numpy as np
from collections import deque, defaultdict
from config import (
    SEQ_LEN, RSSI_MIN, RSSI_MAX,
    ROOM_W, ROOM_H, ROOM_Z,
    STATION_POSITIONS
)

STATION_ORDER     = list(STATION_POSITIONS.keys())
STATS_PER_STATION = 5   # mean, std, min, max, count
# Total features = 4 stations × 5 = 20  (matches N_FEATURES in config)

# Per-station RSSI window size for computing statistics.
# Kept at 4 to match the training CSV columns (rssi_value_1..4).
RSSI_WINDOW_SIZE = 4


class DataProcessor:
    """
    Manages per-tag RSSI windows and builds sequential feature vectors
    for the PSO-LSTM positioning model.

    Two-level buffering:
        1. rssi_window[tag][station] : deque of last RSSI_WINDOW_SIZE raw
           (but gateway-corrected) readings per station — used to compute
           the 5-stat feature block [mean, std, min, max, count].

        2. buffers[tag] : deque of last SEQ_LEN 20-dim feature vectors —
           forms the temporal sliding window fed to the LSTM.
           SEQ_LEN=8 means the LSTM sees 8 consecutive time steps,
           capturing temporal RSSI patterns caused by shadowing events,
           multipath, and NLOS dynamics.

    Gateway Correction Integration:
        update_station_rssi() accepts an optional correction_offset.
        This offset is pre-computed by GatewayCorrection and represents
        the beacon transmit power deviation (ΔA_l = R_gateway - R̄_gateway).
        The corrected RSSI = raw_rssi - correction_offset is stored in the
        window so that all downstream statistics and LSTM inputs reflect
        stable, power-normalized signal values.
    """

    def __init__(self):
        self.rssi_window = defaultdict(
            lambda: {sid: [] for sid in STATION_ORDER}
        )
        self.buffers = {}   # tag_id → deque(maxlen=SEQ_LEN)

    # ── Normalization ─────────────────────────────────────────────────────

    def norm_rssi(self, v: float) -> float:
        return (v - RSSI_MIN) / (RSSI_MAX - RSSI_MIN)

    def denorm_x(self, v: float) -> float:
        return round(v * ROOM_W, 3)

    def denorm_y(self, v: float) -> float:
        return round(v * ROOM_H, 3)

    def denorm_z(self, v: float) -> float:
        return round(v * ROOM_Z, 3)

    # ── Compute 5 Stats From RSSI List ────────────────────────────────────

    def compute_station_stats(self, rssi_values: list) -> list:
        """
        Computes 5 normalized stats from gateway-corrected RSSI readings
        for one station.

        0 values → [0, 0, 0, 0, 0]
        1 value  → [mean, 0, min, max, 0.25]
        2 values → [mean, std, min, max, 0.50]
        3 values → [mean, std, min, max, 0.75]
        4 values → [mean, std, min, max, 1.00]
        """
        if not rssi_values:
            return [0.0, 0.0, 0.0, 0.0, 0.0]

        arr = np.array(rssi_values, dtype=np.float32)

        mean_v  = float(np.mean(arr))
        std_v   = float(np.std(arr))
        min_v   = float(np.min(arr))
        max_v   = float(np.max(arr))
        count_v = len(rssi_values)

        norm_mean  = self.norm_rssi(mean_v)
        norm_std   = std_v / ((RSSI_MAX - RSSI_MIN) / 2.0)
        norm_min   = self.norm_rssi(min_v)
        norm_max   = self.norm_rssi(max_v)
        norm_count = count_v / float(RSSI_WINDOW_SIZE)  # 0.25/0.5/0.75/1.0

        return [norm_mean, norm_std, norm_min, norm_max, norm_count]

    # ── Update RSSI From One Station ──────────────────────────────────────

    def update_station_rssi(self, station_id: str, tag_id: str,
                             rssi: float, correction_offset: float = 0.0):
        """
        Record a (gateway-corrected) RSSI reading for a tag from one station.

        The correction_offset is computed by GatewayCorrection:
            correction_offset = ΔA_l = R_gateway_current - R_gateway_historical_avg

        corrected_rssi = raw_rssi - correction_offset

        This normalizes beacon transmit power fluctuations before the value
        enters the statistics window and the LSTM feature pipeline.

        Args:
            station_id        : e.g. "STATION1"
            tag_id            : BLE tag MAC address
            rssi              : raw RSSI reading from mobile node (dBm)
            correction_offset : ΔA_l from GatewayCorrection (default 0 = no correction)
        """
        tag_id = tag_id.lower()

        if tag_id not in self.rssi_window:
            self.rssi_window[tag_id] = {sid: [] for sid in STATION_ORDER}

        corrected = float(np.clip(rssi - correction_offset, -120.0, 0.0))

        window = self.rssi_window[tag_id][station_id]
        window.append(corrected)

        # Keep only last RSSI_WINDOW_SIZE readings (matches training CSV columns)
        if len(window) > RSSI_WINDOW_SIZE:
            self.rssi_window[tag_id][station_id] = window[-RSSI_WINDOW_SIZE:]

    # ── Build 20-dim Feature Vector ───────────────────────────────────────

    def build_feature_vector(self, tag_id: str) -> list:
        """
        Assemble 20-dim feature vector from current corrected RSSI window:
            [mean, std, min, max, count] × 4 stations = 20 values

        Stations with no readings → 5 zeros.
        All values normalized to [0, 1].
        """
        tag_id  = tag_id.lower()
        window  = self.rssi_window.get(
            tag_id, {sid: [] for sid in STATION_ORDER}
        )

        feature_vec = []
        for sid in STATION_ORDER:
            readings = window.get(sid, [])
            stats    = self.compute_station_stats(readings)
            feature_vec.extend(stats)   # 5 values per station

        return feature_vec   # exactly 20 floats

    # ── Push to Temporal Buffer ───────────────────────────────────────────

    def push_to_buffer(self, tag_id: str):
        """
        Build a 20-dim feature vector and append it to the temporal buffer.

        The buffer holds the last SEQ_LEN (=8) feature vectors, forming the
        sliding window that the LSTM uses to capture temporal RSSI dynamics.

        Returns:
            np.array shape (SEQ_LEN, 20) — when buffer is full (ready for LSTM)
            None                          — still collecting initial readings
        """
        tag_id = tag_id.lower()

        if tag_id not in self.buffers:
            self.buffers[tag_id] = deque(maxlen=SEQ_LEN)

        feature_vec = self.build_feature_vector(tag_id)
        self.buffers[tag_id].append(feature_vec)

        if len(self.buffers[tag_id]) == SEQ_LEN:
            return np.array(
                list(self.buffers[tag_id]),
                dtype=np.float32
            )   # shape: (SEQ_LEN, 20)

        return None

    # ── Buffer Status ─────────────────────────────────────────────────────

    def get_buffer_status(self, tag_id: str) -> str:
        tag_id = tag_id.lower()
        buf    = self.buffers.get(tag_id, deque())
        return f"{len(buf)}/{SEQ_LEN}"

    # ── Station Coverage ──────────────────────────────────────────────────

    def get_station_coverage(self, tag_id: str) -> str:
        """
        Show latest mean corrected RSSI per station for this tag.
        Example: STATION1:-67dBm | STATION2:--- | ...
        """
        tag_id = tag_id.lower()
        window = self.rssi_window.get(
            tag_id, {sid: [] for sid in STATION_ORDER}
        )
        parts = []
        for sid in STATION_ORDER:
            readings = window.get(sid, [])
            if readings:
                mean_rssi = int(round(sum(readings) / len(readings)))
                parts.append(f"{sid}:{mean_rssi}dBm")
            else:
                parts.append(f"{sid}:---")
        return " | ".join(parts)

    # ── Parse MQTT Payload ────────────────────────────────────────────────

    def parse_payload(self, payload: dict):
        """
        Parse incoming MQTT payload.

        Type A (tag data):
          {"stationId":"STATION1", "tags":[{"tagId":"...","rssi":-67},...]}

        Type B (status only — returns empty list):
          {"stationId":"STATION3","errorCode":0,"rpm":0}

        Returns: (station_id, [(tag_id, rssi), ...])
        """
        station_id = payload.get("stationId", "").upper()
        tags       = payload.get("tags", [])

        if station_id not in STATION_POSITIONS:
            return station_id, []

        if not tags:
            return station_id, []

        valid_tags = []
        for tag in tags:
            tag_id = tag.get("tagId", "").lower().strip()
            rssi   = tag.get("rssi", None)

            if not tag_id or rssi is None:
                continue
            if not (-120 <= rssi <= 0):
                continue

            valid_tags.append((tag_id, float(rssi)))

        return station_id, valid_tags

    # ── All Tracked Tags ──────────────────────────────────────────────────

    def get_all_tags(self) -> list:
        return list(self.rssi_window.keys())


"""
Feature Pipeline Summary
========================

Training feature vector (20 values per time step, SEQ_LEN=8 steps):
    [mean, std, min, max, count] × 4 stations

Gateway correction is applied before RSSI enters the window:
    corrected_rssi = raw_rssi - ΔA_l
    where ΔA_l = R_gateway_current - R_gateway_historical_avg

LSTM input shape: (batch, 8, 20)
    - 8 time steps of corrected, power-normalized RSSI statistics
    - Enables the LSTM cell state to capture temporal signal dynamics

Station window (RSSI_WINDOW_SIZE=4) is for intra-timestep stats.
Temporal window (SEQ_LEN=8) is for inter-timestep LSTM context.
"""
