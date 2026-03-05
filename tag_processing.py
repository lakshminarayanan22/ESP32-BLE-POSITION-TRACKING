import numpy as np
from collections import deque, defaultdict
from config import (
    SEQ_LEN, RSSI_MIN, RSSI_MAX,
    ROOM_W, ROOM_H, ROOM_Z,
    STATION_POSITIONS
)

STATION_ORDER    = list(STATION_POSITIONS.keys())
STATS_PER_STATION = 5   # mean, std, min, max, count
# Total features = 4 stations × 5 = 20  (matches N_FEATURES in config)


class DataProcessor:
    def __init__(self):
        """
        rssi_window[tag_id][station_id] = deque of recent rssi floats
            → accumulates readings within current interval
            → used to compute mean, std, min, max, count

        buffers[tag_id] = deque of feature vectors (maxlen=SEQ_LEN)
            → each entry is 20 floats
            → when full → fed to LSTM
        """
        self.rssi_window = defaultdict(
            lambda: {sid: [] for sid in STATION_ORDER}
        )
        self.buffers = {}   # tag_id → deque(maxlen=SEQ_LEN)

    # ── Normalization ─────────────────────────────────
    def norm_rssi(self, v: float) -> float:
        return (v - RSSI_MIN) / (RSSI_MAX - RSSI_MIN)

    def denorm_x(self, v: float) -> float:
        return round(v * ROOM_W, 3)

    def denorm_y(self, v: float) -> float:
        return round(v * ROOM_H, 3)

    def denorm_z(self, v: float) -> float:
        return round(v * ROOM_Z, 3)

    # ── Compute 5 Stats From RSSI List ────────────────
    def compute_station_stats(self, rssi_values: list) -> list:
        """
        Computes 5 normalized stats from however many
        non-null RSSI readings exist for one station.

        No padding — uses only actual values.

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

        # Normalize
        norm_mean  = self.norm_rssi(mean_v)
        norm_std   = std_v / ((RSSI_MAX - RSSI_MIN) / 2.0)
        norm_min   = self.norm_rssi(min_v)
        norm_max   = self.norm_rssi(max_v)
        norm_count = count_v / 4.0    # max 4 readings → 0.25/0.5/0.75/1.0

        return [norm_mean, norm_std, norm_min, norm_max, norm_count]

    # ── Update RSSI From One Station ──────────────────
    def update_station_rssi(self, station_id: str,
                             tag_id: str, rssi: float):
        """
        Called each time a station publishes a reading.
        Appends RSSI to the rolling window for this
        station/tag pair.

        Window keeps last 4 readings per station
        (matches training: max 4 rssi_value columns).
        """
        tag_id = tag_id.lower()

        if tag_id not in self.rssi_window:
            self.rssi_window[tag_id] = {
                sid: [] for sid in STATION_ORDER
            }

        window = self.rssi_window[tag_id][station_id]
        window.append(float(rssi))

        # Keep only last 4 readings — matches training CSV
        if len(window) > 4:
            self.rssi_window[tag_id][station_id] = window[-4:]

    # ── Build 20-dim Feature Vector ───────────────────
    def build_feature_vector(self, tag_id: str) -> list:
        """
        Assembles 20-dim feature vector from current
        RSSI window for all stations:

        [mean,std,min,max,count] × 4 stations = 20 values

        Exactly matches training feature construction.
        Stations with no readings → 5 zeros.
        """
        tag_id  = tag_id.lower()
        window  = self.rssi_window.get(
            tag_id,
            {sid: [] for sid in STATION_ORDER}
        )

        feature_vec = []
        for sid in STATION_ORDER:
            readings = window.get(sid, [])
            stats    = self.compute_station_stats(readings)
            feature_vec.extend(stats)   # 5 values per station

        # feature_vec is exactly 20 floats
        return feature_vec

    # ── Push to Buffer ────────────────────────────────
    def push_to_buffer(self, tag_id: str):
        """
        Builds 20-dim feature vector and pushes into
        rolling buffer for this tag.

        Returns:
            np.array shape (SEQ_LEN, 20) — when buffer full
            None — still collecting
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

    # ── Buffer Status ─────────────────────────────────
    def get_buffer_status(self, tag_id: str) -> str:
        tag_id = tag_id.lower()
        buf    = self.buffers.get(tag_id, deque())
        return f"{len(buf)}/{SEQ_LEN}"

    # ── Station Coverage ──────────────────────────────
    def get_station_coverage(self, tag_id: str) -> str:
        """
        Shows latest mean RSSI per station for this tag.
        Example: STATION1:-67dBm | STATION2:--- | ...
        """
        tag_id = tag_id.lower()
        window = self.rssi_window.get(
            tag_id,
            {sid: [] for sid in STATION_ORDER}
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

    # ── Parse MQTT Payload ────────────────────────────
    def parse_payload(self, payload: dict):
        """
        Parse incoming MQTT payload.

        Type A (tag data):
          {"stationId":"STATION1",
           "tags":[{"tagId":"...","rssi":-67},...]}

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

    # ── All Tracked Tags ──────────────────────────────
    def get_all_tags(self) -> list:
        return list(self.rssi_window.keys())


"""

Training feature vector (20 values):
    [mean, std, min, max, count] × 4 stations

Old data_processor sent (4 values):
    [norm_rssi_s1, norm_rssi_s2, norm_rssi_s3, norm_rssi_s4]
    → LSTM expected 20, got 4 → crash

New data_processor sends (20 values):
    STATION1: [mean, std, min, max, count]  ← from last 4 readings
    STATION2: [mean, std, min, max, count]
    STATION3: [mean, std, min, max, count]
    STATION4: [mean, std, min, max, count]
    → exactly matches training → no crash

"""
