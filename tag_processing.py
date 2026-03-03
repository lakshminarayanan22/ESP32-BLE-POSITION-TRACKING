import numpy as np
from collections import deque
from config import SEQ_LEN, RSSI_MIN, RSSI_MAX

class DataProcessor:
    def __init__(self):
        # buffers[station_id][tag_id] = deque of normalized RSSI values
        self.buffers = {}

    # ── Normalize ────────────────────────────────────
    def normalize_rssi(self, rssi):
        return (rssi - RSSI_MIN) / (RSSI_MAX - RSSI_MIN)

    def denormalize_rssi(self, value):
        return value * (RSSI_MAX - RSSI_MIN) + RSSI_MIN

    # ── Parse Payload ─────────────────────────────────
    def parse_payload(self, payload: dict):
        """
        Input:
        {
            "stationId": "STATION14",
            "tags": [
                {"tagId": "bc:57:29:05:8e:d6", "rssi": -96},
                ...
            ]
        }
        Returns: (station_id, list of (tag_id, rssi))
        """
        station_id = payload.get("stationId", "UNKNOWN")
        tags       = payload.get("tags", [])

        parsed_tags = []
        for tag in tags:
            tag_id = tag.get("tagId", "").upper()
            rssi   = tag.get("rssi",  None)

            if not tag_id or rssi is None:
                continue
            if not (-120 <= rssi <= 0):
                print(f"[WARN] Invalid RSSI {rssi} for tag {tag_id}")
                continue

            parsed_tags.append((tag_id, rssi))

        return station_id, parsed_tags

    # ── Buffer Management ─────────────────────────────
    def update_buffer(self, station_id, tag_id, rssi):
        """
        Add one RSSI reading to the rolling buffer.
        Returns the sequence array if buffer is full, else None.
        """
        if station_id not in self.buffers:
            self.buffers[station_id] = {}

        if tag_id not in self.buffers[station_id]:
            self.buffers[station_id][tag_id] = deque(maxlen=SEQ_LEN)

        normalized = self.normalize_rssi(rssi)
        self.buffers[station_id][tag_id].append(normalized)

        buf_len = len(self.buffers[station_id][tag_id])

        if buf_len == SEQ_LEN:
            # shape: (SEQ_LEN, 1)
            sequence = np.array(
                list(self.buffers[station_id][tag_id]),
                dtype=np.float32
            ).reshape(SEQ_LEN, 1)
            return sequence

        return None   # not enough data yet

    def get_buffer_status(self):
        """Print how full each buffer is."""
        for station, tags in self.buffers.items():
            for tag, buf in tags.items():
                print(f"  {station} | {tag} | {len(buf)}/{SEQ_LEN}")