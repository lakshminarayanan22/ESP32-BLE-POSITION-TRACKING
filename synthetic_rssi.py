import json
import random
import math
import os

# ── Configuration ─────────────────────────────────────
NUM_PAYLOADS   = 100          # number of time steps (100 × 3s = 5 mins of data)
INTERVAL_S     = 3            # seconds between each payload
OUTPUT_FILE    = "/Users/lnarayanansk/Documents/Doodil2r/beacon_data.json"

STATION_ID     = "STATION14"

# Fixed tag IDs matching your real format
TAG_IDS = [
    "bc:57:29:05:8e:d6",
    "ca:15:ac:2e:e6:fc",
    "fb:de:e9:40:fb:2b",
    "54:ab:f6:5e:4e:03",
    "dc:0a:e4:d2:da:e3",
    "cb:fa:84:59:d7:06",
    "23:97:5e:7f:36:45",
]

# ── Tag Behavior Profiles ─────────────────────────────
# Each tag has a base RSSI and a movement pattern
TAG_PROFILES = {
    "bc:57:29:05:8e:d6": {"base": -96, "pattern": "stationary",  "noise": 3},
    "ca:15:ac:2e:e6:fc": {"base": -50, "pattern": "approaching", "noise": 4},
    "fb:de:e9:40:fb:2b": {"base": -46, "pattern": "moving_away", "noise": 4},
    "54:ab:f6:5e:4e:03": {"base": -67, "pattern": "oscillating", "noise": 5},
    "dc:0a:e4:d2:da:e3": {"base": -57, "pattern": "random_walk", "noise": 6},
    "cb:fa:84:59:d7:06": {"base": -91, "pattern": "stationary",  "noise": 2},
    "23:97:5e:7f:36:45": {"base": -95, "pattern": "oscillating", "noise": 3},
}

# ── RSSI Generator Per Pattern ────────────────────────
def generate_rssi_series(profile: dict, n_steps: int) -> list:
    """
    Generate a realistic RSSI time series based on movement pattern.

    Patterns:
      stationary  — small random noise around fixed base
      approaching — RSSI gradually increases (tag moves closer)
      moving_away — RSSI gradually decreases (tag moves away)
      oscillating — tag moves back and forth (sinusoidal)
      random_walk — RSSI drifts randomly (unpredictable movement)
    """
    base    = profile["base"]
    noise   = profile["noise"]
    pattern = profile["pattern"]
    series  = []

    current = base

    for step in range(n_steps):
        t = step / n_steps   # normalized time 0.0 → 1.0

        if pattern == "stationary":
            # Small random noise around base
            rssi = base + random.gauss(0, noise)

        elif pattern == "approaching":
            # Moves from base RSSI to base+30 (closer)
            trend = 30 * t
            rssi  = base + trend + random.gauss(0, noise)

        elif pattern == "moving_away":
            # Moves from base RSSI to base-25 (farther)
            trend = -25 * t
            rssi  = base + trend + random.gauss(0, noise)

        elif pattern == "oscillating":
            # Sinusoidal movement — 2 full cycles across dataset
            swing = 15 * math.sin(2 * math.pi * 2 * t)
            rssi  = base + swing + random.gauss(0, noise)

        elif pattern == "random_walk":
            # Each step drifts from previous value
            drift   = random.gauss(0, noise * 0.5)
            current = current + drift
            # Slowly pull back toward base (mean reversion)
            current = current + 0.05 * (base - current)
            rssi    = current

        # Clamp to valid RSSI range
        rssi = max(-120, min(-20, rssi))
        series.append(int(round(rssi)))

    return series


# ── Occasional Tag Dropout ────────────────────────────
def apply_dropout(series: list, dropout_prob: float = 0.05) -> list:
    """
    Randomly drop tag from some payloads (simulates real BLE packet loss).
    Returns None for dropped steps.
    """
    return [val if random.random() > dropout_prob else None for val in series]


# ── Generate Full Dataset ─────────────────────────────
def generate_dataset(num_payloads: int) -> list:
    print(f"[GEN] Generating {num_payloads} payloads for {len(TAG_IDS)} tags...")

    # Pre-generate full RSSI series for each tag
    tag_series = {}
    for tag_id in TAG_IDS:
        profile          = TAG_PROFILES[tag_id]
        series           = generate_rssi_series(profile, num_payloads)
        tag_series[tag_id] = apply_dropout(series, dropout_prob=0.03)

    # Build payload list
    payloads = []
    for step in range(num_payloads):
        tags = []
        for tag_id in TAG_IDS:
            rssi = tag_series[tag_id][step]
            if rssi is None:
                continue   # tag dropped out this step
            tags.append({
                "tagId": tag_id,
                "rssi":  rssi
            })

        payload = {
            "stationId": STATION_ID,
            "tags":      tags
        }
        payloads.append(payload)

    return payloads


# ── Save to File ──────────────────────────────────────
def save_dataset(payloads: list, filepath: str):
    with open(filepath, "w") as f:
        json.dump(payloads, f, indent=2)
    print(f"[GEN] Saved {len(payloads)} payloads to {filepath}")
    print(f"[GEN] File size: {os.path.getsize(filepath) / 1024:.1f} KB")


# ── Print Sample ──────────────────────────────────────
def print_sample(payloads: list, n: int = 3):
    print(f"\n[GEN] Sample (first {n} payloads):")
    for i, p in enumerate(payloads[:n]):
        print(f"\n  Payload {i+1}:")
        print(f"  stationId: {p['stationId']}")
        for tag in p["tags"]:
            profile = TAG_PROFILES.get(tag["tagId"], {})
            pattern = profile.get("pattern", "unknown")
            print(f"    {tag['tagId']}  rssi={tag['rssi']:4d} dBm  [{pattern}]")


# ── RSSI Statistics ───────────────────────────────────
def print_statistics(payloads: list):
    print(f"\n[GEN] RSSI Statistics across {len(payloads)} payloads:")
    print(f"  {'TAG ID':<22}  {'MIN':>5}  {'MAX':>5}  {'AVG':>7}  {'PATTERN':<15}  DROPOUTS")
    print("  " + "-" * 75)

    for tag_id in TAG_IDS:
        rssi_values = [
            tag["rssi"]
            for p in payloads
            for tag in p["tags"]
            if tag["tagId"] == tag_id
        ]

        if not rssi_values:
            continue

        dropouts = len(payloads) - len(rssi_values)
        pattern  = TAG_PROFILES[tag_id]["pattern"]

        print(
            f"  {tag_id:<22}  "
            f"{min(rssi_values):>5}  "
            f"{max(rssi_values):>5}  "
            f"{sum(rssi_values)/len(rssi_values):>7.1f}  "
            f"{pattern:<15}  "
            f"{dropouts}"
        )


# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  Synthetic BLE Dataset Generator")
    print(f"  Payloads  : {NUM_PAYLOADS}")
    print(f"  Tags      : {len(TAG_IDS)}")
    print(f"  Duration  : ~{NUM_PAYLOADS * INTERVAL_S}s of simulated data")
    print("=" * 60 + "\n")

    payloads = generate_dataset(NUM_PAYLOADS)

    print_sample(payloads, n=3)
    print_statistics(payloads)
    save_dataset(payloads, OUTPUT_FILE)

    print(f"\n[GEN] Done! Run main.py to process {OUTPUT_FILE}")