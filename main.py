import json
import csv
import time
import os
import threading
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import paho.mqtt.client as mqtt
from scipy.optimize import least_squares

from config import (
    MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD,
    STATION_IDS, MQTT_TOPIC_PATTERN,
    STATION_POSITIONS, ROOM_W, ROOM_H, ROOM_Z,
    SEQ_LEN, RSSI_MIN, TARGET_TAG
)
from tag_processing import DataProcessor
from rssi_predictor import PositionPredictor

# ── Globals ───────────────────────────────────────────
processor     = DataProcessor()
predictor     = PositionPredictor()
results       = {}          # tag_id → latest LSTM prediction
lock          = threading.Lock()
STATION_ORDER = list(STATION_POSITIONS.keys())


# ── Trilateration Config ──────────────────────────────
TX_POWER  = -65
PATH_LOSS = 2.0

# Station positions for trilateration (x, y) in meters
# Stations placed at center of each wall
STATION_COORDS = {
    "STATION1": (ROOM_W / 2, ROOM_H),
    "STATION2": (ROOM_W,     ROOM_H / 2),
    "STATION3": (ROOM_W / 2, 0),
    "STATION4": (0,          ROOM_H / 2),
}

# ── Per-tag RSSI storage for trilateration ────────────
# latest_rssi[tag_id][station_id] = rssi
latest_rssi    = defaultdict(dict)
rssi_lock      = threading.Lock()

# ── RSSI Kalman Filter (1D per station per tag) ───────
kalman_state   = defaultdict(lambda: defaultdict(lambda: None))
kalman_cov     = defaultdict(lambda: defaultdict(lambda: 1.0))
PROCESS_NOISE  = 0.1
MEAS_NOISE     = 10.0

def kalman_filter_rssi(tag_id, station, measurement):
    if kalman_state[tag_id][station] is None:
        kalman_state[tag_id][station] = measurement

    pred_state = kalman_state[tag_id][station]
    pred_cov   = kalman_cov[tag_id][station] + PROCESS_NOISE
    K          = pred_cov / (pred_cov + MEAS_NOISE)

    kalman_state[tag_id][station] = pred_state + K * (measurement - pred_state)
    kalman_cov[tag_id][station]   = (1 - K) * pred_cov

    return kalman_state[tag_id][station]

# ── RSSI → Distance ───────────────────────────────────
def rssi_to_distance(rssi):
    return 10 ** ((TX_POWER - rssi) / (10 * PATH_LOSS))

# ── Nonlinear Least Squares Trilateration ─────────────
def nonlinear_trilateration(distances: dict):
    if len(distances) < 3:
        return None

    def residuals(pos):
        x, y = pos
        errors = []
        for station, dist in distances.items():
            sx, sy = STATION_COORDS[station]
            calc   = np.sqrt((x - sx)**2 + (y - sy)**2)
            errors.append(calc - dist)
        return errors

    result = least_squares(
        residuals,
        x0 = [ROOM_W / 2, ROOM_H / 2]
    )
    x = max(0, min(ROOM_W, result.x[0]))
    y = max(0, min(ROOM_H, result.x[1]))
    return x, y

# ── 2D Kalman Filter (position smoothing) ─────────────
dt = 0.5

# One 2D Kalman state per tag
kalman_2d_state = {}
kalman_2d_P     = {}

F = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1,  0],
    [0, 0, 0,  1]
])
H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]])
Q = np.eye(4) * 0.02
R = np.eye(2) * 0.3

def kalman_2d(tag_id, measurement):
    if tag_id not in kalman_2d_state:
        kalman_2d_state[tag_id] = np.array(
            [measurement[0], measurement[1], 0.0, 0.0]
        )
        kalman_2d_P[tag_id] = np.eye(4)

    state = kalman_2d_state[tag_id]
    P     = kalman_2d_P[tag_id]

    # Predict
    state = F @ state
    P     = F @ P @ F.T + Q

    # Update
    z = np.array(measurement)
    y = z - H @ state
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    state = state + K @ y
    P     = (np.eye(4) - K @ H) @ P

    kalman_2d_state[tag_id] = state
    kalman_2d_P[tag_id]     = P

    return float(state[0]), float(state[1])

# ── Tag Position Storage ──────────────────────────────
# tag_positions[tag_id] = {
#     "trilat":  (x, y)      ← trilateration result
#     "lstm":    (x, y, z)   ← LSTM prediction
#     "kalman":  (x, y)      ← kalman smoothed trilat
# }
tag_positions = {}
pos_lock      = threading.Lock()

# ─────────────────────────────────────────────────────
# SECTION 1 — PROCESS PAYLOAD
# ─────────────────────────────────────────────────────

def process_payload(payload: dict):
    station_id = payload.get("stationId", "").upper()
    tags       = payload.get("tags", [])

    if not tags:
        return

    if station_id not in STATION_POSITIONS:
        print(f"[WARN] Unknown station: {station_id}")
        return

    for tag in tags:
        tag_id = tag.get("tagId", "").lower().strip()
        rssi   = tag.get("rssi", None)

        # ── Filter: only process TARGET_TAG ───────────
        if tag_id != TARGET_TAG.lower():
            continue
        # ──────────────────────────────────────────────

        if not tag_id or rssi is None:
            continue
        if not (-120 <= rssi <= 0):
            continue

        # Store raw RSSI
        with rssi_lock:
            latest_rssi[tag_id][station_id] = float(rssi)

        # LSTM: update buffer + predict
        processor.update_station_rssi(station_id, tag_id, rssi)
        sequence = processor.push_to_buffer(tag_id)

        if sequence is not None:
            lstm_result = predictor.predict(sequence)
            with lock:
                results[tag_id] = {
                    "station":  station_id,
                    "rssi":     rssi,
                    "coverage": processor.get_station_coverage(tag_id),
                    **lstm_result
                }
            with pos_lock:
                if tag_id not in tag_positions:
                    tag_positions[tag_id] = {}
                tag_positions[tag_id]["lstm"] = (
                    lstm_result["x"],
                    lstm_result["y"],
                    lstm_result["z"]
                )

    # Trilateration — only for TARGET_TAG
    target = TARGET_TAG.lower()
    with rssi_lock:
        if target not in latest_rssi:
            return
        tag_rssi = latest_rssi[target].copy()

    if len(tag_rssi) < 3:
        return

    distances = {}
    for sid, r in tag_rssi.items():
        if sid not in STATION_COORDS:
            continue
        filtered      = kalman_filter_rssi(target, sid, r)
        distances[sid] = rssi_to_distance(filtered)

    trilat_pos = nonlinear_trilateration(distances)
    if trilat_pos is None:
        return

    smooth_pos = kalman_2d(target, trilat_pos)

    with pos_lock:
        if target not in tag_positions:
            tag_positions[target] = {}
        tag_positions[target]["trilat"] = trilat_pos
        tag_positions[target]["kalman"] = smooth_pos
# ─────────────────────────────────────────────────────
# SECTION 2 — MQTT CALLBACKS
# ─────────────────────────────────────────────────────

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"[MQTT] Connected → {MQTT_BROKER}:{MQTT_PORT}")
        for sid in STATION_IDS:
            topic = MQTT_TOPIC_PATTERN.format(sid)
            client.subscribe(topic)
            print(f"[MQTT] Subscribed → {topic}")
        print()
    else:
        print(f"[MQTT] Failed rc={reason_code}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        process_payload(payload)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Bad JSON: {e}")
    except Exception as e:
        print(f"[ERROR] {e}")

def on_disconnect(client, userdata, rc, properties=None):
    print(f"[MQTT] Disconnected rc={rc}")


# ─────────────────────────────────────────────────────
# SECTION 3 — MQTT THREAD
# ─────────────────────────────────────────────────────

def start_mqtt():
    """Runs in background thread — handles all MQTT communication."""
    
    # Allow main thread and network stack to fully initialize
    time.sleep(2)
    
    while True:
        try:
            # Force DNS/socket resolution before paho touches it
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(10)
            sock.connect((MQTT_BROKER, MQTT_PORT))
            sock.close()
            print(f"[MQTT] Socket pre-check passed → {MQTT_BROKER}:{MQTT_PORT}")

            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id="position_predictor_3d"
            )
            client.on_connect    = on_connect
            client.on_message    = on_message
            client.on_disconnect = on_disconnect

            if MQTT_USERNAME:
                client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

            client.connect(
                MQTT_BROKER,
                MQTT_PORT,
                keepalive = 60
            )
            client.loop_forever()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[MQTT] Error: {e} — retrying in 5s...")
            time.sleep(5)

# ─────────────────────────────────────────────────────
# SECTION 4 — REAL-TIME PLOT
# ─────────────────────────────────────────────────────

# Tag colors for distinguishing multiple tags
TAG_COLORS = [
    "red", "blue", "green", "orange",
    "purple", "cyan", "magenta", "brown"
]
tag_color_map = {}
color_idx     = [0]

def get_tag_color(tag_id):
    if tag_id not in tag_color_map:
        tag_color_map[tag_id] = TAG_COLORS[
            color_idx[0] % len(TAG_COLORS)
        ]
        color_idx[0] += 1
    return tag_color_map[tag_id]


def run_plot():
    """
    Real-time matplotlib plot showing:
    Left  — room map with tag positions (trilat + LSTM)
    Right — live RSSI table per tag per station
    """
    plt.ion()
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle("BLE RTLS — Trilateration + LSTM 3D Position",
                 fontsize=13)
    gs  = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    ax_room  = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    print("[PLOT] Real-time display started. Press Ctrl+C to stop.")

    while True:
        try:
            ax_room.clear()
            ax_table.clear()

            # ── Draw Room ─────────────────────────────
            ax_room.plot(
                [0, ROOM_W, ROOM_W, 0, 0],
                [0, 0, ROOM_H, ROOM_H, 0],
                "k-", linewidth=2
            )
            ax_room.set_xlim(-0.5, ROOM_W + 0.5)
            ax_room.set_ylim(-0.5, ROOM_H + 0.5)
            ax_room.set_aspect("equal")
            ax_room.set_xlabel("X (m)")
            ax_room.set_ylabel("Y (m)")
            ax_room.set_title(
                f"Room {ROOM_W}m × {ROOM_H}m × {ROOM_Z}m\n"
                f"▲ = Station  ● = Trilat  ★ = LSTM"
            )
            ax_room.grid(True, alpha=0.3)

            # ── Draw Stations ─────────────────────────
            for sid, (sx, sy) in STATION_COORDS.items():
                ax_room.plot(sx, sy, "k^", markersize=12)
                ax_room.annotate(
                    sid,
                    (sx, sy),
                    textcoords="offset points",
                    xytext=(6, 6),
                    fontsize=8,
                    fontweight="bold"
                )

            # ── Draw Tag Positions ────────────────────
            with pos_lock:
                positions_snapshot = {
                    k: v.copy() for k, v in tag_positions.items()
                }

            for tag_id, pos in positions_snapshot.items():
                color    = get_tag_color(tag_id)
                short_id = tag_id[-5:]   # last 5 chars for label

                # Trilateration position (circle)
                if "kalman" in pos:
                    kx, ky = pos["kalman"]
                    ax_room.scatter(
                        kx, ky,
                        s=120, c=color,
                        marker="o", zorder=5,
                        label=f"{short_id} trilat"
                    )
                    ax_room.annotate(
                        f"{short_id}\n({kx:.1f},{ky:.1f})",
                        (kx, ky),
                        textcoords="offset points",
                        xytext=(8, 8),
                        fontsize=7,
                        color=color
                    )

                # LSTM position (star)
                if "lstm" in pos:
                    lx, ly, lz = pos["lstm"]
                    ax_room.scatter(
                        lx, ly,
                        s=200, c=color,
                        marker="*", zorder=6,
                        label=f"{short_id} LSTM"
                    )
                    ax_room.annotate(
                        f"LSTM\n({lx:.1f},{ly:.1f},{lz:.1f}m)",
                        (lx, ly),
                        textcoords="offset points",
                        xytext=(8, -16),
                        fontsize=7,
                        color=color,
                        style="italic"
                    )

                # Draw line between trilat and LSTM
                if "kalman" in pos and "lstm" in pos:
                    kx, ky    = pos["kalman"]
                    lx, ly, _ = pos["lstm"]
                    ax_room.plot(
                        [kx, lx], [ky, ly],
                        "--", color=color,
                        alpha=0.4, linewidth=1
                    )

            if positions_snapshot:
                ax_room.legend(
                    loc="upper right",
                    fontsize=7,
                    framealpha=0.8
                )

            # ── RSSI Table ────────────────────────────
            with rssi_lock:
                rssi_snapshot = {
                    k: v.copy() for k, v in latest_rssi.items()
                }
            with lock:
                lstm_snapshot = results.copy()

            # Build table rows
            table_data  = []
            col_headers = (
                ["Tag"] +
                [s.replace("STATION", "S") for s in STATION_ORDER] +
                ["X", "Y", "Z"]
            )

            for tag_id, station_rssi in rssi_snapshot.items():
                short_id = tag_id[-11:]
                rssi_vals = [
                    str(int(station_rssi.get(sid, 0)))
                    if sid in station_rssi else "—"
                    for sid in STATION_ORDER
                ]
                lstm_res = lstm_snapshot.get(tag_id, {})
                x_str = f"{lstm_res.get('x', 0):.1f}"
                y_str = f"{lstm_res.get('y', 0):.1f}"
                z_str = f"{lstm_res.get('z', 0):.1f}"

                table_data.append(
                    [short_id] + rssi_vals + [x_str, y_str, z_str]
                )

            ax_table.axis("off")

            if table_data:
                table = ax_table.table(
                    cellText   = table_data,
                    colLabels  = col_headers,
                    loc        = "center",
                    cellLoc    = "center"
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.1, 1.6)

                # Header row styling
                for col in range(len(col_headers)):
                    table[0, col].set_facecolor("#2c3e50")
                    table[0, col].set_text_props(color="white",
                                                  fontweight="bold")
                # Alternate row colors
                for row in range(1, len(table_data) + 1):
                    color_bg = "#ecf0f1" if row % 2 == 0 else "white"
                    for col in range(len(col_headers)):
                        table[row, col].set_facecolor(color_bg)
            else:
                ax_table.text(
                    0.5, 0.5,
                    "Waiting for\nMQTT data...",
                    ha="center", va="center",
                    fontsize=12, color="gray",
                    transform=ax_table.transAxes
                )

            ax_table.set_title("Live RSSI + LSTM Position",
                               fontsize=10)

            plt.tight_layout()
            plt.pause(0.5)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[PLOT] Error: {e}")
            time.sleep(0.5)

    plt.close()


# ─────────────────────────────────────────────────────
# SECTION 5 — SUMMARY THREAD
# ─────────────────────────────────────────────────────

def print_summary():
    print("\n" + "=" * 80)
    print("  POSITION SUMMARY  —  x, y, z in meters")
    print("=" * 80)

    with lock:
        if not results:
            print(f"  No LSTM predictions yet — "
                  f"need {SEQ_LEN} readings per tag")
        else:
            print(
                f"  {'TAG ID':<22}  {'X':>6}  {'Y':>6}  {'Z':>6}  "
                f"{'ZONE':<14}  CONFIDENCE"
            )
            print("  " + "-" * 70)
            for tag_id, res in results.items():
                print(
                    f"  {tag_id:<22}  "
                    f"{res.get('x', 0):>6.2f}  "
                    f"{res.get('y', 0):>6.2f}  "
                    f"{res.get('z', 0):>6.2f}  "
                    f"{res.get('zone', '?'):<14}  "
                    f"{res.get('confidence', '?')}"
                )

    with pos_lock:
        if tag_positions:
            print(f"\n  TRILATERATION:")
            print(f"  {'TAG ID':<22}  {'TRILAT X':>10}  "
                  f"{'TRILAT Y':>10}  {'KALMAN X':>10}  {'KALMAN Y':>10}")
            print("  " + "-" * 70)
            for tag_id, pos in tag_positions.items():
                tx = f"{pos['trilat'][0]:.2f}" if "trilat" in pos else "—"
                ty = f"{pos['trilat'][1]:.2f}" if "trilat" in pos else "—"
                kx = f"{pos['kalman'][0]:.2f}" if "kalman" in pos else "—"
                ky = f"{pos['kalman'][1]:.2f}" if "kalman" in pos else "—"
                print(f"  {tag_id:<22}  {tx:>10}  {ty:>10}  "
                      f"{kx:>10}  {ky:>10}")

    print("=" * 80 + "\n")


def print_summary_loop():
    while True:
        time.sleep(30)
        print_summary()


# ─────────────────────────────────────────────────────
# SECTION 6 — MAIN
# ─────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  BLE RTLS — Trilateration + LSTM 3D Predictor")
    print(f"  Broker    : {MQTT_BROKER}:{MQTT_PORT}")
    print(f"  Room      : {ROOM_W}m × {ROOM_H}m × {ROOM_Z}m")
    print(f"  SEQ_LEN   : {SEQ_LEN} readings per tag")
    print(f"  Stations  : {STATION_ORDER}")
    print("=" * 60 + "\n")

    # Background MQTT thread
    mqtt_thread = threading.Thread(
        target=start_mqtt,
        daemon=True
    )
    mqtt_thread.start()

    # Background summary thread
    threading.Thread(
        target=print_summary_loop,
        daemon=True
    ).start()

    # Main thread runs the plot (must be on main thread on macOS)
    try:
        run_plot()
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down...")

"""


## What the Plot Shows

┌─────────────────────────────┬──────────────────────┐
│                             │  Tag  S1  S2  S3  S4 │
│   ▲ STATION1                │  :cf -67 -61  —  -72 │
│                             │  :70 -71  —  -69 -74 │
│         ● trilat            │                      │
│         ★ LSTM              │  X    Y    Z         │
│   ▲ STATION4   ▲ STATION2   │  2.1  1.8  1.2       │
│                             │                      │
│         ▲ STATION3          │                      │
└─────────────────────────────┴──────────────────────┘
  ● = Kalman-smoothed trilateration position
  ★ = LSTM predicted position
  dashed line = difference between the two methods

"""

