"""
BLE Indoor Positioning System — Real-Time Inference
=====================================================

Pipeline:
    MQTT → Gateway Correction → Kalman RSSI Filter → WLS Trilateration
                                                   → Kalman 2D Smoother
         → Gateway-Corrected RSSI → DataProcessor  → PSO-LSTM → 3D Position

Positioning Methods:
    1. Weighted Least Squares (WLS) Trilateration [primary geometric method]
       - Converts RSSI to distance via log-distance path loss
       - Linearizes the overdetermined circle intersection system
       - Assigns inverse-distance-squared weights to each anchor equation
       - Solves X = (A^T W A)^-1 A^T W b for the best-fit (x, y)
       - Smoothed by a 2D constant-velocity Kalman filter

    2. PSO-LSTM 3D Positioning [primary neural method]
       - LSTM processes 8 consecutive 20-dim feature snapshots
       - Captures temporal RSSI dynamics (shadowing, NLOS, multipath)
       - Outputs (x, y, z) in meters with confidence label

    3. Gateway RSSI Correction [preprocessing]
       - Fixed gateway monitors beacon RSSI to detect transmit power drift
       - Correction offset ΔA_l applied before any distance estimation
"""

import json
import time
import threading
import numpy as np
from collections import defaultdict, deque
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import paho.mqtt.client as mqtt

from config import (
    MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD,
    STATION_IDS, MQTT_TOPIC_PATTERN,
    STATION_POSITIONS, ROOM_W, ROOM_H, ROOM_Z,
    SEQ_LEN, TARGET_TAG
)
from tag_processing import DataProcessor
from rssi_predictor import PositionPredictor
from gateway_correction import GatewayCorrection

# ── Globals ───────────────────────────────────────────────────────────────────
processor  = DataProcessor()
predictor  = PositionPredictor()
corrector  = GatewayCorrection()
results    = {}          # tag_id → latest LSTM prediction dict
lock       = threading.Lock()
STATION_ORDER = list(STATION_POSITIONS.keys())


# ── Trilateration Config ──────────────────────────────────────────────────────
TX_POWER  = -65    # reference RSSI at 1 meter (dBm)
PATH_LOSS = 2.0    # path loss exponent (free space = 2.0; indoor ≈ 2–4)

# 2D station coordinates used for trilateration (x, y in meters)
STATION_COORDS = {
    "STATION1": (ROOM_W / 2, ROOM_H),
    "STATION2": (ROOM_W,     ROOM_H / 2),
    "STATION3": (ROOM_W / 2, 0),
    "STATION4": (0,          ROOM_H / 2),
}

# ── Per-tag RSSI storage for trilateration ────────────────────────────────────
# latest_rssi[tag_id][station_id] = most recent (raw) RSSI
latest_rssi = defaultdict(dict)
rssi_lock   = threading.Lock()

# ── 1D Kalman Filter — RSSI Smoothing ────────────────────────────────────────
# Applied per (tag, station) pair before RSSI → distance conversion.
# Smooths out momentary BLE packet noise.
kalman_state = defaultdict(lambda: defaultdict(lambda: None))
kalman_cov   = defaultdict(lambda: defaultdict(lambda: 1.0))
PROCESS_NOISE = 0.1
MEAS_NOISE    = 10.0


def kalman_filter_rssi(tag_id: str, station: str, measurement: float) -> float:
    """1D Kalman filter for RSSI smoothing per (tag, station) channel."""
    if kalman_state[tag_id][station] is None:
        kalman_state[tag_id][station] = measurement

    pred_state = kalman_state[tag_id][station]
    pred_cov   = kalman_cov[tag_id][station] + PROCESS_NOISE
    K          = pred_cov / (pred_cov + MEAS_NOISE)

    kalman_state[tag_id][station] = pred_state + K * (measurement - pred_state)
    kalman_cov[tag_id][station]   = (1 - K) * pred_cov

    return kalman_state[tag_id][station]


# ── RSSI → Distance ───────────────────────────────────────────────────────────
def rssi_to_distance(rssi: float) -> float:
    """
    Log-distance path loss model:
        d = 10 ^ ((TX_POWER - RSSI) / (10 * n))
    where n is the path loss exponent.
    """
    return 10 ** ((TX_POWER - rssi) / (10 * PATH_LOSS))


# ── Weighted Least Squares Trilateration ─────────────────────────────────────
def weighted_least_squares(distances: dict):
    """
    Estimate (x, y) from noisy distance measurements using Weighted Least Squares.

    Mathematical Framework:
        Given m anchor nodes at known positions (x_i, y_i) with measured
        distances d_i, the exact system of circle equations is non-linear:

            (x - x_i)^2 + (y - y_i)^2 = d_i^2

        To linearize, we select the m-th anchor as the reference and subtract
        its equation from all others, canceling the squared unknowns x^2, y^2:

            2(x_i - x_m)·x + 2(y_i - y_m)·y = x_i^2 - x_m^2
                                               + y_i^2 - y_m^2
                                               + d_m^2 - d_i^2

        This yields a linear system A·X = b where X = [x, y]^T.

        For the Weighted variant (WLS):
            X = (A^T W A)^-1 A^T W b

        The weight matrix W = diag(w_1, ..., w_{m-1}) assigns higher confidence
        to closer anchors because RSSI is more reliable at shorter distances:
            w_i = 1 / (d_i^2 + ε)

        Advantages over Nonlinear Least Squares:
            - No iterative solver needed (closed-form solution)
            - No sensitivity to initial guess
            - Unique solution even with contradictory measurements
            - Closer anchors have more influence (lower RSSI error)

    Args:
        distances: dict mapping station_id → estimated distance (meters)

    Returns:
        (x, y) in meters, clamped to room bounds — or None if insufficient data.
    """
    valid = {s: d for s, d in distances.items() if s in STATION_COORDS}
    if len(valid) < 3:
        return None

    stations = list(valid.keys())
    # Use last station as the reference anchor (m-th)
    ref    = stations[-1]
    xm, ym = STATION_COORDS[ref]
    dm     = valid[ref]

    rows_A   = []
    rows_b   = []
    weights  = []

    for sid in stations[:-1]:
        xi, yi = STATION_COORDS[sid]
        di     = valid[sid]

        # Linearized equation coefficients
        rows_A.append([2.0 * (xi - xm), 2.0 * (yi - ym)])
        rows_b.append(
            xi**2 - xm**2 + yi**2 - ym**2 + dm**2 - di**2
        )
        # Weight: closer anchors → smaller distance → higher weight
        weights.append(1.0 / (di**2 + 1e-6))

    A = np.array(rows_A, dtype=np.float64)
    b = np.array(rows_b, dtype=np.float64)
    W = np.diag(weights)

    # WLS normal equations: X = (A^T W A)^-1 A^T W b
    AtWA = A.T @ W @ A
    AtWb = A.T @ W @ b

    try:
        X = np.linalg.solve(AtWA, AtWb)
    except np.linalg.LinAlgError:
        # Fallback to unweighted pseudoinverse if matrix is singular
        X, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

    x = float(np.clip(X[0], 0.0, ROOM_W))
    y = float(np.clip(X[1], 0.0, ROOM_H))
    return x, y


# ── 2D Kalman Filter — Position Smoothing ────────────────────────────────────
# Constant-velocity model applied to WLS trilateration output.
# Smooths trajectory for display and reduces WLS noise jitter.
dt = 0.5

kalman_2d_state = {}
kalman_2d_P     = {}

F = np.array([
    [1, 0, dt, 0],
    [0, 1, 0, dt],
    [0, 0, 1,  0],
    [0, 0, 0,  1]
], dtype=np.float64)

H = np.array([[1, 0, 0, 0],
              [0, 1, 0, 0]], dtype=np.float64)

Q = np.eye(4, dtype=np.float64) * 0.02   # process noise covariance
R = np.eye(2, dtype=np.float64) * 0.3    # measurement noise covariance


def kalman_2d(tag_id: str, measurement: tuple) -> tuple:
    """
    2D constant-velocity Kalman filter for position smoothing.

    State vector: [x, y, vx, vy]
    """
    if tag_id not in kalman_2d_state:
        kalman_2d_state[tag_id] = np.array(
            [measurement[0], measurement[1], 0.0, 0.0]
        )
        kalman_2d_P[tag_id] = np.eye(4, dtype=np.float64)

    state = kalman_2d_state[tag_id]
    P     = kalman_2d_P[tag_id]

    # Predict
    state = F @ state
    P     = F @ P @ F.T + Q

    # Update
    z = np.array(measurement, dtype=np.float64)
    y = z - H @ state
    S = H @ P @ H.T + R
    K = P @ H.T @ np.linalg.inv(S)

    state = state + K @ y
    P     = (np.eye(4, dtype=np.float64) - K @ H) @ P

    kalman_2d_state[tag_id] = state
    kalman_2d_P[tag_id]     = P

    return float(state[0]), float(state[1])


# ── Tag Position Storage ──────────────────────────────────────────────────────
# tag_positions[tag_id] = {
#     "wls"   : (x, y)      ← WLS trilateration result
#     "lstm"  : (x, y, z)   ← PSO-LSTM prediction
#     "kalman": (x, y)      ← Kalman-smoothed WLS
# }
tag_positions = {}
pos_lock      = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PROCESS PAYLOAD
# ─────────────────────────────────────────────────────────────────────────────

def process_payload(payload: dict):
    """
    Full processing pipeline for one incoming MQTT payload:

    1. Parse station and tag data
    2. Apply gateway RSSI correction (transmit power normalization)
    3. Update DataProcessor RSSI window with corrected value
    4. Build temporal feature sequence → run PSO-LSTM for 3D position
    5. Apply 1D Kalman filter to raw RSSI → convert to distance
    6. Compute WLS trilateration → apply 2D Kalman smoothing
    """
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

        # Only process the target tag
        if tag_id != TARGET_TAG.lower():
            continue

        if not tag_id or rssi is None:
            continue
        if not (-120 <= rssi <= 0):
            continue

        # ── Step 1: Store raw RSSI for display table ──────────────────────
        with rssi_lock:
            latest_rssi[tag_id][station_id] = float(rssi)

        # ── Step 2: Compute gateway correction offset ─────────────────────
        # The corrector maintains a rolling history of gateway RSSI per station.
        # Since gateway is at a fixed position, RSSI changes there reflect
        # beacon transmit power drift, not physical movement.
        correction_offset = 0.0
        if corrector.has_calibration(station_id):
            correction_offset = corrector.get_correction_offset(
                station_id,
                corrector.gateway_history[station_id][-1]
                if corrector.gateway_history[station_id] else rssi
            )

        # ── Step 3: Update RSSI window with corrected value ───────────────
        processor.update_station_rssi(
            station_id, tag_id, rssi,
            correction_offset=correction_offset
        )

        # ── Step 4: Build temporal sequence → PSO-LSTM prediction ─────────
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

    # ── Step 5 & 6: WLS Trilateration for TARGET_TAG ──────────────────────
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
        # 1D Kalman filter smooths per-station RSSI before distance conversion
        filtered       = kalman_filter_rssi(target, sid, r)
        distances[sid] = rssi_to_distance(filtered)

    # Weighted Least Squares trilateration
    wls_pos = weighted_least_squares(distances)
    if wls_pos is None:
        return

    # 2D Kalman filter smooths the WLS position for trajectory display
    smooth_pos = kalman_2d(target, wls_pos)

    with pos_lock:
        if target not in tag_positions:
            tag_positions[target] = {}
        tag_positions[target]["wls"]    = wls_pos
        tag_positions[target]["kalman"] = smooth_pos


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — MQTT CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

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


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MQTT THREAD
# ─────────────────────────────────────────────────────────────────────────────

def start_mqtt():
    """Runs in background thread — handles all MQTT communication."""
    time.sleep(2)

    while True:
        try:
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

            client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)
            client.loop_forever()

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[MQTT] Error: {e} — retrying in 5s...")
            time.sleep(5)


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — REAL-TIME PLOT
# ─────────────────────────────────────────────────────────────────────────────

TAG_COLORS = [
    "red", "blue", "green", "orange",
    "purple", "cyan", "magenta", "brown"
]
tag_color_map = {}
color_idx     = [0]


def get_tag_color(tag_id: str) -> str:
    if tag_id not in tag_color_map:
        tag_color_map[tag_id] = TAG_COLORS[
            color_idx[0] % len(TAG_COLORS)
        ]
        color_idx[0] += 1
    return tag_color_map[tag_id]


def run_plot():
    """
    Real-time matplotlib display:
        Left  — room map with WLS (●) and PSO-LSTM (★) tag positions
        Right — live RSSI + coordinate table
    """
    plt.ion()
    fig = plt.figure(figsize=(14, 7))
    fig.suptitle(
        "BLE Indoor Positioning — PSO-LSTM + Weighted Least Squares",
        fontsize=13
    )
    gs       = gridspec.GridSpec(1, 2, width_ratios=[2, 1])
    ax_room  = fig.add_subplot(gs[0])
    ax_table = fig.add_subplot(gs[1])

    print("[PLOT] Real-time display started. Press Ctrl+C to stop.")

    while True:
        try:
            ax_room.clear()
            ax_table.clear()

            # ── Draw Room ──────────────────────────────────────────────────
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
                f"▲ = Station   ● = WLS Trilat (Kalman)   ★ = PSO-LSTM"
            )
            ax_room.grid(True, alpha=0.3)

            # ── Draw Stations ──────────────────────────────────────────────
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

            # ── Draw Tag Positions ─────────────────────────────────────────
            with pos_lock:
                positions_snapshot = {
                    k: v.copy() for k, v in tag_positions.items()
                }

            for tag_id, pos in positions_snapshot.items():
                color    = get_tag_color(tag_id)
                short_id = tag_id[-5:]

                # WLS + Kalman-smoothed position (circle)
                if "kalman" in pos:
                    kx, ky = pos["kalman"]
                    ax_room.scatter(
                        kx, ky,
                        s=120, c=color, marker="o", zorder=5,
                        label=f"{short_id} WLS"
                    )
                    ax_room.annotate(
                        f"{short_id}\n({kx:.1f},{ky:.1f})",
                        (kx, ky),
                        textcoords="offset points",
                        xytext=(8, 8), fontsize=7, color=color
                    )

                # PSO-LSTM position (star)
                if "lstm" in pos:
                    lx, ly, lz = pos["lstm"]
                    ax_room.scatter(
                        lx, ly,
                        s=200, c=color, marker="*", zorder=6,
                        label=f"{short_id} LSTM"
                    )
                    ax_room.annotate(
                        f"LSTM\n({lx:.1f},{ly:.1f},{lz:.1f}m)",
                        (lx, ly),
                        textcoords="offset points",
                        xytext=(8, -16), fontsize=7, color=color,
                        style="italic"
                    )

                # Dashed line between WLS and LSTM estimates
                if "kalman" in pos and "lstm" in pos:
                    kx, ky    = pos["kalman"]
                    lx, ly, _ = pos["lstm"]
                    ax_room.plot(
                        [kx, lx], [ky, ly],
                        "--", color=color, alpha=0.4, linewidth=1
                    )

            if positions_snapshot:
                ax_room.legend(
                    loc="upper right", fontsize=7, framealpha=0.8
                )

            # ── RSSI Table ─────────────────────────────────────────────────
            with rssi_lock:
                rssi_snapshot = {
                    k: v.copy() for k, v in latest_rssi.items()
                }
            with lock:
                lstm_snapshot = results.copy()

            table_data  = []
            col_headers = (
                ["Tag"] +
                [s.replace("STATION", "S") for s in STATION_ORDER] +
                ["X", "Y", "Z", "Conf"]
            )

            for tag_id, station_rssi in rssi_snapshot.items():
                short_id  = tag_id[-11:]
                rssi_vals = [
                    str(int(station_rssi.get(sid, 0)))
                    if sid in station_rssi else "—"
                    for sid in STATION_ORDER
                ]
                lstm_res = lstm_snapshot.get(tag_id, {})
                x_str    = f"{lstm_res.get('x', 0):.1f}"
                y_str    = f"{lstm_res.get('y', 0):.1f}"
                z_str    = f"{lstm_res.get('z', 0):.1f}"
                conf_str = lstm_res.get("confidence", "—")[:4]

                table_data.append(
                    [short_id] + rssi_vals + [x_str, y_str, z_str, conf_str]
                )

            ax_table.axis("off")

            if table_data:
                table = ax_table.table(
                    cellText  = table_data,
                    colLabels = col_headers,
                    loc       = "center",
                    cellLoc   = "center"
                )
                table.auto_set_font_size(False)
                table.set_fontsize(8)
                table.scale(1.1, 1.6)

                for col in range(len(col_headers)):
                    table[0, col].set_facecolor("#2c3e50")
                    table[0, col].set_text_props(
                        color="white", fontweight="bold"
                    )
                for row in range(1, len(table_data) + 1):
                    bg = "#ecf0f1" if row % 2 == 0 else "white"
                    for col in range(len(col_headers)):
                        table[row, col].set_facecolor(bg)
            else:
                ax_table.text(
                    0.5, 0.5,
                    "Waiting for\nMQTT data...",
                    ha="center", va="center",
                    fontsize=12, color="gray",
                    transform=ax_table.transAxes
                )

            ax_table.set_title(
                "Live RSSI + PSO-LSTM Position", fontsize=10
            )

            plt.tight_layout()
            plt.pause(0.5)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[PLOT] Error: {e}")
            time.sleep(0.5)

    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 5 — SUMMARY THREAD
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    print("\n" + "=" * 80)
    print("  POSITION SUMMARY  —  x, y, z in meters")
    print("=" * 80)

    with lock:
        if not results:
            print(f"  No PSO-LSTM predictions yet — "
                  f"need {SEQ_LEN} sequential readings per tag")
        else:
            print(
                f"  {'TAG ID':<22}  {'X':>6}  {'Y':>6}  {'Z':>6}  "
                f"{'ZONE':<16}  CONFIDENCE"
            )
            print("  " + "-" * 72)
            for tag_id, res in results.items():
                print(
                    f"  {tag_id:<22}  "
                    f"{res.get('x', 0):>6.2f}  "
                    f"{res.get('y', 0):>6.2f}  "
                    f"{res.get('z', 0):>6.2f}  "
                    f"{res.get('zone', '?'):<16}  "
                    f"{res.get('confidence', '?')}"
                )

    with pos_lock:
        if tag_positions:
            print(f"\n  WEIGHTED LEAST SQUARES TRILATERATION:")
            print(
                f"  {'TAG ID':<22}  {'WLS X':>8}  {'WLS Y':>8}  "
                f"{'KAL X':>8}  {'KAL Y':>8}"
            )
            print("  " + "-" * 72)
            for tag_id, pos in tag_positions.items():
                wx = f"{pos['wls'][0]:.2f}"    if "wls"    in pos else "—"
                wy = f"{pos['wls'][1]:.2f}"    if "wls"    in pos else "—"
                kx = f"{pos['kalman'][0]:.2f}" if "kalman" in pos else "—"
                ky = f"{pos['kalman'][1]:.2f}" if "kalman" in pos else "—"
                print(
                    f"  {tag_id:<22}  {wx:>8}  {wy:>8}  "
                    f"{kx:>8}  {ky:>8}"
                )

    # Show gateway correction status
    print(f"\n  GATEWAY CORRECTION: {corrector.get_status()}")
    print("=" * 80 + "\n")


def print_summary_loop():
    while True:
        time.sleep(30)
        print_summary()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 64)
    print("  BLE Indoor Positioning — PSO-LSTM + WLS Trilateration")
    print(f"  Broker     : {MQTT_BROKER}:{MQTT_PORT}")
    print(f"  Room       : {ROOM_W}m × {ROOM_H}m × {ROOM_Z}m")
    print(f"  SEQ_LEN    : {SEQ_LEN} time steps (temporal LSTM window)")
    print(f"  Stations   : {STATION_ORDER}")
    print(f"  Algorithm  : PSO-LSTM (3D) + WLS Trilateration (2D)")
    print(f"  Correction : Gateway RSSI power correction enabled")
    print("=" * 64 + "\n")

    # Background MQTT thread
    mqtt_thread = threading.Thread(target=start_mqtt, daemon=True)
    mqtt_thread.start()

    # Background summary thread
    threading.Thread(target=print_summary_loop, daemon=True).start()

    # Main thread: real-time plot (must run on main thread on macOS)
    try:
        run_plot()
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down...")


"""
Processing Pipeline Diagram
============================

MQTT Message (stationId, tags: [{tagId, rssi}])
    │
    ├─► Store raw RSSI in latest_rssi[tag][station]
    │
    ├─► GatewayCorrection.get_correction_offset(station)
    │       ΔA_l = R_gateway_current - R_gateway_historical_avg
    │
    ├─► DataProcessor.update_station_rssi(..., correction_offset=ΔA_l)
    │       corrected = raw_rssi - ΔA_l   (power-normalized)
    │       stored in rssi_window[tag][station] (last 4 readings)
    │
    ├─► DataProcessor.push_to_buffer(tag)
    │       builds 20-dim feature vector from 4-reading stats
    │       appends to temporal buffer (SEQ_LEN=8 vectors)
    │       when full → returns (8, 20) sequence
    │
    ├─► PositionPredictor.predict(sequence (8,20))
    │       PSO-LSTM → normalized (x,y,z) → denorm → meters
    │       → tag_positions[tag]["lstm"] = (x, y, z)
    │
    └─► Trilateration branch (using raw RSSI from latest_rssi):
            KalmanFilter(raw_rssi) → smooth_rssi
            rssi_to_distance(smooth_rssi) → d_i per station
            WeightedLeastSquares({station: d_i})
                A = linearized anchor equations
                W = diag(1/d_i^2)  ← closer = higher weight
                X = (A^T W A)^-1 A^T W b  → (x, y)
            Kalman2D(x, y) → smooth trajectory
            → tag_positions[tag]["wls"]    = (x, y)
            → tag_positions[tag]["kalman"] = (x_smooth, y_smooth)

Plot:
    ● = Kalman-smoothed WLS trilateration position
    ★ = PSO-LSTM predicted position
    dashed = vector between the two estimates
"""
