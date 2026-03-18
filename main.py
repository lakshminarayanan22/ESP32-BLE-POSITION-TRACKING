"""
BLE Indoor Positioning — Real-Time Inference (Three-Model Comparison)
======================================================================

All three models run simultaneously on every RSSI update and their
positions are plotted on the same room map for side-by-side comparison.

Models:
    ★  PSO-LSTM     — temporal LSTM (SEQ_LEN=8) with PSO-optimised weights.
                      Requires 8 consecutive readings before first prediction.
    ■  SVM (RBF)    — GridSearchCV-optimised SVR, single-snapshot input.
                      Predicts immediately on first RSSI reading.
    ◆  Random Forest— GridSearchCV-optimised RF, single-snapshot input.
                      Predicts immediately on first RSSI reading.

Preprocessing:
    Gateway Correction  : ΔA_l removes beacon transmit power drift.
    1D Kalman (RSSI)    : smooths per-station readings before SVM/RF input.

Plot layout:
    Left  (2/3) — room map 27 m × 10 m; stations ▲, three model markers.
    Right (1/3) — position comparison table and top-station RSSI summary.
"""

import json
import time
import threading
import numpy as np
from collections import defaultdict
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import paho.mqtt.client as mqtt

from config import (
    MQTT_BROKER, MQTT_PORT, MQTT_USERNAME, MQTT_PASSWORD,
    STATION_IDS, MQTT_TOPIC_PATTERN,
    STATION_POSITIONS, ROOM_W, ROOM_H, ROOM_Z,
    SEQ_LEN, TARGET_TAG,
)
from tag_processing  import DataProcessor
from rssi_predictor  import PositionPredictor
from svm_model       import SVMPositionPredictor
from rf_model        import RFPositionPredictor
from gateway_correction import GatewayCorrection

# ── Globals ───────────────────────────────────────────────────────────────────
processor     = DataProcessor()
lstm_pred     = PositionPredictor()
svm_pred      = SVMPositionPredictor()
rf_pred       = RFPositionPredictor()
corrector     = GatewayCorrection()
results       = {}       # tag_id → dict with lstm/svm/rf predictions
lock          = threading.Lock()
STATION_ORDER = list(STATION_POSITIONS.keys())

# ── Per-tag RSSI storage (for display table and feature input) ─────────────
latest_rssi = defaultdict(dict)
rssi_lock   = threading.Lock()

# ── Per-tag position storage ──────────────────────────────────────────────────
# tag_positions[tag_id] = {
#     "lstm": (x, y, z)
#     "svm" : (x, y, z)
#     "rf"  : (x, y, z)
# }
tag_positions = {}
pos_lock      = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — PROCESS PAYLOAD
# ─────────────────────────────────────────────────────────────────────────────

def process_payload(payload: dict):
    """
    Full real-time inference pipeline per MQTT payload.

    Steps:
        1. Parse station + tag from MQTT JSON.
        2. Store raw RSSI for display table.
        3. Compute gateway correction offset (ΔA_l).
        4. Update DataProcessor RSSI window with corrected value.
        5. Build single-snapshot 70-dim feature vector → SVM + RF predict.
        6. Accumulate temporal buffer (SEQ_LEN=8) → PSO-LSTM predict when full.
        7. Write all predictions to tag_positions.
    """
    station_id = payload.get("stationId", "").upper()
    tags       = payload.get("tags", [])

    if not tags:
        return
    if station_id not in STATION_POSITIONS:
        return

    for tag in tags:
        tag_id = tag.get("tagId", "").lower().strip()
        rssi   = tag.get("rssi", None)

        if tag_id != TARGET_TAG.lower():
            continue
        if not tag_id or rssi is None:
            continue
        if not (-120 <= rssi <= 0):
            continue

        # ── Step 2: store raw for display ─────────────────────────────────
        with rssi_lock:
            latest_rssi[tag_id][station_id] = float(rssi)

        # ── Step 3: gateway correction offset ─────────────────────────────
        correction_offset = 0.0
        if corrector.has_calibration(station_id):
            history = corrector.gateway_history[station_id]
            if history:
                correction_offset = corrector.get_correction_offset(
                    station_id, history[-1]
                )

        # ── Step 4: update RSSI window (corrected) ────────────────────────
        processor.update_station_rssi(
            station_id, tag_id, rssi,
            correction_offset=correction_offset
        )

        # ── Step 5: single-snapshot feature vector → SVM + RF ─────────────
        feat_vec = processor.build_feature_vector(tag_id)   # list[70 floats]

        svm_result = svm_pred.predict(np.array(feat_vec, dtype=np.float64))
        rf_result  = rf_pred.predict(np.array(feat_vec, dtype=np.float64))

        with pos_lock:
            if tag_id not in tag_positions:
                tag_positions[tag_id] = {}
            tag_positions[tag_id]["svm"] = (
                svm_result["x"], svm_result["y"], svm_result["z"]
            )
            tag_positions[tag_id]["rf"] = (
                rf_result["x"], rf_result["y"], rf_result["z"]
            )

        # ── Step 6: temporal buffer → PSO-LSTM ────────────────────────────
        sequence = processor.push_to_buffer(tag_id)
        if sequence is not None:
            lstm_result = lstm_pred.predict(sequence)
            with pos_lock:
                tag_positions[tag_id]["lstm"] = (
                    lstm_result["x"], lstm_result["y"], lstm_result["z"]
                )

        # ── Step 7: store combined result for summary table ────────────────
        with lock:
            results[tag_id] = {
                "station":  station_id,
                "rssi":     rssi,
                "coverage": processor.get_station_coverage(tag_id),
                "lstm_x":   tag_positions[tag_id].get("lstm", (0, 0, 0))[0],
                "lstm_y":   tag_positions[tag_id].get("lstm", (0, 0, 0))[1],
                "lstm_z":   tag_positions[tag_id].get("lstm", (0, 0, 0))[2],
                "svm_x":    svm_result["x"],
                "svm_y":    svm_result["y"],
                "svm_z":    svm_result["z"],
                "rf_x":     rf_result["x"],
                "rf_y":     rf_result["y"],
                "rf_z":     rf_result["z"],
            }


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — MQTT CALLBACKS
# ─────────────────────────────────────────────────────────────────────────────

def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"[MQTT] Connected → {MQTT_BROKER}:{MQTT_PORT}")
        client.subscribe(MQTT_TOPIC_PATTERN)
        print(f"[MQTT] Subscribed → {MQTT_TOPIC_PATTERN}")
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
    time.sleep(2)
    while True:
        try:
            import socket
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.settimeout(10)
            sock.connect((MQTT_BROKER, MQTT_PORT))
            sock.close()
            print(f"[MQTT] Socket pre-check OK → {MQTT_BROKER}:{MQTT_PORT}")

            client = mqtt.Client(
                mqtt.CallbackAPIVersion.VERSION2,
                client_id="ble_rtls_3models"
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

# Marker styles per model — distinguishable on mono and colour displays
MODEL_STYLES = {
    "lstm": {"marker": "*", "size": 280, "color": "#e74c3c", "label": "PSO-LSTM"},
    "svm":  {"marker": "s", "size": 140, "color": "#2980b9", "label": "SVM (RBF)"},
    "rf":   {"marker": "D", "size": 120, "color": "#27ae60", "label": "Rand Forest"},
}

TAG_COLORS = [
    "#e74c3c", "#2980b9", "#27ae60", "#f39c12",
    "#8e44ad", "#16a085", "#d35400", "#2c3e50"
]
tag_color_map = {}
color_idx     = [0]


def get_tag_color(tag_id: str) -> str:
    if tag_id not in tag_color_map:
        tag_color_map[tag_id] = TAG_COLORS[color_idx[0] % len(TAG_COLORS)]
        color_idx[0] += 1
    return tag_color_map[tag_id]


def run_plot():
    """
    Real-time matplotlib display.

    Layout:
        Left (2/3 width)  — 27 m × 10 m room map.
                            ▲ = 14 anchor stations
                            ★ = PSO-LSTM prediction
                            ■ = SVM (RBF) prediction
                            ◆ = Random Forest prediction

        Right (1/3 width) — Two stacked panels:
                            Top    : RSSI summary per tag (top 6 stations)
                            Bottom : Position table (LSTM / SVM / RF)
    """
    plt.ion()
    fig  = plt.figure(figsize=(20, 8))
    fig.suptitle(
        "BLE Indoor Positioning — PSO-LSTM  ★   SVM  ■   Random Forest  ◆",
        fontsize=13, fontweight="bold"
    )

    outer = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1], figure=fig,
                              left=0.04, right=0.98, top=0.92, bottom=0.06,
                              wspace=0.06)
    ax_room = fig.add_subplot(outer[0])

    right   = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=outer[1], hspace=0.45
    )
    ax_rssi = fig.add_subplot(right[0])
    ax_pos  = fig.add_subplot(right[1])

    print("[PLOT] Real-time display started. Press Ctrl+C to stop.")

    while True:
        try:
            ax_room.clear()
            ax_rssi.clear()
            ax_pos.clear()

            # ── Room boundary ─────────────────────────────────────────────
            ax_room.plot(
                [0, ROOM_W, ROOM_W, 0, 0],
                [0, 0, ROOM_H, ROOM_H, 0],
                "k-", linewidth=2
            )
            ax_room.set_xlim(-1, ROOM_W + 1)
            ax_room.set_ylim(-1, ROOM_H + 1)
            ax_room.set_aspect("equal")
            ax_room.set_xlabel("X (m)", fontsize=9)
            ax_room.set_ylabel("Y (m)", fontsize=9)
            ax_room.set_title(
                f"Room {ROOM_W:.0f} m × {ROOM_H:.0f} m × {ROOM_Z:.0f} m",
                fontsize=10
            )
            ax_room.grid(True, alpha=0.25)

            # ── Anchor stations ────────────────────────────────────────────
            for sid, (sx, sy, _) in STATION_POSITIONS.items():
                ax_room.plot(sx, sy, "k^", markersize=9, zorder=4)
                ax_room.annotate(
                    sid.replace("STATION", "S"),
                    (sx, sy),
                    textcoords="offset points",
                    xytext=(4, 5),
                    fontsize=6,
                    fontweight="bold",
                    color="#2c3e50"
                )

            # ── Model predictions per tag ─────────────────────────────────
            with pos_lock:
                pos_snap = {k: v.copy() for k, v in tag_positions.items()}

            for tag_id, pos in pos_snap.items():
                short = tag_id[-5:]

                for model_key, style in MODEL_STYLES.items():
                    if model_key not in pos:
                        continue
                    mx, my, mz = pos[model_key]
                    ax_room.scatter(
                        mx, my,
                        s      = style["size"],
                        color  = style["color"],
                        marker = style["marker"],
                        zorder = 6,
                        alpha  = 0.85,
                        label  = f"{short} {style['label']}"
                    )
                    ax_room.annotate(
                        f"{style['label'][:4]}\n"
                        f"({mx:.1f},{my:.1f},{mz:.1f})",
                        (mx, my),
                        textcoords = "offset points",
                        xytext     = (7, 7),
                        fontsize   = 6.5,
                        color      = style["color"],
                    )

                # Connect the three model estimates with thin lines
                pts = [(pos[k][0], pos[k][1])
                       for k in ("lstm", "svm", "rf") if k in pos]
                if len(pts) >= 2:
                    xs = [p[0] for p in pts] + [pts[0][0]]
                    ys = [p[1] for p in pts] + [pts[0][1]]
                    ax_room.plot(xs, ys, "--", color="#95a5a6",
                                 alpha=0.5, linewidth=0.8)

            # Legend — one entry per model type (deduplicate)
            handles, labels_seen = [], set()
            for style in MODEL_STYLES.values():
                lbl = style["label"]
                if lbl not in labels_seen:
                    labels_seen.add(lbl)
                    handles.append(
                        plt.scatter([], [],
                                    s=style["size"] * 0.6,
                                    color=style["color"],
                                    marker=style["marker"],
                                    label=lbl)
                    )
            if handles:
                ax_room.legend(handles=handles, loc="upper right",
                               fontsize=8, framealpha=0.85)

            # ── RSSI Summary (top 6 stations by signal strength) ──────────
            with rssi_lock:
                rssi_snap = {k: v.copy() for k, v in latest_rssi.items()}

            ax_rssi.axis("off")
            if rssi_snap:
                rssi_rows = []
                for tag_id, stn_rssi in rssi_snap.items():
                    short = tag_id[-11:]
                    # Sort stations by RSSI descending, take top 6
                    top6 = sorted(stn_rssi.items(),
                                  key=lambda kv: kv[1], reverse=True)[:6]
                    best_str = "  ".join(
                        f"{s.replace('STATION','S')}:{int(r)}" for s, r in top6
                    )
                    rssi_rows.append([short, best_str])

                rssi_table = ax_rssi.table(
                    cellText  = rssi_rows,
                    colLabels = ["Tag", "Top-6 Station RSSI (dBm)"],
                    loc       = "center", cellLoc="left"
                )
                rssi_table.auto_set_font_size(False)
                rssi_table.set_fontsize(7)
                rssi_table.scale(1.0, 1.5)
                for c in range(2):
                    rssi_table[0, c].set_facecolor("#2c3e50")
                    rssi_table[0, c].set_text_props(
                        color="white", fontweight="bold"
                    )
                ax_rssi.set_title("Live RSSI (top 6 stations)", fontsize=9)
            else:
                ax_rssi.text(0.5, 0.5, "Waiting for MQTT data...",
                             ha="center", va="center", fontsize=10,
                             color="gray", transform=ax_rssi.transAxes)

            # ── Position Comparison Table ──────────────────────────────────
            ax_pos.axis("off")
            with lock:
                res_snap = results.copy()

            if res_snap:
                pos_rows = []
                col_hdrs = ["Tag",
                            "LSTM x", "LSTM y", "LSTM z",
                            "SVM x",  "SVM y",  "SVM z",
                            "RF x",   "RF y",   "RF z"]
                for tag_id, r in res_snap.items():
                    short = tag_id[-11:]
                    pos_rows.append([
                        short,
                        f"{r.get('lstm_x',0):.1f}",
                        f"{r.get('lstm_y',0):.1f}",
                        f"{r.get('lstm_z',0):.1f}",
                        f"{r.get('svm_x', 0):.1f}",
                        f"{r.get('svm_y', 0):.1f}",
                        f"{r.get('svm_z', 0):.1f}",
                        f"{r.get('rf_x',  0):.1f}",
                        f"{r.get('rf_y',  0):.1f}",
                        f"{r.get('rf_z',  0):.1f}",
                    ])

                pos_table = ax_pos.table(
                    cellText  = pos_rows,
                    colLabels = col_hdrs,
                    loc       = "center",
                    cellLoc   = "center"
                )
                pos_table.auto_set_font_size(False)
                pos_table.set_fontsize(7)
                pos_table.scale(1.0, 1.6)

                # Header row colours matching model styles
                header_colors = (
                    ["#2c3e50"] +
                    ["#c0392b"] * 3 +   # LSTM
                    ["#1a5276"] * 3 +   # SVM
                    ["#1d6a39"] * 3     # RF
                )
                for c, hc in enumerate(header_colors):
                    pos_table[0, c].set_facecolor(hc)
                    pos_table[0, c].set_text_props(
                        color="white", fontweight="bold"
                    )
                for row in range(1, len(pos_rows) + 1):
                    bg = "#ecf0f1" if row % 2 == 0 else "white"
                    for c in range(len(col_hdrs)):
                        pos_table[row, c].set_facecolor(bg)

                ax_pos.set_title(
                    "★ PSO-LSTM   ■ SVM (RBF)   ◆ Random Forest",
                    fontsize=9
                )
            else:
                ax_pos.text(0.5, 0.5, "No predictions yet\n"
                            f"(LSTM needs {SEQ_LEN} readings)",
                            ha="center", va="center", fontsize=9,
                            color="gray", transform=ax_pos.transAxes)

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
    print("  POSITION SUMMARY  —  coordinates in meters")
    print("=" * 80)

    with lock:
        if not results:
            print(f"  No predictions yet (PSO-LSTM needs {SEQ_LEN} sequential readings)")
        else:
            hdr = (f"  {'TAG':<22}  "
                   f"{'LSTM_X':>7}  {'LSTM_Y':>7}  {'LSTM_Z':>7}  "
                   f"{'SVM_X':>7}  {'SVM_Y':>7}  {'SVM_Z':>7}  "
                   f"{'RF_X':>6}  {'RF_Y':>6}  {'RF_Z':>6}")
            print(hdr)
            print("  " + "-" * 78)
            for tag_id, r in results.items():
                print(
                    f"  {tag_id:<22}  "
                    f"{r.get('lstm_x',0):>7.2f}  "
                    f"{r.get('lstm_y',0):>7.2f}  "
                    f"{r.get('lstm_z',0):>7.2f}  "
                    f"{r.get('svm_x', 0):>7.2f}  "
                    f"{r.get('svm_y', 0):>7.2f}  "
                    f"{r.get('svm_z', 0):>7.2f}  "
                    f"{r.get('rf_x',  0):>6.2f}  "
                    f"{r.get('rf_y',  0):>6.2f}  "
                    f"{r.get('rf_z',  0):>6.2f}"
                )

    print(f"\n  Gateway correction status: {corrector.get_status()}")
    print("=" * 80 + "\n")


def print_summary_loop():
    while True:
        time.sleep(30)
        print_summary()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("  BLE Indoor Positioning — Three-Model Real-Time Comparison")
    print(f"  Broker    : {MQTT_BROKER}:{MQTT_PORT}")
    print(f"  Room      : {ROOM_W} m × {ROOM_H} m × {ROOM_Z} m")
    print(f"  Stations  : {len(STATION_ORDER)}")
    print(f"  Models    : PSO-LSTM (SEQ={SEQ_LEN})  |  SVM-RBF  |  RandomForest")
    print(f"  Anchor    : Gateway correction enabled")
    print("=" * 68 + "\n")

    # Background MQTT thread
    threading.Thread(target=start_mqtt, daemon=True).start()

    # Background summary thread
    threading.Thread(target=print_summary_loop, daemon=True).start()

    # Main thread: real-time plot
    try:
        run_plot()
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down...")


"""
Data Flow Diagram
=================

MQTT → process_payload(payload)
           │
           ├─ latest_rssi[tag][station] = raw_rssi          (display table)
           │
           ├─ GatewayCorrection.get_correction_offset(station)
           │       ΔA_l = R_gw_current - mean(R_gw_history)
           │
           ├─ DataProcessor.update_station_rssi(station, tag, rssi, ΔA_l)
           │       corrected = raw - ΔA_l  →  rssi_window[tag][station]
           │
           ├─ feat = DataProcessor.build_feature_vector(tag)  (70-dim)
           │
           ├─ SVMPositionPredictor.predict(feat)  → (x, y, z)   [SVM ■]
           │
           ├─ RFPositionPredictor.predict(feat)   → (x, y, z)   [RF  ◆]
           │
           └─ DataProcessor.push_to_buffer(tag)  → (8, 70) when full
                   └─ PositionPredictor.predict(seq) → (x, y, z) [LSTM ★]
"""
