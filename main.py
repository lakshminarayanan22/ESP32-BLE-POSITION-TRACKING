"""
BLE Indoor Positioning — Real-Time Inference (Three-Model Comparison)
======================================================================

Timing Architecture
-------------------
Each BLE station publishes one MQTT payload every PREDICTION_INTERVAL (3 s).
Predictions are deliberately decoupled from ingestion:

    MQTT thread      — receives payloads continuously, updates RSSI windows
                       and latest_rssi store. No model calls here.

    Prediction thread — wakes exactly once every PREDICTION_INTERVAL seconds.
                        By then all visible stations have had a chance to
                        report. Builds one complete 70-dim feature vector
                        from the accumulated window and runs all three models.

This prevents the "empty feature" problem: if predictions triggered on every
arriving message, only 1-3 of 14 stations would have reported, giving
11-13 zero-filled station blocks and therefore garbage coordinates.

Models run in the prediction thread (once per 3 s cycle):
    ★  PSO-LSTM     — temporal LSTM (SEQ_LEN=8); one buffer entry per cycle.
                      Needs 8 cycles (24 s) before first prediction.
    ■  SVM (RBF)    — single-snapshot 70-dim input; predicts from cycle 1.
    ◆  Random Forest— single-snapshot 70-dim input; predicts from cycle 1.

Preprocessing:
    Gateway Correction: ΔA_l removes beacon transmit power drift per cycle.
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
    SEQ_LEN, TARGET_TAG, PREDICTION_INTERVAL,
)
from tag_processing     import DataProcessor
from rssi_predictor     import PositionPredictor
from svm_model          import SVMPositionPredictor
from rf_model           import RFPositionPredictor
from gateway_correction import GatewayCorrection

# ── Globals ───────────────────────────────────────────────────────────────────
processor     = DataProcessor()
lstm_pred     = PositionPredictor()
svm_pred      = SVMPositionPredictor()
rf_pred       = RFPositionPredictor()
corrector     = GatewayCorrection()
results       = {}       # tag_id → latest combined prediction dict
lock          = threading.Lock()
STATION_ORDER = list(STATION_POSITIONS.keys())

# ── Raw RSSI store — ingestion only, used for display table ───────────────────
latest_rssi = defaultdict(dict)
rssi_lock   = threading.Lock()

# ── Position store — written by prediction thread, read by plot ───────────────
# tag_positions[tag_id] = {"lstm": (x,y,z), "svm": (x,y,z), "rf": (x,y,z)}
tag_positions = {}
pos_lock      = threading.Lock()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 1 — DATA INGESTION  (MQTT thread, no predictions)
# ─────────────────────────────────────────────────────────────────────────────

def process_payload(payload: dict):
    """
    Pure ingestion — updates RSSI windows only. No model calls.

    Called on every arriving MQTT message. May be invoked many times per
    3-second cycle (one call per station that reports). Each call simply
    records the corrected RSSI into the rolling window so the prediction
    thread finds complete data when it wakes.
    """
    station_id = payload.get("stationId", "").upper()
    tags       = payload.get("tags", [])

    if not tags or station_id not in STATION_POSITIONS:
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

        # ── Store raw RSSI for the display table ──────────────────────────
        with rssi_lock:
            latest_rssi[tag_id][station_id] = float(rssi)

        # ── Gateway correction offset ──────────────────────────────────────
        correction_offset = 0.0
        if corrector.has_calibration(station_id):
            history = corrector.gateway_history[station_id]
            if history:
                correction_offset = corrector.get_correction_offset(
                    station_id, history[-1]
                )

        # ── Update corrected RSSI window — NO PREDICTION ──────────────────
        processor.update_station_rssi(
            station_id, tag_id, rssi,
            correction_offset=correction_offset
        )


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 2 — PREDICTION LOOP  (dedicated timer thread, once per cycle)
# ─────────────────────────────────────────────────────────────────────────────

def prediction_loop():
    """
    Wakes every PREDICTION_INTERVAL seconds (default 3 s).

    By then all stations visible in this BLE cycle have sent their payload
    and processor's RSSI windows are fully populated for this cycle.

    Sequence per wake:
        1. Build one 70-dim feature vector from all station windows.
        2. Count active stations — skip cycle if none visible yet.
        3. SVM + RF predict from the single snapshot (instant).
        4. Push snapshot into LSTM temporal buffer (1 entry per cycle).
           When buffer reaches SEQ_LEN (8 cycles = 24 s), LSTM predicts.
        5. Write results to tag_positions and results dicts.
    """
    target = TARGET_TAG.lower()
    cycle  = 0

    # Give MQTT time to receive first readings before attempting predictions
    time.sleep(PREDICTION_INTERVAL)

    while True:
        cycle_start = time.time()

        # ── 1. Build complete feature vector ──────────────────────────────
        feat_vec = processor.build_feature_vector(target)  # list[70 floats]

        # ── 2. Count stations that have data this cycle ────────────────────
        # mean feature is index 0, 5, 10, … (first of each 5-stat block)
        active_stations = sum(
            1 for i in range(0, len(feat_vec), 5)
            if feat_vec[i] > 0   # norm_mean > 0 → station has readings
        )

        if active_stations == 0:
            # No RSSI data from any station yet — wait for next cycle
            elapsed = time.time() - cycle_start
            time.sleep(max(0.0, PREDICTION_INTERVAL - elapsed))
            continue

        feat_arr = np.array(feat_vec, dtype=np.float64)

        # ── 3. SVM + RF: single snapshot ──────────────────────────────────
        svm_result = svm_pred.predict(feat_arr)
        rf_result  = rf_pred.predict(feat_arr)

        # ── 4. LSTM: push one entry per cycle into temporal buffer ─────────
        # push_to_buffer internally builds the same feature vector and
        # appends it; returns (SEQ_LEN, 70) array when buffer is full.
        sequence = processor.push_to_buffer(target)

        lstm_result = None
        if sequence is not None:
            lstm_result = lstm_pred.predict(sequence)

        # ── 5. Write to shared position and result stores ──────────────────
        with pos_lock:
            if target not in tag_positions:
                tag_positions[target] = {}
            tag_positions[target]["svm"] = (
                svm_result["x"], svm_result["y"], svm_result["z"]
            )
            tag_positions[target]["rf"] = (
                rf_result["x"], rf_result["y"], rf_result["z"]
            )
            if lstm_result is not None:
                tag_positions[target]["lstm"] = (
                    lstm_result["x"], lstm_result["y"], lstm_result["z"]
                )

        buf_status = processor.get_buffer_status(target)

        with lock:
            results[target] = {
                "coverage":        processor.get_station_coverage(target),
                "active_stations": active_stations,
                "buf_status":      buf_status,
                "svm_x":  svm_result["x"],
                "svm_y":  svm_result["y"],
                "svm_z":  svm_result["z"],
                "rf_x":   rf_result["x"],
                "rf_y":   rf_result["y"],
                "rf_z":   rf_result["z"],
                "lstm_x": lstm_result["x"] if lstm_result else None,
                "lstm_y": lstm_result["y"] if lstm_result else None,
                "lstm_z": lstm_result["z"] if lstm_result else None,
            }

        cycle += 1
        lstm_str = (f"LSTM=({lstm_result['x']:.1f},{lstm_result['y']:.1f})"
                    if lstm_result else f"LSTM=warming[{buf_status}]")
        print(
            f"[PRED] Cycle {cycle:4d}  stations={active_stations:2d}/14  "
            f"SVM=({svm_result['x']:.1f},{svm_result['y']:.1f})  "
            f"RF=({rf_result['x']:.1f},{rf_result['y']:.1f})  "
            f"{lstm_str}"
        )

        # Sleep for the remainder of this 3-second cycle
        elapsed = time.time() - cycle_start
        time.sleep(max(0.0, PREDICTION_INTERVAL - elapsed))


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 3 — MQTT CALLBACKS
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
        process_payload(payload)   # ingestion only — no predictions
    except json.JSONDecodeError as e:
        print(f"[ERROR] Bad JSON: {e}")
    except Exception as e:
        print(f"[ERROR] {e}")


def on_disconnect(client, userdata, rc, properties=None):
    print(f"[MQTT] Disconnected rc={rc}")


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 4 — MQTT THREAD
# ─────────────────────────────────────────────────────────────────────────────

def start_mqtt():
    time.sleep(1)
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
# SECTION 5 — REAL-TIME PLOT
# ─────────────────────────────────────────────────────────────────────────────

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
    Real-time display — refreshes every PREDICTION_INTERVAL seconds to stay
    in sync with the prediction cycle (no faster updates needed).

    Layout:
        Left (2/3)  — 27 m × 10 m room map.
                      ▲ = 14 anchor stations
                      ★ = PSO-LSTM   ■ = SVM (RBF)   ◆ = Random Forest

        Right (1/3) — two stacked panels:
                      Top    : top-6 station RSSI per tag
                      Bottom : position comparison table
    """
    plt.ion()
    fig  = plt.figure(figsize=(20, 8))
    fig.suptitle(
        "BLE Indoor Positioning — PSO-LSTM  ★   SVM  ■   Random Forest  ◆"
        f"   (predictions every {PREDICTION_INTERVAL:.0f} s)",
        fontsize=12, fontweight="bold"
    )

    outer   = gridspec.GridSpec(1, 2, width_ratios=[2.2, 1], figure=fig,
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
                f"Room {ROOM_W:.0f} m × {ROOM_H:.0f} m × {ROOM_Z:.0f} m  "
                f"│  {len(STATION_ORDER)} stations",
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
                    xytext=(4, 5), fontsize=6,
                    fontweight="bold", color="#2c3e50"
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
                        s=style["size"], color=style["color"],
                        marker=style["marker"], zorder=6, alpha=0.85,
                        label=f"{short} {style['label']}"
                    )
                    ax_room.annotate(
                        f"{style['label'][:4]}\n({mx:.1f},{my:.1f},{mz:.1f})",
                        (mx, my),
                        textcoords="offset points", xytext=(7, 7),
                        fontsize=6.5, color=style["color"]
                    )

                # Dashed triangle connecting the three estimates
                pts = [(pos[k][0], pos[k][1])
                       for k in ("lstm", "svm", "rf") if k in pos]
                if len(pts) >= 2:
                    xs = [p[0] for p in pts] + [pts[0][0]]
                    ys = [p[1] for p in pts] + [pts[0][1]]
                    ax_room.plot(xs, ys, "--", color="#95a5a6",
                                 alpha=0.5, linewidth=0.8)

            # Deduplicated legend
            handles, seen = [], set()
            for style in MODEL_STYLES.values():
                lbl = style["label"]
                if lbl not in seen:
                    seen.add(lbl)
                    handles.append(
                        plt.scatter([], [], s=style["size"] * 0.6,
                                    color=style["color"],
                                    marker=style["marker"], label=lbl)
                    )
            if handles:
                ax_room.legend(handles=handles, loc="upper right",
                               fontsize=8, framealpha=0.85)

            # ── Top-6 RSSI table ──────────────────────────────────────────
            ax_rssi.axis("off")
            with rssi_lock:
                rssi_snap = {k: v.copy() for k, v in latest_rssi.items()}
            with lock:
                res_snap = results.copy()

            if rssi_snap:
                rssi_rows = []
                for tag_id, stn_rssi in rssi_snap.items():
                    short  = tag_id[-11:]
                    top6   = sorted(stn_rssi.items(),
                                    key=lambda kv: kv[1], reverse=True)[:6]
                    best   = "  ".join(
                        f"{s.replace('STATION','S')}:{int(r)}" for s, r in top6
                    )
                    n_act  = res_snap.get(tag_id, {}).get("active_stations", "—")
                    rssi_rows.append([short, str(n_act), best])

                rssi_tbl = ax_rssi.table(
                    cellText  = rssi_rows,
                    colLabels = ["Tag", "Active", "Top-6 RSSI (dBm)"],
                    loc="center", cellLoc="left"
                )
                rssi_tbl.auto_set_font_size(False)
                rssi_tbl.set_fontsize(7)
                rssi_tbl.scale(1.0, 1.5)
                for c in range(3):
                    rssi_tbl[0, c].set_facecolor("#2c3e50")
                    rssi_tbl[0, c].set_text_props(
                        color="white", fontweight="bold"
                    )
                ax_rssi.set_title(
                    f"Live RSSI  (cycle = {PREDICTION_INTERVAL:.0f} s)",
                    fontsize=9
                )
            else:
                ax_rssi.text(0.5, 0.5, "Waiting for MQTT data...",
                             ha="center", va="center", fontsize=10,
                             color="gray", transform=ax_rssi.transAxes)

            # ── Position comparison table ─────────────────────────────────
            ax_pos.axis("off")
            if res_snap:
                pos_rows  = []
                col_hdrs  = ["Tag", "LSTM x", "LSTM y", "LSTM z",
                              "SVM x",  "SVM y",  "SVM z",
                              "RF x",   "RF y",   "RF z",  "Buf"]

                for tag_id, r in res_snap.items():
                    short  = tag_id[-11:]
                    lx = f"{r['lstm_x']:.1f}" if r.get("lstm_x") is not None else "—"
                    ly = f"{r['lstm_y']:.1f}" if r.get("lstm_y") is not None else "—"
                    lz = f"{r['lstm_z']:.1f}" if r.get("lstm_z") is not None else "—"
                    pos_rows.append([
                        short,
                        lx, ly, lz,
                        f"{r.get('svm_x',0):.1f}",
                        f"{r.get('svm_y',0):.1f}",
                        f"{r.get('svm_z',0):.1f}",
                        f"{r.get('rf_x', 0):.1f}",
                        f"{r.get('rf_y', 0):.1f}",
                        f"{r.get('rf_z', 0):.1f}",
                        r.get("buf_status", "—"),
                    ])

                pos_tbl = ax_pos.table(
                    cellText=pos_rows, colLabels=col_hdrs,
                    loc="center", cellLoc="center"
                )
                pos_tbl.auto_set_font_size(False)
                pos_tbl.set_fontsize(7)
                pos_tbl.scale(1.0, 1.6)

                hdr_colors = (
                    ["#2c3e50"] +
                    ["#922b21"] * 3 +   # LSTM
                    ["#1a5276"] * 3 +   # SVM
                    ["#1d6a39"] * 3 +   # RF
                    ["#4a235a"]          # Buf
                )
                for c, hc in enumerate(hdr_colors):
                    pos_tbl[0, c].set_facecolor(hc)
                    pos_tbl[0, c].set_text_props(color="white", fontweight="bold")
                for row in range(1, len(pos_rows) + 1):
                    bg = "#ecf0f1" if row % 2 == 0 else "white"
                    for c in range(len(col_hdrs)):
                        pos_tbl[row, c].set_facecolor(bg)

                ax_pos.set_title(
                    "★ PSO-LSTM   ■ SVM (RBF)   ◆ Random Forest   Buf=LSTM window",
                    fontsize=9
                )
            else:
                ax_pos.text(0.5, 0.5,
                            f"No predictions yet\n"
                            f"SVM/RF: ready after 1 cycle\n"
                            f"LSTM: ready after {SEQ_LEN} cycles ({SEQ_LEN * PREDICTION_INTERVAL:.0f} s)",
                            ha="center", va="center", fontsize=9,
                            color="gray", transform=ax_pos.transAxes)

            plt.tight_layout()
            plt.pause(PREDICTION_INTERVAL)   # refresh in sync with predictions

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"[PLOT] Error: {e}")
            time.sleep(1.0)

    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 6 — SUMMARY THREAD
# ─────────────────────────────────────────────────────────────────────────────

def print_summary():
    print("\n" + "=" * 82)
    print("  POSITION SUMMARY  —  coordinates in metres")
    print("=" * 82)
    with lock:
        if not results:
            print(f"  No predictions yet.  "
                  f"SVM/RF ready after 1 cycle ({PREDICTION_INTERVAL:.0f} s).  "
                  f"LSTM after {SEQ_LEN} cycles ({SEQ_LEN * PREDICTION_INTERVAL:.0f} s).")
        else:
            print(f"  {'TAG':<22}  "
                  f"{'LSTM x':>7} {'LSTM y':>7} {'LSTM z':>7}  "
                  f"{'SVM x':>7} {'SVM y':>7} {'SVM z':>7}  "
                  f"{'RF x':>6} {'RF y':>6} {'RF z':>6}  BUF")
            print("  " + "-" * 80)
            for tag_id, r in results.items():
                lx = f"{r['lstm_x']:.2f}" if r.get("lstm_x") is not None else "  —   "
                ly = f"{r['lstm_y']:.2f}" if r.get("lstm_y") is not None else "  —   "
                lz = f"{r['lstm_z']:.2f}" if r.get("lstm_z") is not None else "  —   "
                print(
                    f"  {tag_id:<22}  "
                    f"{lx:>7} {ly:>7} {lz:>7}  "
                    f"{r.get('svm_x',0):>7.2f} {r.get('svm_y',0):>7.2f} {r.get('svm_z',0):>7.2f}  "
                    f"{r.get('rf_x',0):>6.2f} {r.get('rf_y',0):>6.2f} {r.get('rf_z',0):>6.2f}  "
                    f"{r.get('buf_status','—')}"
                )
    print(f"\n  Gateway: {corrector.get_status()}")
    print("=" * 82 + "\n")


def print_summary_loop():
    while True:
        time.sleep(30)
        print_summary()


# ─────────────────────────────────────────────────────────────────────────────
# SECTION 7 — MAIN
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 68)
    print("  BLE Indoor Positioning — Three-Model Real-Time Comparison")
    print(f"  Broker        : {MQTT_BROKER}:{MQTT_PORT}")
    print(f"  Room          : {ROOM_W} m × {ROOM_H} m × {ROOM_Z} m")
    print(f"  Stations      : {len(STATION_ORDER)}")
    print(f"  Cycle period  : {PREDICTION_INTERVAL:.0f} s  (matches MQTT rate)")
    print(f"  Models        : PSO-LSTM (needs {SEQ_LEN} cycles = "
          f"{SEQ_LEN * PREDICTION_INTERVAL:.0f} s warm-up)")
    print(f"                  SVM-RBF + RandomForest (instant from cycle 1)")
    print(f"  Gateway nodes : STATION45, STATION47  (central anchors)")
    print("=" * 68 + "\n")

    # Ingestion thread — processes every arriving MQTT message
    threading.Thread(target=start_mqtt, daemon=True).start()

    # Prediction thread — fires once per PREDICTION_INTERVAL regardless of MQTT load
    threading.Thread(target=prediction_loop, daemon=True).start()

    # Summary thread — console printout every 30 s
    threading.Thread(target=print_summary_loop, daemon=True).start()

    # Main thread — real-time plot (must run on main thread on macOS/Linux GUI)
    try:
        run_plot()
    except KeyboardInterrupt:
        print("\n[MAIN] Shutting down...")


"""
Thread Timing Diagram
=====================

t=0s    MQTT arrives (STATION11 reports)  → process_payload → update window
t=0.2s  MQTT arrives (STATION12 reports)  → process_payload → update window
t=0.4s  MQTT arrives (STATION41 reports)  → process_payload → update window
  ...   (up to 14 station payloads within the 3s window)
t=3s  ──────────────────────────────────────────────────────────────────────
        prediction_loop wakes up
        feat_vec = build_feature_vector()   ← uses data from ALL 14 stations
        active_stations = count(feat > 0)
        svm_result  = SVMPositionPredictor.predict(feat_vec)
        rf_result   = RFPositionPredictor.predict(feat_vec)
        sequence    = push_to_buffer()      ← 1 temporal entry added (of 8)
        lstm_result = PositionPredictor.predict(seq)  [only after 8 cycles]
        write → tag_positions, results
t=6s  ── next cycle ──────────────────────────────────────────────────────
        (repeat)

Plot thread wakes every 3s (plt.pause(PREDICTION_INTERVAL)) — one refresh
per prediction cycle, so UI and predictions stay synchronised.
"""
