import json
import time
import threading
import os
import paho.mqtt.client as mqtt

from config import *
from tag_processing import DataProcessor
from rssi_predictor import RSSIPredictor

# ── Global Instances ──────────────────────────────────
processor = DataProcessor()
predictor = RSSIPredictor()
results   = {}
lock      = threading.Lock()

# ── Process One Full Payload ──────────────────────────
def process_payload(payload: dict):
    station_id, tags = processor.parse_payload(payload)

    if not tags:
        print(f"[WARN] No valid tags in payload from {station_id}")
        return

    print(f"\n[{station_id}] Received {len(tags)} tags")
    print(f"  {'TAG ID':<22} {'RSSI':>6}  {'BUFFER':>10}  {'PRED RSSI':>10}  {'DIST':>8}  QUALITY")
    print("  " + "-" * 75)

    for tag_id, rssi in tags:
        sequence = processor.update_buffer(station_id, tag_id, rssi)
        buf_len  = len(processor.buffers[station_id][tag_id])
        buf_str  = f"{buf_len}/{SEQ_LEN}"

        if sequence is None:
            print(f"  {tag_id:<22} {rssi:>6} dBm  {buf_str:>10}  {'collecting...':>10}")
            continue

        result = predictor.predict(sequence)

        with lock:
            if station_id not in results:
                results[station_id] = {}
            results[station_id][tag_id] = {
                "current_rssi": rssi,
                **result
            }

        print(
            f"  {tag_id:<22} {rssi:>6} dBm  {buf_str:>10}"
            f"  {result['predicted_rssi']:>8.1f}  "
            f"{result['predicted_distance']:>6.1f}m  "
            f"{result['signal_quality']}"
        )

JSON_REPEAT = 10
# ── Load From JSON File ───────────────────────────────
def run_from_json(json_path: str):
    if not os.path.exists(json_path):
        print(f"[ERROR] JSON file not found: {json_path}")
        return

    with open(json_path, "r") as f:
        data = json.load(f)

    # Handle single dict or list
    if isinstance(data, dict):
        # Single payload — repeat it JSON_REPEAT times to fill buffer
        payloads = [data] * JSON_REPEAT
        print(f"[FILE] Single payload detected — repeating {JSON_REPEAT}x to fill buffer")
    elif isinstance(data, list):
        payloads = data
        print(f"[FILE] Loaded {len(payloads)} payloads from {json_path}")
    else:
        print("[ERROR] JSON must be a dict or list of dicts")
        return

    print(f"[FILE] Need {SEQ_LEN} readings per tag before first prediction\n")

    for i, payload in enumerate(payloads):
        print(f"[FILE] ── Iteration {i + 1}/{len(payloads)} ──")
        process_payload(payload)
        time.sleep(0.5)   # small delay so output is readable

    print_final_summary()

# ── MQTT Callbacks ────────────────────────────────────
def on_connect(client, userdata, flags, reason_code, properties):
    if reason_code == 0:
        print(f"[MQTT] Connected to {MQTT_BROKER}:{MQTT_PORT}")
        for sid in STATION_IDS:
            topic = MQTT_TOPIC_PATTERN.format(sid)
            client.subscribe(topic)
            print(f"[MQTT] Subscribed → {topic}")
    else:
        print(f"[MQTT] Connection failed rc={reason_code}")

def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode("utf-8"))
        process_payload(payload)
    except json.JSONDecodeError as e:
        print(f"[ERROR] Bad JSON on {msg.topic}: {e}")
    except Exception as e:
        print(f"[ERROR] {e}")

def on_disconnect(client, userdata, rc, properties=None):
    print(f"[MQTT] Disconnected rc={rc}")

# ── Summary ───────────────────────────────────────────
def print_final_summary():
    print("\n" + "=" * 60)
    print("FINAL PREDICTION SUMMARY")
    print("=" * 60)
    with lock:
        if not results:
            print("  No predictions yet — increase JSON_REPEAT or lower SEQ_LEN")
            return
        for station, tags in results.items():
            print(f"\n  {station}:")
            for tag, res in tags.items():
                print(
                    f"    {tag:<22}  |  "
                    f"Current: {res['current_rssi']:>5} dBm  |  "
                    f"Predicted: {res['predicted_rssi']:>7.1f} dBm  |  "
                    f"Dist: {res['predicted_distance']:>5.1f}m  |  "
                    f"{res['signal_quality']}"
                )
    print("=" * 60 + "\n")

def print_summary_loop():
    while True:
        time.sleep(30)
        print_final_summary()

JSON_FILE = "/Users/lnarayanansk/Documents/Doodil2r/beacon_data.json"
# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 60)
    print("  BLE RSSI LSTM Predictor")
    print(f"  SEQ_LEN  : {SEQ_LEN} readings ({SEQ_LEN * 3}s warmup per tag)")
    print("=" * 60 + "\n")

    if JSON_FILE and os.path.exists(JSON_FILE):
        print(f"[MODE] FILE — {JSON_FILE}\n")
        run_from_json(JSON_FILE)

    else:
        print("[MODE] LIVE MQTT\n")
        summary_thread = threading.Thread(target=print_summary_loop, daemon=True)
        summary_thread.start()

        client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2, client_id="rssi_lstm_predictor")
        client.on_connect    = on_connect
        client.on_message    = on_message
        client.on_disconnect = on_disconnect

        if MQTT_USERNAME:
            client.username_pw_set(MQTT_USERNAME, MQTT_PASSWORD)

        client.connect(MQTT_BROKER, MQTT_PORT, keepalive=60)

        try:
            client.loop_forever()
        except KeyboardInterrupt:
            print("\n[MAIN] Shutting down...")
            client.disconnect()

"""

============================================================
  BLE RSSI LSTM Predictor
  Stations : [14, 15, 16]
  SEQ_LEN  : 10 readings (30s warmup per tag)
============================================================

[MQTT] Connected to 192.168.1.100:1883
[MQTT] Subscribed → /nodejs/mqtt/STATION14

[STATION14] Received 7 tags
  TAG ID                  RSSI      BUFFER   PRED RSSI    DIST  QUALITY
  ---------------------------------------------------------------------------
  BC:57:29:05:8E:D6        -96 dBm       1/10  collecting...
  CA:15:AC:2E:E6:FC        -50 dBm       1/10  collecting...
  FB:DE:E9:40:FB:2B        -46 dBm       1/10  collecting...
  54:AB:F6:5E:4E:03        -67 dBm       1/10  collecting...
  DC:0A:E4:D2:DA:E3        -57 dBm       1/10  collecting...
  CB:FA:84:59:D7:06        -91 dBm       1/10  collecting...
  23:97:5E:7F:36:45        -95 dBm       1/10  collecting...

# ... after 10 readings (30 seconds) ...

[STATION14] Received 7 tags
  TAG ID                  RSSI      BUFFER   PRED RSSI    DIST  QUALITY
  ---------------------------------------------------------------------------
  BC:57:29:05:8E:D6        -96 dBm      10/10      -94.3   44.7m  VERY_WEAK
  CA:15:AC:2E:E6:FC        -50 dBm      10/10      -51.2    3.6m  EXCELLENT
  FB:DE:E9:40:FB:2B        -46 dBm      10/10      -47.1    2.9m  EXCELLENT
  54:AB:F6:5E:4E:03        -67 dBm      10/10      -65.8    9.8m  GOOD
  DC:0A:E4:D2:DA:E3        -57 dBm      10/10      -58.3    5.5m  GOOD
  CB:FA:84:59:D7:06        -91 dBm      10/10      -89.7   30.5m  WEAK
  23:97:5E:7F:36:45        -95 dBm      10/10      -93.1   40.2m  VERY_WEAK

"""

