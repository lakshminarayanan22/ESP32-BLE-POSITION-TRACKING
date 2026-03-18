# ── MQTT ──────────────────────────────────────────────
MQTT_BROKER        = "192.168.1.11"
MQTT_PORT          = 1883
MQTT_USERNAME      = ""
MQTT_PASSWORD      = ""

# ── Station IDs (numeric suffix of station names) ─────
STATION_IDS = [11, 12, 13, 14, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
MQTT_TOPIC_PATTERN = "/nodejs/mqtt/#"

# ── Station Positions (x, y, z) in meters ─────────────
# Room 27 m × 10 m × 3 m
# Row 1 (y ≈ 2.0): STATION11–14  — south corridor
# Row 2 (y ≈ 5.9 / 7.9): STATION41–50 — north corridor
STATION_POSITIONS = {
    "STATION11": ( 4.8, 2.0, 1.8),
    "STATION12": ( 9.7, 2.0, 1.8),
    "STATION13": (15.7, 2.0, 1.8),
    "STATION14": (20.1, 2.0, 1.8),
    "STATION41": ( 1.3, 5.9, 1.8),
    "STATION42": ( 4.8, 7.9, 1.8),
    "STATION43": ( 6.2, 5.9, 1.8),
    "STATION44": ( 9.7, 7.9, 1.8),
    "STATION45": (11.7, 5.9, 1.8),
    "STATION46": (15.2, 7.9, 1.8),
    "STATION47": (16.6, 5.9, 1.8),
    "STATION48": (20.1, 7.9, 1.8),
    "STATION49": (22.1, 5.9, 1.8),
    "STATION50": (25.6, 7.9, 1.8),
}

# ── Room Dimensions (meters) ──────────────────────────
ROOM_W = 27.0
ROOM_H = 10.0
ROOM_Z =  3.0

# ── LSTM Model Parameters ─────────────────────────────
# SEQ_LEN=8: temporal sliding window — LSTM sees 8 consecutive snapshots,
#            capturing time-varying shadowing, NLOS, and multipath dynamics.
# N_FEATURES: 14 stations × 5 stats (mean, std, min, max, count) = 70
SEQ_LEN     = 8    # temporal window length (8–10 recommended)
N_FEATURES  = 70   # 14 stations × 5 stats per station
N_OUTPUTS   = 2    # predict (x, y) only — z not plotted
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2
MODEL_PATH  = "position_model.pth"

# ── Normalization ─────────────────────────────────────
RSSI_MIN = -120.0
RSSI_MAX =    0.0

# ── Data ──────────────────────────────────────────────
TRAINING_DATA_PATH = "training_data.json"
JSON_FILE          = "beacon_data.json"
JSON_REPEAT        = 1
TARGET_TAG         = "d6:06:9c:7e:ba:f7"

# ── Prediction Cycle ───────────────────────────────────
# Each BLE station publishes one MQTT payload every 3 seconds.
# Predictions are triggered by a dedicated timer thread at the same rate,
# AFTER the ingestion window has closed. This ensures all stations that
# are visible in a cycle have contributed their RSSI before any model runs.
# Setting this to match the station publish interval prevents partial/empty
# feature vectors from reaching SVM, RF, and PSO-LSTM.
PREDICTION_INTERVAL = 3.0   # seconds — must match station publish rate

# ── PSO Hyperparameter Optimization ───────────────────
# PSO finds the optimal LSTM config before Adam fine-tuning (PSO-Adam hybrid).
# Each particle encodes: [hidden_size, num_layers, log10(lr), dropout]
# Fitness = validation MSE after PSO_EVAL_EPOCHS of Adam training per particle.
PSO_SWARM_SIZE  = 20   # particles in the swarm
PSO_MAX_ITER    = 50   # maximum PSO iterations
PSO_C1          = 1.5  # cognitive coefficient (personal best attraction)
PSO_C2          = 1.5  # social coefficient (global best attraction)
PSO_W_INIT      = 0.9  # initial inertia weight (broad exploration)
PSO_W_MIN       = 0.4  # minimum inertia weight (linearly decayed → exploitation)
PSO_EVAL_EPOCHS = 30   # epochs to evaluate each particle configuration

# ── Bluetooth Gateway RSSI Correction ─────────────────
# Two central anchor stations monitor beacon RSSI from a fixed position.
# Any RSSI change at the gateway = beacon transmit power drift, not movement.
# Correction offset: ΔA_l = R_gateway_current - R̄_gateway_historical
# Mobile corrected RSSI = raw_rssi - ΔA_l
#
# Anchor selection — closest to room geometric center (13.5 m, 5.0 m):
#   STATION45 at (11.7, 5.9) — distance ≈ 2.0 m from center  ← primary
#   STATION47 at (16.6, 5.9) — distance ≈ 3.2 m from center  ← secondary
GATEWAY_HISTORY_LEN  = 50          # rolling window size for gateway RSSI history
GATEWAY_STATION_ID   = "STATION45" # primary correction reference anchor
GATEWAY_STATION_IDS  = ["STATION45", "STATION47"]  # both anchor nodes

# ── SVM and Random Forest Model Paths ─────────────────
SVM_MODEL_PATH = "svm_model.pkl"
RF_MODEL_PATH  = "rf_model.pkl"
