# ── MQTT ──────────────────────────────────────────────
MQTT_BROKER        = "192.168.1.11"
MQTT_PORT          = 1883
MQTT_USERNAME      = ""
MQTT_PASSWORD      = ""

# ── Stations & their known coordinates (meters) ───────
STATION_IDS = [14, 15, 16, 17]
MQTT_TOPIC_PATTERN = "/nodejs/mqtt/#"

STATION_POSITIONS = {
    "STATION1": (1.6, 0.0, 1.5),
    "STATION2": (0.0, 2.0, 1.5),
    "STATION3": (2.0, 3.6, 1.5),
    "STATION4": (3.8, 1.8, 1.5)
}

# ── Room Dimensions (meters) ──────────────────────────
ROOM_W = 3.8
ROOM_H = 3.6
ROOM_Z = 3.0

# ── LSTM Model Parameters ─────────────────────────────
# SEQ_LEN=8: sliding window of 8 consecutive feature snapshots fed to LSTM
# This enables the network to learn temporal RSSI patterns across time steps
SEQ_LEN     = 8    # temporal sliding window length (8–10 recommended)
N_FEATURES  = 20   # 4 stations × 5 stats (mean, std, min, max, count)
N_OUTPUTS   = 3    # predict (x, y, z)
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
TARGET_TAG = "d6:06:9c:7e:ba:f7"

# ── PSO Hyperparameter Optimization ───────────────────
# PSO is used to find optimal LSTM hyperparameters before Adam fine-tuning.
# Each particle encodes: [hidden_size, num_layers, log10(lr), dropout]
# Fitness function: validation MSE after PSO_EVAL_EPOCHS of Adam training.
PSO_SWARM_SIZE  = 20   # number of particles in the swarm
PSO_MAX_ITER    = 50   # maximum PSO iterations
PSO_C1          = 1.5  # cognitive acceleration (personal best attraction)
PSO_C2          = 1.5  # social acceleration (global best attraction)
PSO_W_INIT      = 0.9  # initial inertia weight (allows broad exploration)
PSO_W_MIN       = 0.4  # minimum inertia weight (linearly decayed for exploitation)
PSO_EVAL_EPOCHS = 30   # epochs to train each particle config for fitness eval

# ── Bluetooth Gateway RSSI Correction ─────────────────
# A gateway at a known fixed position monitors each beacon's RSSI.
# Since distance is constant, RSSI changes at gateway == transmit power drift.
# Correction offset: delta_A = R_gateway_current - R_gateway_historical_mean
# Mobile node corrected RSSI = raw_rssi - delta_A
GATEWAY_HISTORY_LEN = 50   # rolling window size for gateway RSSI history
GATEWAY_STATION_ID  = "STATION1"  # which station acts as correction reference
