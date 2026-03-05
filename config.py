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
ROOM_H =  3.6
ROOM_Z = 3.0

# ── Model ─────────────────────────────────────────────
SEQ_LEN     = 1    # timesteps of RSSI history
N_FEATURES  = 20      # 4stationsx5features
N_OUTPUTS   = 3       # predict (x, y, z)
HIDDEN_SIZE = 128
NUM_LAYERS  = 2
DROPOUT     = 0.2
MODEL_PATH  = "/Users/lnarayanansk/Documents/Doodil2r/position_model.pth"

# ── Normalization ─────────────────────────────────────
RSSI_MIN = -120.0
RSSI_MAX =    0.0

# ── Data ──────────────────────────────────────────────
TRAINING_DATA_PATH = "/Users/lnarayanansk/Documents/Doodil2r/training_data.json"
JSON_FILE          = "/Users/lnarayanansk/Documents/Doodil2r/beacon_data.json"
JSON_REPEAT        = 1
TARGET_TAG = "d6:06:9c:7e:ba:f7"
