# config.py

# ── MQTT Settings ─────────────────────────────────────
MQTT_BROKER        = ""
MQTT_PORT          = 1883
MQTT_USERNAME      = ""
MQTT_PASSWORD      = ""

# ── Station Settings ──────────────────────────────────
STATION_IDS        = [1,2,3,4]
MQTT_TOPIC_PATTERN = "/nodejs/mqtt/STATION{}"

# ── Station Physical Positions ────────────────────────
STATION_POSITIONS  = {
    "STATION1": (0.0,  0.0),
    "STATION2": (10.0, 0.0),
    "STATION3": (5.0,  8.0),
    "STATION4": ()
}

# ── Model Settings ────────────────────────────────────
SEQ_LEN       = 5
N_FEATURES    = 1
HIDDEN_SIZE   = 64
NUM_LAYERS    = 2
DROPOUT       = 0.2

# ── FIX: Use full absolute path ───────────────────────
MODEL_PATH    = "/Users/lnarayanansk/Documents/Doodil2r/best_model.pth"

# ── RSSI Range ────────────────────────────────────────
RSSI_MIN      = -120.0
RSSI_MAX      =    0.0
TX_POWER      =  -59.0
PATH_LOSS_N   =    2.0

# ── JSON File Settings ────────────────────────────────
JSON_FILE     = "/Users/lnarayanansk/Documents/Doodil2r/beacon_data.json"
JSON_REPEAT   = 1
