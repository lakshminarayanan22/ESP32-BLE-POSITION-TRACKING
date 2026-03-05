import csv
import numpy as np
import torch
import torch.nn as nn
from collections             import defaultdict
from torch.utils.data        import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys

from LSTM   import LSTMPositionModel
from config import (
    SEQ_LEN, N_FEATURES, N_OUTPUTS,
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    MODEL_PATH, RSSI_MIN, RSSI_MAX,
    ROOM_W, ROOM_H, ROOM_Z,
    STATION_POSITIONS
)

# ── Config ────────────────────────────────────────────
TRAINING_CSV_PATH    = "/Users/lnarayanansk/Downloads/tag_training_data.csv"
PLOT_PATH            = "/Users/lnarayanansk/Documents/training_results.png"
EPOCHS               = 5000
BATCH_SIZE           = 32
LR                   = 0.01
TRAIN_SPLIT          = 0.8
STATION_ORDER        = list(STATION_POSITIONS.keys())
RSSI_COLS            = ["rssi_value_1", "rssi_value_2",
                         "rssi_value_3", "rssi_value_4"]
STATS_PER_STATION    = 5     # mean, std, min, max, count
# N_FEATURES = 4 stations × 5 stats = 20  (set in config.py)

# ── Normalize / Denormalize ───────────────────────────
def norm_rssi(v):  return (v - RSSI_MIN) / (RSSI_MAX - RSSI_MIN)
def norm_x(v):     return v / ROOM_W
def norm_y(v):     return v / ROOM_H
def norm_z(v):     return v / ROOM_Z
def denorm_x(v):   return v * ROOM_W
def denorm_y(v):   return v * ROOM_H
def denorm_z(v):   return v * ROOM_Z

# ── Extract Non-Null RSSI Values From One Row ─────────
def extract_rssi_values(row: dict) -> list:
    """
    Reads rssi_value_1 to rssi_value_4.
    Returns ONLY non-empty values — no padding, no nulls.

    Examples:
        row has -67, -71, '',  ''  → returns [-67.0, -71.0]
        row has -67, '',  '',  ''  → returns [-67.0]
        row has -67, -71, -69, -72 → returns [-67.0, -71.0, -69.0, -72.0]
        row has '',  '',  '',  ''  → returns []
    """
    values = []
    for col in RSSI_COLS:
        raw = row.get(col, "").strip()
        if raw:                          # only non-empty
            try:
                values.append(float(raw))
            except ValueError:
                pass
    return values                        # length: 0, 1, 2, 3, or 4


# ── Compute Stats From Variable-Length RSSI List ──────
def compute_station_stats(rssi_values: list) -> list:
    """
    Takes however many non-null RSSI values exist for
    one station in one interval and returns 5 stats.

    Stats:
        mean  — average signal strength
        std   — spread / deviation of readings
        min   — weakest reading
        max   — strongest reading
        count — how many readings existed (normalized 0–1)

    No padding. No nulls. Works for 0, 1, 2, 3, or 4 values.

    All values normalized to [0, 1].

    Case: 0 values (station not seen)
        → all stats = 0.0

    Case: 1 value
        → mean = that value, std = 0.0, min = max = that value

    Case: 2, 3, or 4 values
        → full stats computed from actual values only
    """
    if not rssi_values:
        # Station completely unseen this interval
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    arr = np.array(rssi_values, dtype=np.float32)

    mean_v  = float(np.mean(arr))
    std_v   = float(np.std(arr))     # 0.0 if only 1 value
    min_v   = float(np.min(arr))
    max_v   = float(np.max(arr))
    count_v = len(rssi_values)       # 1, 2, 3, or 4

    # Normalize
    norm_mean  = norm_rssi(mean_v)
    norm_std   = std_v / ((RSSI_MAX - RSSI_MIN) / 2.0)   # max possible std
    norm_min   = norm_rssi(min_v)
    norm_max   = norm_rssi(max_v)
    norm_count = count_v / len(RSSI_COLS)                 # 0.25, 0.5, 0.75, 1.0

    return [norm_mean, norm_std, norm_min, norm_max, norm_count]


# ── Load CSV ──────────────────────────────────────────
def load_csv(filepath: str) -> list:
    """
    CSV format:
        start_time, end_time, station, tag,
        rssi_value_1, rssi_value_2, rssi_value_3, rssi_value_4,
        true_x, true_y, true_z

    USED    → station, rssi_value_1..4 (non-null only), true_x/y/z
    IGNORED → start_time, end_time, tag

    Per interval, per station:
        - collect all non-null RSSI values (1 to 4)
        - compute 5 stats from actual values only
        - no padding, no null substitution

    Feature vector per interval = 20 values:
        [mean, std, min, max, count] × 4 stations

    Returns: sorted list of interval dicts:
    [
        {
            "feature": [20 floats],
            "target":  [3 floats]    ← x, y, z
        },
        ...
    ]
    """
    # raw[(start, end)][station] = list of all rssi floats
    # Multiple rows for same station/interval are all collected
    raw    = defaultdict(lambda: {sid: [] for sid in STATION_ORDER})
    coords = {}

    total_rows      = 0
    skipped_station = 0
    skipped_no_rssi = 0
    found_stations  = set()

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        print(f"[DEBUG] CSV Headers        : {reader.fieldnames}\n")

        for row in reader:
            total_rows += 1

            # Read identifying columns — NOT used as features
            start   = row.get("start_time", "").strip()
            end     = row.get("end_time",   "").strip()
            station = row.get("station",    "").upper().strip()
            _tag    = row.get("tag", "")     # read and discarded

            if not start or not end:
                continue

            if station not in STATION_ORDER:
                skipped_station += 1
                found_stations.add(station)
                continue

            # Extract only non-null values — no padding
            rssi_values = extract_rssi_values(row)
            if not rssi_values:
                skipped_no_rssi += 1
                continue

            # Accumulate into this station's list for this interval
            # If multiple rows exist for same station/interval
            # all their values are pooled together
            raw[(start, end)][station].extend(rssi_values)

            # Coordinates — store once per interval
            key = (start, end)
            if key not in coords:
                try:
                    coords[key] = (
                        float(row["true_x"]),
                        float(row["true_y"]),
                        float(row["true_z"])
                    )
                except (KeyError, ValueError):
                    pass

    # Debug
    print(f"[DEBUG] Rows read              : {total_rows}")
    print(f"[DEBUG] Skipped (bad station)  : {skipped_station}")
    print(f"[DEBUG] Skipped (no RSSI)      : {skipped_no_rssi}")
    if found_stations - set(STATION_ORDER):
        print(f"[DEBUG] Unknown stations       : "
              f"{found_stations - set(STATION_ORDER)}")
        print(f"[DEBUG] Expected stations      : {STATION_ORDER}")
    print(f"[DEBUG] Valid intervals        : {len(raw)}\n")

    # Build interval list
    intervals = []

    for (start, end), station_readings in sorted(raw.items()):
        key = (start, end)
        if key not in coords:
            continue

        # Build 20-dim feature vector
        # For each station compute stats from actual values only
        feature_vec = []
        station_info = []   # for debug printing

        for sid in STATION_ORDER:
            rssi_vals = station_readings[sid]  # [] if station not seen

            # Stats computed from actual non-null values only
            stats = compute_station_stats(rssi_vals)
            feature_vec.extend(stats)          # 5 values added

            station_info.append(
                f"{sid}:{len(rssi_vals)}vals"
            )

        # Sanity check
        assert len(feature_vec) == len(STATION_ORDER) * STATS_PER_STATION, \
            f"Feature size mismatch: {len(feature_vec)}"

        tx, ty, tz = coords[key]
        intervals.append({
            "feature": feature_vec,
            "target":  [norm_x(tx), norm_y(ty), norm_z(tz)]
        })

    # Print stats summary
    print(f"[DATA] Intervals loaded        : {len(intervals)}")
    print(f"[DATA] Feature vector size     : "
          f"{len(STATION_ORDER)} stations × "
          f"{STATS_PER_STATION} stats = "
          f"{len(STATION_ORDER) * STATS_PER_STATION}")
    print(f"       └─ per station          : "
          f"[mean, std, min, max, count]")
    print(f"       └─ computed from        : "
          f"actual non-null values only (1–4)")
    print(f"[DATA] Target                  : [x, y, z]")
    print(f"[DATA] Excluded                : "
          f"start_time, end_time, tag\n")

    # Show RSSI value count distribution
    _print_value_distribution(raw)

    if not intervals:
        print("[WARN] No intervals built — "
              "check station names and CSV headers")

    return intervals


def _print_value_distribution(raw: dict):
    """Show how many intervals had 1, 2, 3, 4 values per station."""
    print(f"[DATA] RSSI value count distribution per station:")
    print(f"  {'STATION':<12}  {'0vals':>6}  {'1val':>6}  "
          f"{'2vals':>6}  {'3vals':>6}  {'4vals':>6}")
    print("  " + "-" * 50)

    for sid in STATION_ORDER:
        counts = defaultdict(int)
        for interval_stations in raw.values():
            n = len(interval_stations[sid])
            counts[n] += 1
        print(
            f"  {sid:<12}  "
            f"{counts[0]:>6}  "
            f"{counts[1]:>6}  "
            f"{counts[2]:>6}  "
            f"{counts[3]:>6}  "
            f"{counts[4]:>6}"
        )
    print()


# ── Build Sequences ───────────────────────────────────
def build_sequences(intervals: list):
    """
    Slides SEQ_LEN window over interval list.

    X[i] shape: (SEQ_LEN, 20)
        → no padding, no nulls
        → stats computed from actual values per station

    y[i] shape: (3,) → x, y, z
    """
    if len(intervals) < SEQ_LEN + 1:
        print(f"[ERROR] Only {len(intervals)} intervals — "
              f"need {SEQ_LEN + 1}. Lower SEQ_LEN in config.py")
        return np.array([]), np.array([])

    X_all, y_all = [], []

    for i in range(len(intervals) - SEQ_LEN):
        X_all.append([
            intervals[j]["feature"]
            for j in range(i, i + SEQ_LEN)
        ])
        y_all.append(intervals[i + SEQ_LEN]["target"])

    X = np.array(X_all, dtype=np.float32)   # (N, SEQ_LEN, 20)
    y = np.array(y_all, dtype=np.float32)   # (N, 3)

    print(f"[DATA] Sequences               : {len(X)}")
    print(f"[DATA] X shape                 : {X.shape}")
    print(f"       └─ (samples, seq_len, features)")
    print(f"[DATA] y shape                 : {y.shape}")
    print(f"       └─ (samples, x/y/z)\n")
    return X, y


# ── Dataset ───────────────────────────────────────────
class PositionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Train / Eval ──────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        optimizer.zero_grad()
        loss = criterion(model(X_b), y_b)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / len(loader)

def evaluate(model, loader, criterion, device):
    model.eval()
    total = 0
    with torch.no_grad():
        for X_b, y_b in loader:
            total += criterion(
                model(X_b.to(device)), y_b.to(device)
            ).item()
    return total / len(loader)


# ── Plot ──────────────────────────────────────────────
def plot_results(train_losses, val_losses, true_m, pred_m):
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))

    axes[0].plot(train_losses, label="Train")
    axes[0].plot(val_losses,   label="Val")
    axes[0].axvline(x=EPOCHS - 1, color="red",
                    linestyle="--", label=f"Save ep{EPOCHS}")
    axes[0].set_title("Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True)

    for idx, label in enumerate(["X", "Y", "Z"]):
        ax    = axes[idx + 1]
        limit = min(100, len(true_m))
        ax.plot(true_m[:limit, idx], label=f"True {label}")
        ax.plot(pred_m[:limit, idx], label=f"Pred {label}",
                linestyle="--")
        ax.set_title(f"{label} Coordinate")
        ax.set_xlabel("Sample")
        ax.set_ylabel(f"{label} (m)")
        ax.legend()
        ax.grid(True)

    plt.suptitle(f"LSTM 3D Position — Epoch {EPOCHS}", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"[PLOT] Saved → {PLOT_PATH}")


# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  LSTM 3D Position Trainer")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {EPOCHS}")
    print(f"  SEQ_LEN    : {SEQ_LEN}")
    print(f"  Features   : {len(STATION_ORDER)} stations × "
          f"{STATS_PER_STATION} stats = "
          f"{len(STATION_ORDER) * STATS_PER_STATION}")
    print(f"  Stats      : mean, std, min, max, count")
    print(f"  No padding : null values skipped entirely")
    print(f"  Output     : x, y, z")
    print(f"  Excluded   : start_time, end_time, tag")
    print("=" * 60 + "\n")

    # 1. Load
    intervals = load_csv(TRAINING_CSV_PATH)

    # 2. Sequences
    X, y = build_sequences(intervals)

    if len(X) == 0:
        print("\n[ERROR] No sequences. Check:")
        print("  Headers: start_time, end_time, station, tag,")
        print("           rssi_value_1, rssi_value_2,")
        print("           rssi_value_3, rssi_value_4,")
        print("           true_x, true_y, true_z")
        print(f"  Station names match config : {STATION_ORDER}")
        print(f"  SEQ_LEN ({SEQ_LEN}) <= interval count")
        sys.exit(1)

    # Verify feature size
    actual_features = X.shape[2]
    expected        = len(STATION_ORDER) * STATS_PER_STATION
    if actual_features != N_FEATURES:
        print(f"\n[ERROR] Feature size mismatch:")
        print(f"  Script computed : {actual_features}")
        print(f"  config N_FEATURES: {N_FEATURES}")
        print(f"  Fix: set N_FEATURES = {expected} in config.py")
        sys.exit(1)

    # 3. Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=1 - TRAIN_SPLIT, shuffle=False
    )
    print(f"[DATA] Train : {len(X_train)}  Val : {len(X_val)}\n")

    train_loader = DataLoader(
        PositionDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        PositionDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=True
    )

    # 4. Model
    model = LSTMPositionModel(
        input_size  = N_FEATURES,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        output_size = N_OUTPUTS,
        dropout     = DROPOUT
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,mode='min',factor=0.1,patience=20,
        threshold=1e-4,min_lr=1e-4,eps=1e-8
    )

    # 5. Train 40 epochs
    train_losses, val_losses = [], []
    print(f"  {'Epoch':<8} {'Train':>12}  {'Val':>10}  {'LR':>10}")
    print("  " + "-" * 48)

    for epoch in range(EPOCHS):
        tl = train_epoch(model, train_loader, optimizer, criterion, device)
        vl = evaluate(model,   val_loader,   criterion, device)
        scheduler.step(vl)
        train_losses.append(tl)
        val_losses.append(vl)
        lr = optimizer.param_groups[0]["lr"]
        print(f"  {epoch+1:<8} {tl:>12.6f}  {vl:>10.6f}  {lr:>10.6f}")

    # 6. Save after epoch 40
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n[SAVE] → {MODEL_PATH}")

    # 7. Evaluate
    model.eval()
    all_pred, all_true = [], []
    with torch.no_grad():
        for X_b, y_b in val_loader:
            all_pred.append(model(X_b.to(device)).cpu().numpy())
            all_true.append(y_b.numpy())

    all_pred = np.concatenate(all_pred)
    all_true = np.concatenate(all_true)

    pred_m = np.stack([
        denorm_x(all_pred[:, 0]),
        denorm_y(all_pred[:, 1]),
        denorm_z(all_pred[:, 2])
    ], axis=1)
    true_m = np.stack([
        denorm_x(all_true[:, 0]),
        denorm_y(all_true[:, 1]),
        denorm_z(all_true[:, 2])
    ], axis=1)

    errors = np.sqrt(np.sum((pred_m - true_m)**2, axis=1))
    print(f"\n[EVAL] Mean 3D error : {np.mean(errors):.2f} m")
    print(f"[EVAL] Median error  : {np.median(errors):.2f} m")
    print(f"[EVAL] Max error     : {np.max(errors):.2f} m")
    print(f"[EVAL] Error < 1m   : {np.mean(errors < 1.0)*100:.1f}%")
    print(f"[EVAL] Error < 2m   : {np.mean(errors < 2.0)*100:.1f}%")

    plot_results(train_losses, val_losses, true_m, pred_m)
    print("\n[DONE] Run main.py to use the model.")

"""
## Key Changes From Previous Version

| Issue | Old Code | Fixed Code |
|---|---|---|
| CSV column `rssi_value_1..4` | Used `rssi_1..4` | Corrected to `rssi_value_1..4` |
| Mean filter | Averaged per station across rows | Averages non-null values among 4 columns in same row |
| `tag_id` as feature | Grouped by tag | Single tag dataset — tag excluded from features |
| Timestamps as feature | Stored in feature | Used only for sorting, excluded from `X` |
| `denorm_y` bug | `v / ROOM_H` | Fixed to `v * ROOM_H` |
| Data grouping | Per tag then per interval | Directly per interval (single tag assumption) |

---

## How Mean Filter Works
```
CSV row for STATION1, interval 10:00→10:03:
    rssi_value_1 = -67
    rssi_value_2 = -71
    rssi_value_3 =        ← empty
    rssi_value_4 =        ← empty

mean_rssi() → mean(-67, -71) = -69.0
                    ↑
           only non-empty values counted
           denominator = 2, not 4

"""

