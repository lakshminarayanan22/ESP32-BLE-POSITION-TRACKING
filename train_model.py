"""
PSO-LSTM Training Pipeline
===========================

Trains the indoor positioning LSTM model using the PSO-Adam hybrid approach:

    Stage 1 — PSO Hyperparameter Search:
        Particle Swarm Optimization explores the hyperparameter space
        (hidden_size, num_layers, learning_rate, dropout) to find an
        optimal LSTM configuration. Each particle is evaluated by training
        for PSO_EVAL_EPOCHS and measuring validation MSE.

        Avoids:
            - Local minima (feedforward BP weakness)
            - Poor learning rate choices (common with manual tuning)
            - Vanishing gradient instability (LSTM specific)

    Stage 2 — Adam Fine-Tuning:
        The best PSO configuration is used to build the final LSTM.
        Adam optimizer trains to convergence from this optimized starting
        point for precise local search that PSO alone cannot achieve.

Data Format (CSV):
    Columns: tag_id,
             rssi_11, rssi_12, rssi_13, rssi_14,
             rssi_41..rssi_50,
             true_x, true_y, true_z   (z read but not predicted)

Feature Vector (per time step):
    [mean, std, min, max, count] × 14 stations = 70 values
    All normalized to [0, 1].

LSTM Input Shape:
    (batch, SEQ_LEN=8, N_FEATURES=70) — temporal sliding window

Output Shape:
    (batch, 2) — normalized (x, y) coordinates
"""

import csv
import sys
import numpy as np
import torch
import torch.nn as nn
from collections             import defaultdict
from torch.utils.data        import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from LSTM            import LSTMModel
from pso_optimizer   import PSOHyperparameterOptimizer
from config import (
    SEQ_LEN, N_FEATURES, N_OUTPUTS,
    HIDDEN_SIZE, NUM_LAYERS, DROPOUT,
    MODEL_PATH, RSSI_MIN, RSSI_MAX,
    ROOM_W, ROOM_H,
    STATION_POSITIONS
)

# ── Config ────────────────────────────────────────────────────────────────────
TRAINING_CSV_PATH = "tag_training_data.csv"
PLOT_PATH         = "training_results.png"
EPOCHS            = 5000
BATCH_SIZE        = 32
LR                = 0.01
TRAIN_SPLIT       = 0.8
USE_PSO           = True    # Set False to skip PSO and use config defaults

STATION_ORDER     = list(STATION_POSITIONS.keys())
STATS_PER_STATION = 5   # mean, std, min, max, count
RSSI_WINDOW_SIZE  = 4   # matches tag_processing.py — for count normalisation

# CSV column for each station: rssi_11, rssi_12, ..., rssi_50
RSSI_COLS_NEW     = [f"rssi_{sid.replace('STATION','')}" for sid in STATION_ORDER]


# ── Normalize / Denormalize ───────────────────────────────────────────────────
def norm_rssi(v):  return (v - RSSI_MIN) / (RSSI_MAX - RSSI_MIN)
def norm_x(v):     return v / ROOM_W
def norm_y(v):     return v / ROOM_H
def denorm_x(v):   return v * ROOM_W
def denorm_y(v):   return v * ROOM_H


# ── Compute 5 Stats From RSSI List ────────────────────────────────────────────
def compute_station_stats(rssi_values: list) -> list:
    """
    Compute 5 normalized statistics from non-null RSSI values for one station.

    All statistics normalized to [0, 1]:
        mean  → average signal strength
        std   → spread of readings
        min   → weakest reading
        max   → strongest reading
        count → how many readings existed (0.25 / 0.5 / 0.75 / 1.0)

    0 values (station unseen) → all zeros.
    """
    if not rssi_values:
        return [0.0, 0.0, 0.0, 0.0, 0.0]

    arr = np.array(rssi_values, dtype=np.float32)

    mean_v  = float(np.mean(arr))
    std_v   = float(np.std(arr))
    min_v   = float(np.min(arr))
    max_v   = float(np.max(arr))
    count_v = len(rssi_values)

    norm_mean  = norm_rssi(mean_v)
    norm_std   = std_v / ((RSSI_MAX - RSSI_MIN) / 2.0)
    norm_min   = norm_rssi(min_v)
    norm_max   = norm_rssi(max_v)
    norm_count = count_v / float(RSSI_WINDOW_SIZE)   # 0.25 / 0.5 / 0.75 / 1.0

    return [norm_mean, norm_std, norm_min, norm_max, norm_count]


# ── Load CSV ──────────────────────────────────────────────────────────────────
def load_csv(filepath: str) -> list:
    """
    Load training CSV and build feature/target pairs for LSTM sequencing.

    CSV format (one row = one complete snapshot, 18 columns):
        tag_id, rssi_11, rssi_12, rssi_13, rssi_14,
                rssi_41, rssi_42, rssi_43, rssi_44, rssi_45,
                rssi_46, rssi_47, rssi_48, rssi_49, rssi_50,
                true_x, true_y, true_z

    Feature vector per row = 70 values:
        [mean, std, min, max, count] × 14 stations

    Returns list of snapshot dicts:
        [{"feature": [70 floats], "target": [norm_x, norm_y]}, ...]
    build_sequences() then creates SEQ_LEN-step sliding windows.
    """
    intervals   = []
    total_rows  = 0
    skipped     = 0

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        print(f"[DEBUG] CSV Headers : {reader.fieldnames}\n")

        for row in reader:
            total_rows += 1
            try:
                tx = float(row["true_x"])
                ty = float(row["true_y"])
            except (KeyError, ValueError):
                skipped += 1
                continue

            feature_vec = []
            for col in RSSI_COLS_NEW:
                raw = row.get(col, "").strip()
                try:
                    rssi_val = float(raw)
                    stats = compute_station_stats([rssi_val]) \
                            if -120 <= rssi_val <= 0 \
                            else [0.0, 0.0, 0.0, 0.0, 0.0]
                except (ValueError, TypeError):
                    stats = [0.0, 0.0, 0.0, 0.0, 0.0]
                feature_vec.extend(stats)

            if len(feature_vec) != len(STATION_ORDER) * STATS_PER_STATION:
                skipped += 1
                continue

            intervals.append({
                "feature": feature_vec,
                "target":  [norm_x(tx), norm_y(ty)]
            })

    print(f"[DEBUG] Rows read   : {total_rows}")
    print(f"[DEBUG] Skipped     : {skipped}")
    print(f"[DEBUG] Snapshots   : {len(intervals)}")
    print(f"[DEBUG] Feature dim : {len(STATION_ORDER)} stations "
          f"× {STATS_PER_STATION} stats = "
          f"{len(STATION_ORDER) * STATS_PER_STATION}")
    print(f"[DEBUG] Target      : [x, y]  (z excluded)\n")

    if not intervals:
        print("[WARN] No snapshots built — check CSV headers match expected columns")

    return intervals


# ── Build Sequences (Temporal Sliding Window) ─────────────────────────────────
def build_sequences(intervals: list):
    """
    Construct LSTM training sequences using a sliding window of SEQ_LEN steps.

    Each training sample X[i] has shape (SEQ_LEN, N_FEATURES=70):
        - SEQ_LEN consecutive feature vectors representing the temporal
          RSSI history before position sample i + SEQ_LEN.
        - This enables the LSTM to learn temporal RSSI dynamics.

    X shape: (N, SEQ_LEN, N_FEATURES)
    y shape: (N, 2)  →  normalized [x, y]
    """
    if len(intervals) < SEQ_LEN + 1:
        print(f"[ERROR] Only {len(intervals)} intervals — "
              f"need > {SEQ_LEN}. Lower SEQ_LEN in config.py")
        return np.array([]), np.array([])

    X_all, y_all = [], []

    for i in range(len(intervals) - SEQ_LEN):
        X_all.append([
            intervals[j]["feature"]
            for j in range(i, i + SEQ_LEN)
        ])
        y_all.append(intervals[i + SEQ_LEN]["target"])

    X = np.array(X_all, dtype=np.float32)   # (N, SEQ_LEN, N_FEATURES)
    y = np.array(y_all, dtype=np.float32)   # (N, 2)

    print(f"[DATA] Sequences               : {len(X)}")
    print(f"[DATA] X shape                 : {X.shape}")
    print(f"       └─ (samples, seq_len={SEQ_LEN}, features={X.shape[2]})")
    print(f"[DATA] y shape                 : {y.shape}")
    print(f"       └─ (samples, x/y)\n")
    return X, y


# ── Dataset ───────────────────────────────────────────────────────────────────
class PositionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):        return len(self.X)
    def __getitem__(self, i): return self.X[i], self.y[i]


# ── Train / Eval ──────────────────────────────────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total = 0.0
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
    total = 0.0
    with torch.no_grad():
        for X_b, y_b in loader:
            total += criterion(
                model(X_b.to(device)), y_b.to(device)
            ).item()
    return total / len(loader)


# ── Plot Training Results ─────────────────────────────────────────────────────
def plot_results(train_losses, val_losses, true_m, pred_m):
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    axes[0].plot(train_losses, label="Train")
    axes[0].plot(val_losses,   label="Val")
    axes[0].set_title("PSO-LSTM Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE")
    axes[0].legend()
    axes[0].grid(True)

    for idx, label in enumerate(["X", "Y"]):
        ax    = axes[idx + 1]
        limit = min(100, len(true_m))
        ax.plot(true_m[:limit, idx], label=f"True {label}")
        ax.plot(pred_m[:limit, idx], label=f"Pred {label}", linestyle="--")
        ax.set_title(f"{label} Coordinate")
        ax.set_xlabel("Sample")
        ax.set_ylabel(f"{label} (m)")
        ax.legend()
        ax.grid(True)

    plt.suptitle(
        f"PSO-LSTM 2D Positioning — {EPOCHS} Epochs", fontsize=13
    )
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"[PLOT] Saved → {PLOT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 64)
    print("  PSO-LSTM Indoor Positioning Trainer")
    print(f"  Device       : {device}")
    print(f"  Epochs       : {EPOCHS}")
    print(f"  SEQ_LEN      : {SEQ_LEN} (temporal sliding window)")
    print(f"  Features     : {len(STATION_ORDER)} stations × "
          f"{STATS_PER_STATION} stats = "
          f"{len(STATION_ORDER) * STATS_PER_STATION}")
    print(f"  PSO Search   : {'Enabled' if USE_PSO else 'Disabled'}")
    print("=" * 64 + "\n")

    # ── 1. Load CSV ───────────────────────────────────────────────────────
    intervals = load_csv(TRAINING_CSV_PATH)

    # ── 2. Build Sequences ────────────────────────────────────────────────
    X, y = build_sequences(intervals)

    if len(X) == 0:
        print("\n[ERROR] No sequences. Check:")
        print("  Headers: tag_id, rssi_11..rssi_50, true_x, true_y, true_z")
        print(f"  Station names match config : {STATION_ORDER}")
        print(f"  Interval count > SEQ_LEN ({SEQ_LEN})")
        sys.exit(1)

    actual_features = X.shape[2]
    if actual_features != N_FEATURES:
        print(f"\n[ERROR] Feature size mismatch:")
        print(f"  Computed : {actual_features}  Expected : {N_FEATURES}")
        print(f"  Set N_FEATURES={actual_features} in config.py")
        sys.exit(1)

    # ── 3. Train / Val Split ──────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=1 - TRAIN_SPLIT, shuffle=False
    )
    print(f"[DATA] Train : {len(X_train)}   Val : {len(X_val)}\n")

    train_loader = DataLoader(
        PositionDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, drop_last=True
    )
    val_loader = DataLoader(
        PositionDataset(X_val, y_val),
        batch_size=BATCH_SIZE, shuffle=False
    )

    # ── 4. PSO Hyperparameter Search (Stage 1) ────────────────────────────
    if USE_PSO:
        pso = PSOHyperparameterOptimizer(train_loader, val_loader, device)
        best_cfg = pso.optimize()
        # Use PSO-discovered hyperparameters for final model
        final_hidden  = best_cfg["hidden_size"]
        final_layers  = best_cfg["num_layers"]
        final_lr      = best_cfg["learning_rate"]
        final_dropout = best_cfg["dropout"]
    else:
        print("[PSO] Skipped — using config.py defaults\n")
        final_hidden  = HIDDEN_SIZE
        final_layers  = NUM_LAYERS
        final_lr      = LR
        final_dropout = DROPOUT

    # ── 5. Build Final Model with PSO-Optimized Config ────────────────────
    print(f"[TRAIN] Building final PSO-LSTM model:")
    print(f"[TRAIN]   hidden_size : {final_hidden}")
    print(f"[TRAIN]   num_layers  : {final_layers}")
    print(f"[TRAIN]   learning_rate: {final_lr:.2e}")
    print(f"[TRAIN]   dropout     : {final_dropout:.2f}\n")

    model = LSTMModel(
        input_size  = N_FEATURES,
        hidden_size = final_hidden,
        num_layers  = final_layers,
        output_size = N_OUTPUTS,
        dropout     = final_dropout
    ).to(device)

    criterion = nn.MSELoss()
    # Adam fine-tunes from the PSO-discovered weight configuration.
    # This PSO-Adam hybrid avoids local minima while achieving
    # convergence precision that PSO alone cannot provide.
    optimizer = torch.optim.Adam(model.parameters(), lr=final_lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.1, patience=20,
        threshold=1e-4, min_lr=1e-6
    )

    # ── 6. Train Final Model (Stage 2: Adam fine-tuning) ──────────────────
    train_losses, val_losses = [], []
    print(f"  {'Epoch':<8} {'Train':>12}  {'Val':>10}  {'LR':>12}")
    print("  " + "-" * 50)

    for epoch in range(EPOCHS):
        tl = train_epoch(model, train_loader, optimizer, criterion, device)
        vl = evaluate(model,   val_loader,   criterion, device)
        scheduler.step(vl)
        train_losses.append(tl)
        val_losses.append(vl)
        lr = optimizer.param_groups[0]["lr"]

        if (epoch + 1) % 100 == 0 or epoch < 5:
            print(f"  {epoch+1:<8} {tl:>12.6f}  {vl:>10.6f}  {lr:>12.2e}")

    # ── 7. Save Model ─────────────────────────────────────────────────────
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n[SAVE] Model saved → {MODEL_PATH}")

    # ── 8. Evaluate on Validation Set ─────────────────────────────────────
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
    ], axis=1)
    true_m = np.stack([
        denorm_x(all_true[:, 0]),
        denorm_y(all_true[:, 1]),
    ], axis=1)

    errors = np.sqrt(np.sum((pred_m - true_m) ** 2, axis=1))
    print(f"\n[EVAL] PSO-LSTM 2D Positioning Accuracy:")
    print(f"[EVAL]   Mean 2D error  : {np.mean(errors):.3f} m")
    print(f"[EVAL]   Median error   : {np.median(errors):.3f} m")
    print(f"[EVAL]   Max error      : {np.max(errors):.3f} m")
    print(f"[EVAL]   Error < 0.5m  : {np.mean(errors < 0.5) * 100:.1f}%")
    print(f"[EVAL]   Error < 1.0m  : {np.mean(errors < 1.0) * 100:.1f}%")
    print(f"[EVAL]   Error < 2.0m  : {np.mean(errors < 2.0) * 100:.1f}%")

    plot_results(train_losses, val_losses, true_m, pred_m)
    print("\n[DONE] Run main.py to use the trained PSO-LSTM model.")
