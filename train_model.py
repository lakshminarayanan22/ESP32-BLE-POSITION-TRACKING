import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import matplotlib
matplotlib.use('Agg')  # non-interactive backend for saving plots
import matplotlib.pyplot as plt

from LSTM    import LSTMModel
from config  import (
    SEQ_LEN, N_FEATURES, HIDDEN_SIZE,
    NUM_LAYERS, DROPOUT, MODEL_PATH,
    RSSI_MIN, RSSI_MAX
)

# ── Training Config ───────────────────────────────────
EPOCHS      = 40
BATCH_SIZE  = 32
LR          = 0.001
TRAIN_SPLIT = 0.8
JSON_FILE = "/Users/lnarayanansk/Documents/Doodil2r/beacon_data.json"
PLOT_PATH   = "/Users/lnarayanansk/Documents/Doodil2r/training_results.png"

# ── Normalize / Denormalize ───────────────────────────
def normalize(rssi: float) -> float:
    return (rssi - RSSI_MIN) / (RSSI_MAX - RSSI_MIN)

def denormalize(val: float) -> float:
    return val * (RSSI_MAX - RSSI_MIN) + RSSI_MIN

# ── Load Dataset ──────────────────────────────────────
def load_dataset(filepath: str) -> dict:
    with open(filepath, "r") as f:
        payloads = json.load(f)

    tag_series = {}
    for payload in payloads:
        for tag in payload.get("tags", []):
            tag_id = tag["tagId"]
            rssi   = tag["rssi"]
            if tag_id not in tag_series:
                tag_series[tag_id] = []
            tag_series[tag_id].append(rssi)

    print(f"[DATA] Loaded   : {len(payloads)} payloads")
    print(f"[DATA] Tags     : {len(tag_series)}")
    print()
    print(f"  {'TAG ID':<22}  {'READINGS':>8}  {'MIN':>5}  {'MAX':>5}  {'AVG':>7}")
    print("  " + "-" * 55)
    for tid, series in tag_series.items():
        print(
            f"  {tid:<22}  {len(series):>8}  "
            f"{min(series):>5}  {max(series):>5}  "
            f"{sum(series)/len(series):>7.1f}"
        )
    return tag_series

# ── Build Sequences ───────────────────────────────────
def build_sequences(tag_series: dict):
    X_all, y_all = [], []

    for tag_id, series in tag_series.items():
        if len(series) < SEQ_LEN + 1:
            print(f"[WARN] {tag_id} too short ({len(series)} readings), skipping")
            continue

        norm = [normalize(r) for r in series]

        for i in range(len(norm) - SEQ_LEN):
            X_all.append(norm[i : i + SEQ_LEN])
            y_all.append(norm[i + SEQ_LEN])

    X = np.array(X_all, dtype=np.float32).reshape(-1, SEQ_LEN, 1)
    y = np.array(y_all, dtype=np.float32).reshape(-1, 1)

    print(f"\n[DATA] Sequences: {len(X)}")
    print(f"[DATA] X shape  : {X.shape}")
    print(f"[DATA] y shape  : {y.shape}")
    return X, y

# ── Dataset ───────────────────────────────────────────
class RSSIDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# ── Train One Epoch ───────────────────────────────────
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)
        optimizer.zero_grad()
        output = model(X_batch)
        loss   = criterion(output, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

# ── Evaluate ──────────────────────────────────────────
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            output  = model(X_batch)
            total_loss += criterion(output, y_batch).item()
    return total_loss / len(loader)

# ── Plot ──────────────────────────────────────────────
def plot_results(train_losses, val_losses, y_true, y_pred):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss curve
    axes[0].plot(train_losses, label="Train Loss", linewidth=1.5)
    axes[0].plot(val_losses,   label="Val Loss",   linewidth=1.5)
    axes[0].axvline(x=EPOCHS - 1, color="red",
                    linestyle="--", label=f"Save point (epoch {EPOCHS})")
    axes[0].set_title("Training Loss Curve")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss")
    axes[0].legend()
    axes[0].grid(True)

    # Predicted vs Actual
    axes[1].plot(y_true[:100], label="Actual RSSI",    linewidth=1.5)
    axes[1].plot(y_pred[:100], label="Predicted RSSI",
                 linewidth=1.5, linestyle="--")
    axes[1].set_title("Actual vs Predicted RSSI (first 100 samples)")
    axes[1].set_xlabel("Sample")
    axes[1].set_ylabel("RSSI (dBm)")
    axes[1].legend()
    axes[1].grid(True)

    plt.suptitle(f"LSTM Training — Saved at Epoch {EPOCHS}", fontsize=13)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"[PLOT] Saved → {PLOT_PATH}")

# ── Main ──────────────────────────────────────────────
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 60)
    print("  LSTM RSSI Trainer")
    print(f"  Device     : {device}")
    print(f"  Epochs     : {EPOCHS}  (model saved after epoch {EPOCHS})")
    print(f"  SEQ_LEN    : {SEQ_LEN}")
    print(f"  Batch size : {BATCH_SIZE}")
    print(f"  LR         : {LR}")
    print("=" * 60 + "\n")

    # 1. Load
    tag_series = load_dataset(JSON_FILE)

    # 2. Sequences
    X, y = build_sequences(tag_series)

    # 3. Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=1 - TRAIN_SPLIT, shuffle=False
    )
    print(f"\n[DATA] Train samples : {len(X_train)}")
    print(f"[DATA] Val samples   : {len(X_val)}\n")

    train_loader = DataLoader(
        RSSIDataset(X_train, y_train),
        batch_size = BATCH_SIZE,
        shuffle    = True,
        drop_last  = True       # prevents batch_size=1 edge case
    )
    val_loader = DataLoader(
        RSSIDataset(X_val, y_val),
        batch_size = BATCH_SIZE,
        shuffle    = False,
        drop_last  = False
    )

    # 4. Model
    model = LSTMModel(
        input_size  = N_FEATURES,
        hidden_size = HIDDEN_SIZE,
        num_layers  = NUM_LAYERS,
        output_size = N_FEATURES,
        dropout     = DROPOUT
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )

    # 5. Train all 40 epochs — no early stopping, no best-model saving
    train_losses, val_losses = [], []

    print(f"  {'Epoch':<8} {'Train Loss':>12}  {'Val Loss':>10}  {'LR':>10}")
    print("  " + "-" * 48)

    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss   = evaluate(model,   val_loader,   criterion, device)

        scheduler.step(val_loss)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(f"  {epoch+1:<8} {train_loss:>12.6f}  {val_loss:>10.6f}  {current_lr:>10.6f}")

    # 6. Save after exactly 40 epochs
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"\n[SAVE] Model saved after epoch {EPOCHS} → {MODEL_PATH}")

    # 7. Final evaluation
    model.eval()
    all_preds, all_true = [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            preds = model(X_batch.to(device)).cpu().numpy()
            all_preds.extend(preds.flatten())
            all_true.extend(y_batch.numpy().flatten())

    all_preds_db = [denormalize(p) for p in all_preds]
    all_true_db  = [denormalize(t) for t in all_true]

    mae  = np.mean(np.abs(np.array(all_preds_db) - np.array(all_true_db)))
    rmse = np.sqrt(np.mean((np.array(all_preds_db) - np.array(all_true_db))**2))

    print(f"\n[EVAL] MAE  : {mae:.2f} dBm")
    print(f"[EVAL] RMSE : {rmse:.2f} dBm")
    print(f"[EVAL] Final train loss : {train_losses[-1]:.6f}")
    print(f"[EVAL] Final val loss   : {val_losses[-1]:.6f}")

    # 8. Plot
    plot_results(train_losses, val_losses, all_true_db, all_preds_db)

    print("\n[DONE] Training complete. Run main.py to use the model.")