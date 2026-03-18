"""
SVM (RBF Kernel) and Random Forest Training with GridSearchCV
==============================================================

Trains two positioning models on the same CSV training data used by
the PSO-LSTM pipeline. Both models use a single 70-dim feature snapshot
(no temporal window) — instant predictions without warm-up delay.

GridSearchCV Strategy
---------------------
SVM (RBF kernel) — MultiOutputRegressor wrapper:
    Trains one SVR per output dimension (x, y, z).
    Exhaustive grid over C, gamma, epsilon.
    Large C  → narrow margin, fits training data closely.
    Small C  → wide margin, more regularisation.
    gamma    → controls RBF width (complexity of decision boundary).
    epsilon  → insensitive tube width around predictions.

Random Forest — native multi-output:
    Predicts (x, y, z) in a single call via ensemble voting.
    Grid covers n_estimators, depth, split criteria, feature sampling,
    impurity threshold, and bootstrap strategy.

Both grids use n_jobs=-1 (all CPU cores) and cv=5-fold cross-validation.
Best models are saved as .pkl files loadable by svm_model.py / rf_model.py.

Usage:
    python train_svm_rf.py
"""

import csv
import sys
import time
import joblib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from collections import defaultdict

from sklearn.svm               import SVR
from sklearn.multioutput       import MultiOutputRegressor
from sklearn.ensemble          import RandomForestRegressor
from sklearn.model_selection   import GridSearchCV, train_test_split
from sklearn.metrics           import mean_squared_error
from sklearn.pipeline          import Pipeline
from sklearn.preprocessing     import StandardScaler

from config import (
    N_FEATURES, RSSI_MIN, RSSI_MAX,
    ROOM_W, ROOM_H, ROOM_Z,
    STATION_POSITIONS,
    SVM_MODEL_PATH, RF_MODEL_PATH,
)

# ── Paths & Hyper-params ──────────────────────────────────────────────────────
TRAINING_CSV_PATH = "tag_training_data.csv"
PLOT_PATH         = "svm_rf_results.png"
TRAIN_SPLIT       = 0.8
CV_FOLDS          = 5       # cross-validation folds for GridSearchCV

STATION_ORDER     = list(STATION_POSITIONS.keys())
RSSI_COLS         = ["rssi_value_1", "rssi_value_2",
                      "rssi_value_3", "rssi_value_4"]
STATS_PER_STATION = 5

# ── Normalize / Denormalize ───────────────────────────────────────────────────
def norm_rssi(v):  return (v - RSSI_MIN) / (RSSI_MAX - RSSI_MIN)
def norm_x(v):     return v / ROOM_W
def norm_y(v):     return v / ROOM_H
def norm_z(v):     return v / ROOM_Z
def denorm_x(v):   return v * ROOM_W
def denorm_y(v):   return v * ROOM_H
def denorm_z(v):   return v * ROOM_Z


# ── CSV Helpers ───────────────────────────────────────────────────────────────
def extract_rssi_values(row: dict) -> list:
    values = []
    for col in RSSI_COLS:
        raw = row.get(col, "").strip()
        if raw:
            try:
                values.append(float(raw))
            except ValueError:
                pass
    return values


def compute_station_stats(rssi_values: list) -> list:
    """5 normalized stats from non-null RSSI readings for one station."""
    if not rssi_values:
        return [0.0, 0.0, 0.0, 0.0, 0.0]
    arr        = np.array(rssi_values, dtype=np.float32)
    mean_v     = float(np.mean(arr))
    std_v      = float(np.std(arr))
    min_v      = float(np.min(arr))
    max_v      = float(np.max(arr))
    count_v    = len(rssi_values)
    norm_mean  = norm_rssi(mean_v)
    norm_std   = std_v / ((RSSI_MAX - RSSI_MIN) / 2.0)
    norm_min   = norm_rssi(min_v)
    norm_max   = norm_rssi(max_v)
    norm_count = count_v / len(RSSI_COLS)
    return [norm_mean, norm_std, norm_min, norm_max, norm_count]


# ── Load CSV → flat (X, y) arrays ─────────────────────────────────────────────
def load_dataset(filepath: str):
    """
    Load training CSV and return:
        X : np.ndarray  (N, N_FEATURES)  — per-interval feature vectors
        y : np.ndarray  (N, 3)           — normalised (x, y, z) targets

    Unlike the LSTM pipeline, no temporal windowing is applied here.
    SVM and RF predict from a single-timestep snapshot.
    """
    raw    = defaultdict(lambda: {sid: [] for sid in STATION_ORDER})
    coords = {}

    total_rows = skipped_sta = skipped_rssi = 0
    found_sta  = set()

    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        print(f"[DATA] CSV headers: {reader.fieldnames}\n")

        for row in reader:
            total_rows += 1
            start   = row.get("start_time", "").strip()
            end     = row.get("end_time",   "").strip()
            station = row.get("station",    "").upper().strip()

            if not start or not end:
                continue

            if station not in STATION_ORDER:
                skipped_sta += 1
                found_sta.add(station)
                continue

            rssi_values = extract_rssi_values(row)
            if not rssi_values:
                skipped_rssi += 1
                continue

            raw[(start, end)][station].extend(rssi_values)

            key = (start, end)
            if key not in coords:
                try:
                    coords[key] = (
                        float(row["true_x"]),
                        float(row["true_y"]),
                        float(row["true_z"]),
                    )
                except (KeyError, ValueError):
                    pass

    print(f"[DATA] Rows read          : {total_rows}")
    print(f"[DATA] Skipped (station)  : {skipped_sta}")
    print(f"[DATA] Skipped (no RSSI)  : {skipped_rssi}")
    if found_sta - set(STATION_ORDER):
        print(f"[DATA] Unknown stations   : {found_sta - set(STATION_ORDER)}")
    print(f"[DATA] Valid intervals    : {len(raw)}\n")

    X_list, y_list = [], []

    for (start, end), stn_readings in sorted(raw.items()):
        key = (start, end)
        if key not in coords:
            continue

        feature_vec = []
        for sid in STATION_ORDER:
            feature_vec.extend(compute_station_stats(stn_readings[sid]))

        assert len(feature_vec) == len(STATION_ORDER) * STATS_PER_STATION, \
            f"Feature size {len(feature_vec)} ≠ expected {N_FEATURES}"

        tx, ty, tz = coords[key]
        X_list.append(feature_vec)
        y_list.append([norm_x(tx), norm_y(ty), norm_z(tz)])

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list, dtype=np.float64)

    print(f"[DATA] Dataset shape : X={X.shape}  y={y.shape}")
    print(f"[DATA] Feature dims  : {len(STATION_ORDER)} stations "
          f"× {STATS_PER_STATION} stats = {X.shape[1]}")
    print(f"[DATA] Target range  : x∈[{y[:,0].min():.3f},{y[:,0].max():.3f}]  "
          f"y∈[{y[:,1].min():.3f},{y[:,1].max():.3f}]  "
          f"z∈[{y[:,2].min():.3f},{y[:,2].max():.3f}]\n")
    return X, y


# ── GridSearchCV — SVM (RBF Kernel) ──────────────────────────────────────────
def train_svm(X_train, y_train, X_val, y_val):
    """
    Train MultiOutputRegressor(SVR) with RBF kernel via exhaustive GridSearchCV.

    Pipeline: StandardScaler → SVR(kernel='rbf')
    SVMs are sensitive to feature scale, so StandardScaler is mandatory.

    Grid (8 × 8 × 7 = 448 combinations × CV_FOLDS × 3 outputs):
        C       : regularisation — large C → tighter fit, smaller C → smoother
        gamma   : RBF bandwidth — 'scale'=1/(n_feat·var), 'auto'=1/n_feat
        epsilon : ε-tube width — predictions within tube have zero loss

    n_jobs=-1 uses all available CPU cores in parallel.
    """
    print("=" * 64)
    print("  SVM (RBF Kernel) — GridSearchCV")
    print(f"  CV folds : {CV_FOLDS}   Train samples : {len(X_train)}")
    print("=" * 64)

    svm_param_grid = {
        "estimator__svr__C": [
            0.01, 0.1, 0.5, 1, 5, 10, 50, 100, 500, 1000, 5000
        ],
        "estimator__svr__gamma": [
            "scale", "auto", 1e-5, 1e-4, 1e-3, 0.01, 0.05, 0.1, 0.5, 1.0
        ],
        "estimator__svr__epsilon": [
            0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5
        ],
    }

    total_combos = 1
    for v in svm_param_grid.values():
        total_combos *= len(v)
    print(f"[SVM] Grid size      : {total_combos} combinations "
          f"× {CV_FOLDS} folds × 3 outputs = "
          f"{total_combos * CV_FOLDS * 3} total fits")
    print(f"[SVM] Param grid:\n"
          f"       C       : {svm_param_grid['estimator__svr__C']}\n"
          f"       gamma   : {svm_param_grid['estimator__svr__gamma']}\n"
          f"       epsilon : {svm_param_grid['estimator__svr__epsilon']}\n")

    # Pipeline: scale → SVR
    # StandardScaler is critical for SVMs — RSSI features already in [0,1]
    # but scaler handles residual variance differences between stations.
    svr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("svr",    SVR(kernel="rbf", max_iter=5000))
    ])
    multi_svr = MultiOutputRegressor(svr_pipeline, n_jobs=-1)

    grid_svm = GridSearchCV(
        multi_svr,
        svm_param_grid,
        cv           = CV_FOLDS,
        scoring      = "neg_mean_squared_error",
        n_jobs       = -1,
        verbose      = 2,
        refit        = True,
        return_train_score = True,
    )

    t0 = time.time()
    grid_svm.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n[SVM] Search complete in {elapsed:.1f}s")
    print(f"[SVM] Best params   : {grid_svm.best_params_}")
    print(f"[SVM] Best CV MSE   : {-grid_svm.best_score_:.6f}")

    # Evaluate on validation set
    y_pred    = grid_svm.predict(X_val)
    pred_m    = np.stack([denorm_x(y_pred[:,0]),
                          denorm_y(y_pred[:,1]),
                          denorm_z(y_pred[:,2])], axis=1)
    true_m    = np.stack([denorm_x(y_val[:,0]),
                          denorm_y(y_val[:,1]),
                          denorm_z(y_val[:,2])], axis=1)
    errors_3d = np.sqrt(np.sum((pred_m - true_m)**2, axis=1))

    print(f"\n[SVM] Validation results (denormalized):")
    print(f"       Mean 3D error  : {np.mean(errors_3d):.3f} m")
    print(f"       Median error   : {np.median(errors_3d):.3f} m")
    print(f"       Max error      : {np.max(errors_3d):.3f} m")
    print(f"       Error < 0.5 m  : {np.mean(errors_3d < 0.5)*100:.1f}%")
    print(f"       Error < 1.0 m  : {np.mean(errors_3d < 1.0)*100:.1f}%")
    print(f"       Error < 2.0 m  : {np.mean(errors_3d < 2.0)*100:.1f}%")

    joblib.dump(grid_svm.best_estimator_, SVM_MODEL_PATH)
    print(f"\n[SVM] Model saved → {SVM_MODEL_PATH}")

    return grid_svm, pred_m, true_m, errors_3d


# ── GridSearchCV — Random Forest ─────────────────────────────────────────────
def train_rf(X_train, y_train, X_val, y_val):
    """
    Train RandomForestRegressor with exhaustive GridSearchCV.

    Random Forest natively supports multi-output regression — no wrapper needed.
    The ensemble variance reduction is especially effective for noisy RSSI data.

    Grid (5 × 6 × 4 × 5 × 5 × 3 × 2 = 18,000 combinations):
        n_estimators         : number of trees
        max_depth            : max tree depth (None = fully grown)
        min_samples_split    : minimum samples to split a node
        min_samples_leaf     : minimum samples at a leaf
        max_features         : feature subset per split
        min_impurity_decrease: split only if impurity drops by this much
        bootstrap            : whether to use bootstrap sampling

    Uses n_jobs=-1 for parallel tree fitting.
    """
    print("\n" + "=" * 64)
    print("  Random Forest — GridSearchCV")
    print(f"  CV folds : {CV_FOLDS}   Train samples : {len(X_train)}")
    print("=" * 64)

    rf_param_grid = {
        "n_estimators":          [100, 200, 300, 500, 800],
        "max_depth":             [None, 10, 20, 30, 40, 50],
        "min_samples_split":     [2, 5, 10, 20],
        "min_samples_leaf":      [1, 2, 4, 8, 16],
        "max_features":          ["sqrt", "log2", 0.3, 0.5, 0.7],
        "min_impurity_decrease": [0.0, 0.0005, 0.001],
        "bootstrap":             [True, False],
    }

    total_combos = 1
    for v in rf_param_grid.values():
        total_combos *= len(v)
    print(f"[RF]  Grid size      : {total_combos} combinations "
          f"× {CV_FOLDS} folds = {total_combos * CV_FOLDS} total fits")
    print(f"[RF]  Param grid:\n"
          f"       n_estimators         : {rf_param_grid['n_estimators']}\n"
          f"       max_depth            : {rf_param_grid['max_depth']}\n"
          f"       min_samples_split    : {rf_param_grid['min_samples_split']}\n"
          f"       min_samples_leaf     : {rf_param_grid['min_samples_leaf']}\n"
          f"       max_features         : {rf_param_grid['max_features']}\n"
          f"       min_impurity_decrease: {rf_param_grid['min_impurity_decrease']}\n"
          f"       bootstrap            : {rf_param_grid['bootstrap']}\n")

    rf = RandomForestRegressor(random_state=42, n_jobs=-1)

    grid_rf = GridSearchCV(
        rf,
        rf_param_grid,
        cv                = CV_FOLDS,
        scoring           = "neg_mean_squared_error",
        n_jobs            = -1,
        verbose           = 2,
        refit             = True,
        return_train_score= True,
    )

    t0 = time.time()
    grid_rf.fit(X_train, y_train)
    elapsed = time.time() - t0

    print(f"\n[RF]  Search complete in {elapsed:.1f}s")
    print(f"[RF]  Best params   : {grid_rf.best_params_}")
    print(f"[RF]  Best CV MSE   : {-grid_rf.best_score_:.6f}")

    # Evaluate on validation set
    y_pred    = grid_rf.predict(X_val)
    pred_m    = np.stack([denorm_x(y_pred[:,0]),
                          denorm_y(y_pred[:,1]),
                          denorm_z(y_pred[:,2])], axis=1)
    true_m    = np.stack([denorm_x(y_val[:,0]),
                          denorm_y(y_val[:,1]),
                          denorm_z(y_val[:,2])], axis=1)
    errors_3d = np.sqrt(np.sum((pred_m - true_m)**2, axis=1))

    print(f"\n[RF]  Validation results (denormalized):")
    print(f"       Mean 3D error  : {np.mean(errors_3d):.3f} m")
    print(f"       Median error   : {np.median(errors_3d):.3f} m")
    print(f"       Max error      : {np.max(errors_3d):.3f} m")
    print(f"       Error < 0.5 m  : {np.mean(errors_3d < 0.5)*100:.1f}%")
    print(f"       Error < 1.0 m  : {np.mean(errors_3d < 1.0)*100:.1f}%")
    print(f"       Error < 2.0 m  : {np.mean(errors_3d < 2.0)*100:.1f}%")

    # Feature importance (RF-specific)
    importances = grid_rf.best_estimator_.feature_importances_
    top_idx = np.argsort(importances)[::-1][:10]
    print(f"\n[RF]  Top-10 important features:")
    station_names = []
    for sid in STATION_ORDER:
        station_names += [f"{sid}_mean", f"{sid}_std",
                          f"{sid}_min",  f"{sid}_max",
                          f"{sid}_cnt"]
    for rank, fi in enumerate(top_idx):
        fname = station_names[fi] if fi < len(station_names) else f"feat_{fi}"
        print(f"       #{rank+1:2d}  {fname:<22}  importance={importances[fi]:.4f}")

    joblib.dump(grid_rf.best_estimator_, RF_MODEL_PATH)
    print(f"\n[RF]  Model saved → {RF_MODEL_PATH}")

    return grid_rf, pred_m, true_m, errors_3d


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_comparison(svm_pred, svm_true, svm_err,
                    rf_pred,  rf_true,  rf_err):
    """Side-by-side scatter plots and CDF error curves for SVM vs RF."""
    fig, axes = plt.subplots(2, 4, figsize=(24, 10))
    limit = min(200, len(svm_pred))

    models = [("SVM (RBF)", svm_pred[:limit], svm_true[:limit], svm_err),
              ("RandomForest", rf_pred[:limit], rf_true[:limit], rf_err)]

    for row, (name, pred, true, err) in enumerate(models):
        for col, label in enumerate(["X", "Y", "Z"]):
            ax = axes[row][col]
            ax.scatter(true[:, col], pred[:, col],
                       alpha=0.5, s=15, c="steelblue" if row == 0 else "darkorange")
            mn = min(true[:, col].min(), pred[:, col].min())
            mx = max(true[:, col].max(), pred[:, col].max())
            ax.plot([mn, mx], [mn, mx], "r--", linewidth=1.5, label="Ideal")
            ax.set_xlabel(f"True {label} (norm)")
            ax.set_ylabel(f"Pred {label} (norm)")
            ax.set_title(f"{name} — {label} axis")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

        # CDF of 3D errors
        ax = axes[row][3]
        sorted_err = np.sort(err)
        cdf        = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        color      = "steelblue" if row == 0 else "darkorange"
        ax.plot(sorted_err, cdf, color=color, linewidth=2)
        ax.axvline(np.mean(err),   color="red",   linestyle="--",
                   label=f"Mean={np.mean(err):.2f}m")
        ax.axvline(np.median(err), color="green", linestyle="--",
                   label=f"Median={np.median(err):.2f}m")
        ax.set_xlabel("3D Position Error (m)")
        ax.set_ylabel("CDF")
        ax.set_title(f"{name} — Error CDF")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle("SVM vs Random Forest — Indoor Positioning Accuracy", fontsize=14)
    plt.tight_layout()
    plt.savefig(PLOT_PATH, dpi=150)
    print(f"\n[PLOT] Saved → {PLOT_PATH}")


# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 64)
    print("  SVM (RBF) + Random Forest Trainer — GridSearchCV")
    print(f"  CSV            : {TRAINING_CSV_PATH}")
    print(f"  Stations       : {len(STATION_ORDER)}  Features : {N_FEATURES}")
    print(f"  CV folds       : {CV_FOLDS}")
    print(f"  Train/Val split: {int(TRAIN_SPLIT*100)}/{int((1-TRAIN_SPLIT)*100)}")
    print("=" * 64 + "\n")

    # ── 1. Load Data ──────────────────────────────────────────────────────
    X, y = load_dataset(TRAINING_CSV_PATH)
    if len(X) == 0:
        print("[ERROR] No data loaded. Check CSV path and station names.")
        sys.exit(1)

    actual_features = X.shape[1]
    if actual_features != N_FEATURES:
        print(f"[ERROR] Feature mismatch: got {actual_features}, "
              f"expected {N_FEATURES} ({len(STATION_ORDER)} stations × 5 stats)")
        print("        Update N_FEATURES in config.py to match.")
        sys.exit(1)

    # ── 2. Train / Val Split ──────────────────────────────────────────────
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=1 - TRAIN_SPLIT, shuffle=False
    )
    print(f"[DATA] Train : {len(X_train)}   Val : {len(X_val)}\n")

    # ── 3. Train SVM ──────────────────────────────────────────────────────
    grid_svm, svm_pred, svm_true, svm_err = train_svm(
        X_train, y_train, X_val, y_val
    )

    # ── 4. Train Random Forest ────────────────────────────────────────────
    grid_rf, rf_pred, rf_true, rf_err = train_rf(
        X_train, y_train, X_val, y_val
    )

    # ── 5. Summary Comparison ─────────────────────────────────────────────
    print("\n" + "=" * 64)
    print("  FINAL COMPARISON — Validation Set")
    print("=" * 64)
    print(f"  {'Metric':<22}  {'SVM (RBF)':>12}  {'Random Forest':>14}")
    print("  " + "-" * 52)
    metrics = [
        ("Mean 3D error (m)",  np.mean(svm_err),               np.mean(rf_err)),
        ("Median error (m)",   np.median(svm_err),             np.median(rf_err)),
        ("Max error (m)",      np.max(svm_err),                np.max(rf_err)),
        ("% error < 0.5 m",   np.mean(svm_err < 0.5) * 100,  np.mean(rf_err < 0.5) * 100),
        ("% error < 1.0 m",   np.mean(svm_err < 1.0) * 100,  np.mean(rf_err < 1.0) * 100),
        ("% error < 2.0 m",   np.mean(svm_err < 2.0) * 100,  np.mean(rf_err < 2.0) * 100),
    ]
    for label, sv, rf in metrics:
        print(f"  {label:<22}  {sv:>12.3f}  {rf:>14.3f}")
    print("=" * 64)

    # ── 6. Plot ───────────────────────────────────────────────────────────
    plot_comparison(svm_pred, svm_true, svm_err,
                    rf_pred,  rf_true,  rf_err)

    print("\n[DONE] Models saved:")
    print(f"       SVM → {SVM_MODEL_PATH}")
    print(f"       RF  → {RF_MODEL_PATH}")
    print("       Run main.py to see all three models live.")
