"""
PSO Hyperparameter Optimizer for LSTM Indoor Positioning Model
==============================================================

Implements Particle Swarm Optimization (PSO) to discover the optimal LSTM
hyperparameter configuration before final Adam fine-tuning.

Mathematical Framework:
    Each particle encodes a hyperparameter configuration vector:
        x_i = [hidden_size, num_layers, log10(lr), dropout]

    Velocity and position updates (standard PSO equations):
        v(k+1) = w * v(k) + c1*r1*(pbest - x) + c2*r2*(gbest - x)
        x(k+1) = x(k) + v(k+1)

    where:
        w  = inertia weight (linearly decayed from W_INIT to W_MIN)
        c1 = cognitive coefficient (attraction toward personal best)
        c2 = social coefficient (attraction toward global best)
        r1, r2 = random numbers in [0, 1] per iteration

    Fitness function: validation MSE after PSO_EVAL_EPOCHS of Adam training.
    Lower MSE = better particle.

Optimal Hybrid: PSO-Adam
    PSO finds a mathematically superior weight initialization that avoids:
        - Local minima (feedforward BP limitation)
        - Vanishing gradients (LSTM training instability)
    Adam optimizer then fine-tunes from this starting point for convergence
    precision that PSO alone cannot achieve.

Usage:
    from pso_optimizer import PSOHyperparameterOptimizer
    optimizer = PSOHyperparameterOptimizer(train_loader, val_loader, device)
    best_config = optimizer.optimize()
    # best_config = {hidden_size, num_layers, learning_rate, dropout, val_loss}
"""

import numpy as np
import torch
import torch.nn as nn

from LSTM import LSTMModel
from config import (
    N_FEATURES, N_OUTPUTS,
    PSO_SWARM_SIZE, PSO_MAX_ITER, PSO_C1, PSO_C2,
    PSO_W_INIT, PSO_W_MIN, PSO_EVAL_EPOCHS
)

# ── Search Space Boundaries ───────────────────────────────────────────────────
# Dimensions: [hidden_size, num_layers, log10(lr), dropout]
#   hidden_size : integer in [32, 256]
#   num_layers  : integer in [1, 3]
#   log10(lr)   : float in [-4.0, -1.0]  →  lr in [1e-4, 0.1]
#   dropout     : float in [0.0, 0.5]
LOWER = np.array([32.0,  1.0, -4.0, 0.0])
UPPER = np.array([256.0, 3.0, -1.0, 0.5])


def _decode_particle(pos: np.ndarray) -> tuple:
    """
    Convert continuous particle position vector to discrete hyperparameters.

    Returns: (hidden_size, num_layers, learning_rate, dropout)
    """
    hidden_size = int(np.clip(round(pos[0]), 32, 256))
    num_layers  = int(np.clip(round(pos[1]), 1, 3))
    lr          = float(10 ** np.clip(pos[2], -4.0, -1.0))
    dropout     = float(np.clip(pos[3], 0.0, 0.5))
    return hidden_size, num_layers, lr, dropout


def _evaluate_particle(pos: np.ndarray, train_loader, val_loader, device) -> float:
    """
    Build an LSTM with the hyperparameters encoded in `pos`, train for
    PSO_EVAL_EPOCHS epochs using Adam, then return validation MSE.

    This is the PSO fitness function — lower is better.
    """
    hidden_size, num_layers, lr, dropout = _decode_particle(pos)

    model = LSTMModel(
        input_size  = N_FEATURES,
        hidden_size = hidden_size,
        num_layers  = num_layers,
        output_size = N_OUTPUTS,
        dropout     = dropout
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    for _ in range(PSO_EVAL_EPOCHS):
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for X_b, y_b in val_loader:
            total_loss += criterion(
                model(X_b.to(device)), y_b.to(device)
            ).item()

    val_loss = total_loss / max(len(val_loader), 1)
    del model
    return val_loss


class PSOHyperparameterOptimizer:
    """
    PSO-based LSTM hyperparameter optimizer.

    Searches for the best combination of:
        hidden_size, num_layers, learning_rate, dropout

    by treating each combination as a particle in a D=4 dimensional
    search space and evolving the swarm toward lower validation MSE.

    After optimization, the best config is used to build the final LSTM,
    which is then trained for the full EPOCHS using Adam (PSO-Adam hybrid).
    """

    def __init__(self, train_loader, val_loader, device=None):
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.device       = device or torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.dim = len(LOWER)   # 4-dimensional search space

    def optimize(self) -> dict:
        """
        Run PSO to find optimal hyperparameters.

        Returns dict with keys:
            hidden_size, num_layers, learning_rate, dropout, val_loss
        """
        print("[PSO] ── Hyperparameter Search ──────────────────────────")
        print(f"[PSO] Swarm size  : {PSO_SWARM_SIZE} particles")
        print(f"[PSO] Max iter    : {PSO_MAX_ITER} iterations")
        print(f"[PSO] Eval epochs : {PSO_EVAL_EPOCHS} per particle")
        print(f"[PSO] Search space:")
        print(f"[PSO]   hidden_size : [32, 256]")
        print(f"[PSO]   num_layers  : [1, 3]")
        print(f"[PSO]   lr          : [1e-4, 0.1]  (log scale)")
        print(f"[PSO]   dropout     : [0.0, 0.5]")
        print(f"[PSO] Inertia      : {PSO_W_INIT} → {PSO_W_MIN} (linear decay)")
        print(f"[PSO] c1={PSO_C1} (cognitive)  c2={PSO_C2} (social)\n")

        # ── Initialize Swarm ──────────────────────────────────────────────
        positions  = np.random.uniform(LOWER, UPPER, (PSO_SWARM_SIZE, self.dim))
        velocities = np.zeros((PSO_SWARM_SIZE, self.dim))
        pbest_pos  = positions.copy()
        pbest_fit  = np.full(PSO_SWARM_SIZE, np.inf)
        gbest_pos  = positions[0].copy()
        gbest_fit  = np.inf

        # ── PSO Main Loop ─────────────────────────────────────────────────
        for iteration in range(PSO_MAX_ITER):
            # Linear inertia decay: broad exploration → precise exploitation
            w = PSO_W_INIT - (PSO_W_INIT - PSO_W_MIN) * (iteration / PSO_MAX_ITER)

            for i in range(PSO_SWARM_SIZE):
                fitness = _evaluate_particle(
                    positions[i], self.train_loader, self.val_loader, self.device
                )

                # Update personal best
                if fitness < pbest_fit[i]:
                    pbest_fit[i] = fitness
                    pbest_pos[i] = positions[i].copy()

                # Update global best
                if fitness < gbest_fit:
                    gbest_fit = fitness
                    gbest_pos = positions[i].copy()

            # Update velocities and positions (PSO equations)
            r1 = np.random.rand(PSO_SWARM_SIZE, self.dim)
            r2 = np.random.rand(PSO_SWARM_SIZE, self.dim)

            velocities = (
                w * velocities
                + PSO_C1 * r1 * (pbest_pos - positions)
                + PSO_C2 * r2 * (gbest_pos - positions)
            )
            positions = np.clip(positions + velocities, LOWER, UPPER)

            hidden, layers, lr, drop = _decode_particle(gbest_pos)
            print(
                f"[PSO] Iter {iteration+1:3d}/{PSO_MAX_ITER}  "
                f"best_loss={gbest_fit:.6f}  "
                f"hidden={hidden:3d}  layers={layers}  "
                f"lr={lr:.2e}  dropout={drop:.2f}"
            )

        # ── Report Best Configuration ─────────────────────────────────────
        hidden_size, num_layers, lr, dropout = _decode_particle(gbest_pos)
        print(f"\n[PSO] ── Best Configuration Found ───────────────────────")
        print(f"[PSO] hidden_size   : {hidden_size}")
        print(f"[PSO] num_layers    : {num_layers}")
        print(f"[PSO] learning_rate : {lr:.2e}")
        print(f"[PSO] dropout       : {dropout:.2f}")
        print(f"[PSO] val_loss      : {gbest_fit:.6f}")
        print(f"[PSO] → Now training full model with Adam from this config...\n")

        return {
            "hidden_size":   hidden_size,
            "num_layers":    num_layers,
            "learning_rate": lr,
            "dropout":       dropout,
            "val_loss":      gbest_fit
        }
