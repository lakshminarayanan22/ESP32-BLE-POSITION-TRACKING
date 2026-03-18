import torch
import torch.nn as nn


class LSTMModel(nn.Module):
    """
    PSO-LSTM Architecture for indoor positioning.

    The LSTM processes a temporal sequence of RSSI feature vectors to capture
    the time-varying nature of indoor signal propagation. Unlike feedforward
    networks (BPNN), the LSTM cell state carries information across time steps,
    enabling the model to learn from sequential RSSI patterns caused by:
      - Person walking through the signal path (shadowing events)
      - Multipath effects that evolve over time
      - Non-line-of-sight (NLOS) conditions

    Architecture:
        Input  : (batch, SEQ_LEN, N_FEATURES=20) — sliding window of feature vecs
        LSTM   : multi-layer with dropout between layers, LayerNorm on output
        FC     : hidden → 32 → N_OUTPUTS=3  (predicts normalized x, y, z)

    Initialization:
        Weights are pre-seeded by PSO (see pso_optimizer.py) to avoid local
        minima and vanishing gradient issues. Adam optimizer then fine-tunes
        from this optimized starting point.
    """

    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout=0.2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        self.lstm = nn.LSTM(
            input_size  = input_size,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0
        )

        self.dropout    = nn.Dropout(dropout)

        # LayerNorm over feature dimension — works for any batch size including 1.
        # BatchNorm1d would fail at inference time when batch_size=1.
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        """
        x : (batch, seq_len, input_size)
        Returns: (batch, output_size) — normalized (x, y, z) coordinates
        """
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))   # (batch, seq_len, hidden_size)
        out    = out[:, -1, :]             # take last timestep: (batch, hidden_size)
        out    = self.dropout(out)
        out    = self.layer_norm(out)
        return self.fc(out)                # (batch, output_size)
