import torch
import torch.nn as nn

class LSTMModel(nn.Module):
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

        # ── FIX: Replace BatchNorm1d with LayerNorm ────
        # BatchNorm1d fails on batch_size=1 during training
        # LayerNorm normalizes across features instead of batch — works on any size
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, output_size)
        )

    def forward(self, x):
        batch_size = x.size(0)
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out    = out[:, -1, :]
        out, _ = self.lstm(out, (h0,c0))
        out    = out[:, -1, :]          # last timestep
        out    = self.dropout(out)
        out    = self.layer_norm(out)   # ← LayerNorm instead of BatchNorm
        return self.fc(out)