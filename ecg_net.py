"""
ECG Classification — 5-Block Deep Architecture
Dataset: PTB-XL (Wagner et al., 2020, Nature Scientific Data)

Pipeline:
    Input (B, 12, T)
        |
        v
    [1] CNN              -- local waveform features
        v
    [2] Autoencoder      -- latent compression + reconstruction auxiliary loss
        v
    [3] BiLSTM           -- long-range temporal dependencies (forward + backward)
        v
    [4] GRU              -- lightweight sequential refinement
        v
    [5] Temporal Attn    -- weighted pooling over time
        v
    Classifier head      -- multi-class logits

For ablation studies, set the use_<block>=False flag on ECGNet.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Block 1 — CNN
# ─────────────────────────────────────────────────────────────────────────────
class CNNBlock(nn.Module):
    """Stacked 1D convolutions extract local waveform features (peaks, QRS, etc.)
    while progressively downsampling the time dimension."""
    def __init__(self, in_channels=12, hidden=64, out_channels=128, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, hidden, kernel_size=7, padding=3)
        self.bn1   = nn.BatchNorm1d(hidden)
        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.bn2   = nn.BatchNorm1d(hidden)
        self.conv3 = nn.Conv1d(hidden, out_channels, kernel_size=3, padding=1)
        self.bn3   = nn.BatchNorm1d(out_channels)
        self.pool  = nn.MaxPool1d(kernel_size=2)
        self.drop  = nn.Dropout(dropout)

    def forward(self, x):                  # (B, 12, T)
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        return self.drop(x)                # (B, 128, T/8)


# ─────────────────────────────────────────────────────────────────────────────
# Block 2 — Autoencoder
# ─────────────────────────────────────────────────────────────────────────────
class AutoencoderBlock(nn.Module):
    """Compresses CNN features into a lower-dim latent z, then reconstructs.
    The reconstruction loss is an auxiliary regularizer; z feeds downstream."""
    def __init__(self, in_channels=128, latent=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels, 96, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(96, latent, kernel_size=3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(latent, 96, kernel_size=3, padding=1), nn.ReLU(),
            nn.Conv1d(96, in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x):                  # (B, 128, T')
        z     = self.encoder(x)            # (B, 64, T')
        x_hat = self.decoder(z)            # (B, 128, T')
        return z, x_hat


# ─────────────────────────────────────────────────────────────────────────────
# Block 3 — BiLSTM
# ─────────────────────────────────────────────────────────────────────────────
class BiLSTMBlock(nn.Module):
    """Bidirectional LSTM captures long-range dependencies in both directions."""
    def __init__(self, input_size, hidden=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden, num_layers=num_layers,
                            batch_first=True, bidirectional=True,
                            dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, x):                  # (B, T, C)
        out, _ = self.lstm(x)              # (B, T, 2*hidden)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Block 4 — GRU
# ─────────────────────────────────────────────────────────────────────────────
class GRUBlock(nn.Module):
    """Lightweight recurrent refinement. Distinct from LSTM (no cell state)."""
    def __init__(self, input_size, hidden=64, num_layers=1, dropout=0.2):
        super().__init__()
        self.gru = nn.GRU(input_size, hidden, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)

    def forward(self, x):                  # (B, T, C)
        out, _ = self.gru(x)               # (B, T, hidden)
        return out


# ─────────────────────────────────────────────────────────────────────────────
# Block 5 — Temporal Attention
# ─────────────────────────────────────────────────────────────────────────────
class TemporalAttention(nn.Module):
    """Learns a weighted average over time. Diagnostic moments get higher weight."""
    def __init__(self, hidden):
        super().__init__()
        self.attn = nn.Linear(hidden, 1)

    def forward(self, x):                  # (B, T, H)
        scores  = self.attn(x).squeeze(-1) # (B, T)
        weights = torch.softmax(scores, dim=1)
        context = torch.sum(x * weights.unsqueeze(-1), dim=1)  # (B, H)
        return context, weights


# ─────────────────────────────────────────────────────────────────────────────
# Full Model
# ─────────────────────────────────────────────────────────────────────────────
class ECGNet(nn.Module):
    """Full 5-block model. Toggle blocks off for ablation studies."""
    def __init__(self, in_channels=12, num_classes=5,
                 use_cnn=True, use_ae=True,
                 use_bilstm=True, use_gru=True, use_attention=True,
                 dropout=0.3):
        super().__init__()
        self.flags = dict(cnn=use_cnn, ae=use_ae, bilstm=use_bilstm,
                          gru=use_gru, attn=use_attention)

        # ---- channel bookkeeping ---------------------------------------
        c = in_channels
        self.cnn = CNNBlock(c, out_channels=128) if use_cnn else None
        if use_cnn: c = 128

        self.ae = AutoencoderBlock(c, latent=64) if use_ae else None
        if use_ae: c = 64

        self.bilstm = BiLSTMBlock(c, hidden=64) if use_bilstm else None
        if use_bilstm: c = 128                  # 2 * hidden (bidirectional)

        self.gru = GRUBlock(c, hidden=64) if use_gru else None
        if use_gru: c = 64

        self.attention = TemporalAttention(c) if use_attention else None

        self.classifier = nn.Sequential(
            nn.Linear(c, 64), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):                  # x: (B, 12, T)
        recon_target = recon_pred = None

        if self.flags['cnn']:
            x = self.cnn(x)                # (B, 128, T')

        if self.flags['ae']:
            recon_target = x               # save pre-AE features for loss
            x, recon_pred = self.ae(x)     # x is now z

        # Switch to (B, T, C) for recurrent / attention blocks
        x = x.transpose(1, 2)              # (B, T, C)

        if self.flags['bilstm']:
            x = self.bilstm(x)             # (B, T, 128)

        if self.flags['gru']:
            x = self.gru(x)                # (B, T, 64)

        if self.flags['attn']:
            x, _ = self.attention(x)       # (B, C)
        else:
            x = x.mean(dim=1)              # fallback: mean-pool over time

        logits = self.classifier(x)
        return logits, recon_target, recon_pred


# ─────────────────────────────────────────────────────────────────────────────
# Training step example (joint classification + reconstruction loss)
# ─────────────────────────────────────────────────────────────────────────────
def compute_loss(logits, targets, recon_target, recon_pred,
                 ce_weight=1.0, recon_weight=0.1):
    """Joint loss: cross-entropy + autoencoder reconstruction."""
    ce = F.cross_entropy(logits, targets)
    if recon_target is not None and recon_pred is not None:
        recon = F.mse_loss(recon_pred, recon_target)
        return ce_weight * ce + recon_weight * recon, ce.item(), recon.item()
    return ce_weight * ce, ce.item(), 0.0


if __name__ == "__main__":
    # quick sanity check: 100 Hz PTB-XL means 1000 samples for a 10s recording
    model = ECGNet(in_channels=12, num_classes=5)
    x = torch.randn(4, 12, 1000)                        # batch of 4 ECGs
    y = torch.randint(0, 5, (4,))
    logits, rt, rp = model(x)
    loss, ce, rec = compute_loss(logits, y, rt, rp)
    print(f"logits: {logits.shape}  loss: {loss.item():.4f}  "
          f"(ce={ce:.4f}  recon={rec:.4f})")
    print(f"Params: {sum(p.numel() for p in model.parameters()):,}")
