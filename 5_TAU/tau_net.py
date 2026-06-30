"""
3D CNN surrogate model for per-phase tortuosity factor prediction.

Architecture mirrors 2_CNN/model.py but:
  - ndf=16 (vs 32) — fewer parameters for ~80 real training structures
  - Dropout3d on deeper layers for regularization
  - Single scalar output (τ) instead of SSA

Input:  (batch, 1, 64, 64, 64)  — single-phase binary-like probability map
Output: (batch,)                 — predicted tortuosity factor τ
"""

import torch.nn as nn


class TauNet(nn.Module):
    def __init__(self, ndf=16):
        super().__init__()
        self.net = nn.Sequential(
            # 64³ → 32³
            nn.Conv3d(1, ndf, 5, padding=2, bias=False),
            nn.BatchNorm3d(ndf),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d(2),
            # 32³ → 16³
            nn.Conv3d(ndf, ndf * 2, 5, padding=2, bias=False),
            nn.BatchNorm3d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.MaxPool3d(2),
            # 16³ → 8³
            nn.Conv3d(ndf * 2, ndf * 4, 3, padding=1, bias=False),
            nn.BatchNorm3d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.1),
            nn.MaxPool3d(2),
            # 8³ → 4³
            nn.Conv3d(ndf * 4, ndf * 8, 3, padding=1, bias=False),
            nn.BatchNorm3d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout3d(0.1),
            nn.MaxPool3d(2),
            # 4³ → scalar
            nn.Conv3d(ndf * 8, 1, 3, padding=1, bias=False),
            nn.AdaptiveAvgPool3d(1),
            nn.Flatten(),
        )

    def forward(self, x):
        out = self.net(x)
        return out.squeeze(-1)   # (batch,) — or scalar when batch=1 after squeeze


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)
