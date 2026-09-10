"""Mamba-Transformer fusion classifier (HSI + LiDAR -> severity).

Inputs per the Phase 2 tensor format:
  X_hsi : (B, k, 32, 32)  PCA-reduced hyperspectral channels
  X_dem : (B, 1, 32, 32)  normalized height grid

Architecture:
  HSI branch   conv stem -> 8x8 token grid -> (MambaBlock + TransformerBlock)*n
               -> mean pool
  LiDAR branch small CNN trunk -> global average pool
  head         concat -> MLP -> 3-class severity logits
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .mamba import MambaBlock


class TransformerBlock(nn.Module):
    """Pre-LN self-attention block with an MLP."""

    def __init__(self, d: int, n_heads: int, ff_scale: int = 4, dropout: float = 0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, n_heads, dropout=dropout, batch_first=True)
        self.ln2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            nn.Linear(d, ff_scale * d), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_scale * d, d),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n = self.ln1(x)
        x = x + self.attn(n, n, n)[0]
        x = x + self.mlp(self.ln2(x))
        return x


class HsiBranch(nn.Module):
    def __init__(self, in_channels: int, hsi_cfg: dict):
        super().__init__()
        self.token_grid = int(hsi_cfg["token_grid"])
        stride = 32 // self.token_grid
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, hsi_cfg["stem_channels"], stride, stride=stride),
            nn.GELU(),
            nn.Conv2d(hsi_cfg["stem_channels"], hsi_cfg["stem_channels"], 3, padding=1),
            nn.GELU(),
        )
        self.token_proj = nn.Linear(hsi_cfg["stem_channels"], hsi_cfg["d_model"])
        self.pos = nn.Parameter(
            torch.empty(1, self.token_grid * self.token_grid, hsi_cfg["d_model"])
        )
        nn.init.normal_(self.pos, std=0.02)

        blocks = []
        for _ in range(int(hsi_cfg["n_blocks"])):
            blocks.append(
                MambaBlock(
                    d_model=hsi_cfg["d_model"],
                    d_state=int(hsi_cfg["d_state"]),
                    d_conv=int(hsi_cfg["d_conv"]),
                    expand=int(hsi_cfg["expand"]),
                )
            )
            blocks.append(
                TransformerBlock(
                    d=hsi_cfg["d_model"],
                    n_heads=int(hsi_cfg["n_heads"]),
                    ff_scale=int(hsi_cfg["ff_scale"]),
                    dropout=float(hsi_cfg["dropout"]),
                )
            )
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = self.stem(x)                                   # (B, C, g, g)
        g = self.token_grid
        B = x.shape[0]
        if t.shape[2] != g or t.shape[3] != g:
            raise ValueError(f"stem produced {t.shape[2]}x{t.shape[3]} grid, expected {g}x{g}")
        t = t.flatten(2).transpose(1, 2)                   # (B, g*g, C)
        t = self.token_proj(t) + self.pos[:, : t.shape[1]]
        for blk in self.blocks:
            t = blk(t)
        return t.mean(dim=1)                               # (B, d_model)


class DemBranch(nn.Module):
    def __init__(self, in_channels: int, dem_cfg: dict):
        super().__init__()
        layers = []
        c_in = in_channels
        for c_out in dem_cfg["channels"]:
            layers += [
                nn.Conv2d(c_in, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.GELU(),
                nn.Conv2d(c_out, c_out, 3, padding=1), nn.BatchNorm2d(c_out), nn.GELU(),
                nn.MaxPool2d(2),
            ]
            c_in = c_out
        drop = float(dem_cfg.get("dropout", 0.1))
        layers.append(nn.Dropout2d(drop))
        self.trunk = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.trunk(x)
        return h.mean(dim=(2, 3))                          # (B, c_out)


class FusionModel(nn.Module):
    """Late-fusion Mamba-Transformer classifier."""

    def __init__(self, hsi_channels: int, dem_channels: int, num_classes: int,
                 hsi_cfg: dict, dem_cfg: dict, head_cfg: dict):
        super().__init__()
        self.hsi = HsiBranch(hsi_channels, hsi_cfg)
        self.dem = DemBranch(dem_channels, dem_cfg)
        fused = hsi_cfg["d_model"] + dem_cfg["channels"][-1]
        self.head = nn.Sequential(
            nn.LayerNorm(fused),
            nn.Linear(fused, head_cfg.get("hidden", 64)),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(head_cfg.get("hidden", 64), num_classes),
        )

    def forward(self, x_hsi: torch.Tensor, x_dem: torch.Tensor) -> torch.Tensor:
        f = torch.cat([self.hsi(x_hsi), self.dem(x_dem)], dim=-1)
        return self.head(f)