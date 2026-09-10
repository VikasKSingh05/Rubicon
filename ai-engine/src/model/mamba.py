"""Hardware-agnostic selective state-space layer (Mamba) in pure PyTorch.

This is a faithful PyTorch re-implementation of the selective SSM from
"Mamba: Linear-Time Sequence Modeling with Selective State Spaces" (Gu & Dao,
2023). It trades the official fused-CUDA kernels for a portable loop so the
concrete Rubicon deployment can train and serve on CPU. Numerically it is a
first-order approximation of the discretization and is intended for training
small models at high iteration speed.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def selective_scan(x, delta, A, B, C, D):
    """Selective scan: discrete-time recurrence over the token sequence.

    x, delta, D : (B, L, d_inner)
    A           : (d_inner, d_state)  (negative diagonal)
    B, C        : (B, L, d_state)
    returns     : (B, L, d_inner)
    """
    B_n, L, d_inner = x.shape
    dA = torch.exp(delta.unsqueeze(-1) * A)          # (B, L, d_inner, d_state)
    dB = delta.unsqueeze(-1) * B.unsqueeze(2)        # (B, L, 1, d_state)
    h = torch.zeros(B_n, d_inner, A.shape[1], device=x.device, dtype=x.dtype)
    ys = []
    for t in range(L):
        h = dA[:, t] * h + dB[:, t] * x[:, t].unsqueeze(-1)
        y = torch.einsum("bdn,bn->bd", h, C[:, t]) + D * x[:, t]
        ys.append(y)
    return torch.stack(ys, dim=1)


class MambaBlock(nn.Module):
    """One Mamba layer: in_proj -> conv -> selective SSM -> gate -> out_proj."""

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dt_rank: str = "auto",
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(expand * d_model)
        if dt_rank == "auto":
            dt_rank = max(1, d_model // 4)
        self.dt_rank = dt_rank

        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner, kernel_size=d_conv,
            padding=d_conv - 1, groups=self.d_inner, bias=False,
        )
        self.x_proj = nn.Linear(self.d_inner, dt_rank + d_state * 2, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32)
        self.register_buffer("A_log", torch.log(A).repeat(self.d_inner, 1))
        self.D = nn.Parameter(torch.ones(self.d_inner))

    def forward(self, x: torch.Tensor, residual: torch.Tensor = None):
        """x: (B, L, d_model) -> (B, L, d_model)"""
        B, L, _ = x.shape
        xz = self.in_proj(x)                                  # (B, L, 2*d_inner)
        x_, z = xz.chunk(2, dim=-1)

        x_conv = x_.transpose(1, 2)                           # (B, d_inner, L)
        x_conv = self.conv1d(x_conv)[..., :L].transpose(1, 2)  # causal, (B, L, d_inner)

        dt_b_c = self.x_proj(x_conv)
        dt, B_, C_ = dt_b_c.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = F.softplus(self.dt_proj(dt))                     # (B, L, d_inner)
        A = -torch.exp(self.A_log.float())                    # (d_inner, d_state)

        y = selective_scan(x_conv, dt, A, B_, C_, self.D)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out if residual is None else out + residual