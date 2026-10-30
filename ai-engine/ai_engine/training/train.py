"""Rubicon Siamese U-Net training loop for xBD disaster damage assessment.

Features:
  - Hardware auto-detection (RTX 3050 6 GB / Kaggle / CPU fallback)
  - AMP fp16 + channels_last memory layout for GPU efficiency
  - Gradient accumulation (eff. batch 8 on local, 8 on Kaggle)
  - Automatic OOM recovery: catches cuda.OutOfMemoryError, halves crop
    size, clears cache, resumes without crashing
  - Weighted sampler oversampling destroyed/major tiles
  - Event-based train/val split (zero spatial overlap)
  - Checkpoint: data/models/siamese_unet.pt  + metrics.json
  - TensorBoard logging

Usage:
    python -m ai_engine.training.train [--synthetic] [--xbd-root PATH]
                                       [--epochs N] [--subset N]
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler
try:
    from torch.utils.tensorboard import SummaryWriter
    _TB = True
except ImportError:
    _TB = False

from ..data.splits import load_or_create_splits
from ..data.synthetic import generate_synthetic_dataset
from ..data.dataset import XBDDataset, get_weighted_sampler
from ..model.siamese_unet import SiameseUNet
from ..model.losses import RubiconLoss
from ..model.metrics import XView2Metrics
from ..model.hardware import detect_hardware, HardwareProfile


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _build_loader(tiles, profile: HardwareProfile, is_train: bool, cache_dir: Path) -> DataLoader:
    """Build a DataLoader from tile metadata and hardware profile."""
    ds = XBDDataset(
        tiles=tiles,
        crop_size=profile.crop_size,
        is_train=is_train,
        cache_dir=cache_dir,
    )
    sampler = get_weighted_sampler(ds) if is_train else None
    return DataLoader(
        ds,
        batch_size=profile.batch_size,
        sampler=sampler,
        shuffle=(not is_train and sampler is None),
        num_workers=profile.num_workers,
        pin_memory=profile.pin_memory,
        drop_last=is_train,
    ), ds


def _channels_last(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    if enabled and x.is_cuda:
        return x.to(memory_format=torch.channels_last)
    return x


@contextlib.contextmanager
def _maybe_autocast(enabled: bool):
    if enabled and torch.cuda.is_available():
        with torch.amp.autocast("cuda"):
            yield
    else:
        yield


# ---------------------------------------------------------------------------
# One epoch
# ---------------------------------------------------------------------------

def _train_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: RubiconLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    profile: HardwareProfile,
    accum_steps: int,
    writer,
    epoch: int,
    step_counter: list,
) -> dict[str, float]:
    model.train()
    running = {k: 0.0 for k in ("total_loss", "loc_loss", "dmg_loss")}
    n = 0
    optimizer.zero_grad()

    for batch_idx, (pre, post, loc_t, dmg_t) in enumerate(loader):
        pre = _channels_last(pre.to(device), profile.channels_last)
        post = _channels_last(post.to(device), profile.channels_last)
        loc_t = loc_t.to(device)
        dmg_t = dmg_t.to(device)

        with _maybe_autocast(profile.use_amp):
            loc_logits, dmg_logits = model(pre, post)
            loss, breakdown = criterion(loc_logits, dmg_logits, loc_t, dmg_t)
            loss = loss / accum_steps

        if scaler is not None:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (batch_idx + 1) % accum_steps == 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            optimizer.zero_grad()

        bs = pre.size(0)
        n += bs
        for k in running:
            running[k] += breakdown[k] * bs * accum_steps

        if writer and step_counter:
            step_counter[0] += 1
            writer.add_scalar("train/total_loss_step", breakdown["total_loss"], step_counter[0])

    return {k: v / max(n, 1) for k, v in running.items()}


# ---------------------------------------------------------------------------
# Validation epoch
# ---------------------------------------------------------------------------

def _val_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: RubiconLoss,
    device: torch.device,
    profile: HardwareProfile,
) -> tuple[dict[str, float], dict[str, Any]]:
    model.eval()
    metrics = XView2Metrics()
    running = {k: 0.0 for k in ("total_loss", "loc_loss", "dmg_loss")}
    n = 0

    with torch.no_grad():
        for pre, post, loc_t, dmg_t in loader:
            pre = _channels_last(pre.to(device), profile.channels_last)
            post = _channels_last(post.to(device), profile.channels_last)
            loc_t = loc_t.to(device)
            dmg_t = dmg_t.to(device)

            with _maybe_autocast(profile.use_amp):
                loc_logits, dmg_logits = model(pre, post)
                _, breakdown = criterion(loc_logits, dmg_logits, loc_t, dmg_t)

            bs = pre.size(0)
            n += bs
            for k in running:
                running[k] += breakdown[k] * bs

            # Metrics update
            loc_prob = torch.sigmoid(loc_logits).cpu().numpy()
            dmg_pred = dmg_logits.argmax(dim=1).cpu().numpy()
            loc_np = loc_t.cpu().numpy()
            dmg_np = dmg_t.cpu().numpy()

            metrics.update(loc_prob[:, 0], loc_np[:, 0], dmg_pred, dmg_np)

    loss_dict = {k: v / max(n, 1) for k, v in running.items()}
    return loss_dict, metrics.compute()


# ---------------------------------------------------------------------------
# OOM-safe training wrapper
# ---------------------------------------------------------------------------

def _oom_safe_train(
    model: nn.Module,
    train_ds: XBDDataset,
    val_ds: XBDDataset,
    profile: HardwareProfile,
    criterion: RubiconLoss,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler | None,
    device: torch.device,
    accum_steps: int,
    writer,
    epoch: int,
    step_counter: list,
) -> dict:
    """Run one train epoch; on OOM, halve crop size and retry once."""
    def _make_loader(ds, is_train):
        sampler = get_weighted_sampler(ds) if is_train else None
        return DataLoader(
            ds,
            batch_size=profile.batch_size,
            sampler=sampler,
            shuffle=(not is_train and sampler is None),
            num_workers=profile.num_workers,
            pin_memory=profile.pin_memory,
            drop_last=is_train,
        )

    for attempt in range(3):
        try:
            loader = _make_loader(train_ds, is_train=True)
            return _train_epoch(
                model, loader, criterion, optimizer, scaler,
                device, profile, accum_steps, writer, epoch, step_counter,
            )
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            new_crop = max(256, train_ds.crop_size // 2)
            print(
                f"[WARN] CUDA OOM on epoch {epoch} attempt {attempt+1} "
                f"(crop {train_ds.crop_size}→{new_crop})",
                file=sys.stderr,
            )
            train_ds.set_crop_size(new_crop)
            val_ds.set_crop_size(new_crop)
    raise RuntimeError("OOM persists after 3 crop reductions; aborting training")


# ---------------------------------------------------------------------------
# Main training function
# ---------------------------------------------------------------------------

def train(
    xbd_root=None,
    synthetic: bool = False,
    epochs: int | None = None,
    subset: int | None = None,
    output_dir: str = "data/models",
    model_version: str = "siamese_unet-v1",
) -> dict:
    """Train the Siamese U-Net and return metrics dict."""
    profile = detect_hardware()
    device = torch.device(profile.device)

    # ── Data ──────────────────────────────────────────────────────────────────
    if synthetic or xbd_root is None:
        print("[INFO] Generating synthetic xBD dataset for training/CI...")
        xbd_root_path = generate_synthetic_dataset()
        splits = load_or_create_splits(xbd_root=xbd_root_path, force_recreate=True)
    else:
        splits = load_or_create_splits(xbd_root=xbd_root, force_recreate=False)

    train_tiles = splits["train"]
    val_tiles = splits["val"]

    if subset is not None:
        train_tiles = train_tiles[:subset]
        val_tiles = val_tiles[:max(1, subset // 4)]

    cache_dir = Path("data/xbd_cache")
    cache_dir.mkdir(parents=True, exist_ok=True)

    train_ds = XBDDataset(
        tiles=train_tiles, crop_size=profile.crop_size, is_train=True, cache_dir=cache_dir
    )
    val_ds = XBDDataset(
        tiles=val_tiles, crop_size=profile.crop_size, is_train=False, cache_dir=cache_dir
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=profile.batch_size,
        shuffle=False,
        num_workers=profile.num_workers,
        pin_memory=profile.pin_memory,
    )

    print(f"[INFO] Train tiles: {len(train_ds)} | Val tiles: {len(val_ds)}")

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SiameseUNet(
        encoder_name=profile.encoder,
        encoder_weights="imagenet",
        num_classes=5,
    ).to(device)

    if profile.channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = RubiconLoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs or 30, eta_min=1e-6
    )
    scaler = GradScaler() if profile.use_amp else None
    accum_steps = profile.grad_accum_steps

    # ── TensorBoard ───────────────────────────────────────────────────────────
    writer = None
    if _TB:
        tb_dir = Path("runs") / f"rubicon_{int(time.time())}"
        writer = SummaryWriter(str(tb_dir))

    # ── Training loop ─────────────────────────────────────────────────────────
    n_epochs = epochs or (10 if profile.name == "cpu_fallback" else 30)
    best_score = -1.0
    best_state = None
    step_counter = [0]
    history = []

    for epoch in range(1, n_epochs + 1):
        t0 = time.time()

        train_losses = _oom_safe_train(
            model, train_ds, val_ds, profile, criterion,
            optimizer, scaler, device, accum_steps,
            writer, epoch, step_counter,
        )
        val_losses, val_metrics = _val_epoch(
            model, val_loader, criterion, device, profile
        )
        scheduler.step()

        xv2 = val_metrics["xview2_score"]
        elapsed = round(time.time() - t0, 1)
        marker = " ← best" if xv2 > best_score else ""

        print(
            f"[epoch {epoch:03d}/{n_epochs}] "
            f"train_loss={train_losses['total_loss']:.4f} "
            f"val_loss={val_losses['total_loss']:.4f} "
            f"loc_F1={val_metrics['loc_f1']:.4f} "
            f"xView2={xv2:.4f} "
            f"({elapsed}s){marker}"
        )

        if writer:
            writer.add_scalar("train/total_loss", train_losses["total_loss"], epoch)
            writer.add_scalar("val/total_loss", val_losses["total_loss"], epoch)
            writer.add_scalar("val/xview2_score", xv2, epoch)
            writer.add_scalar("val/loc_f1", val_metrics["loc_f1"], epoch)

        if xv2 > best_score:
            best_score = xv2
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        history.append({"epoch": epoch, "train_loss": train_losses["total_loss"],
                         "val_loss": val_losses["total_loss"], "xview2_score": xv2})

    if writer:
        writer.close()

    # ── Save checkpoint ───────────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    _, final_metrics = _val_epoch(model, val_loader, criterion, device, profile)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    checkpoint = {
        "model_version": model_version,
        "encoder_name": profile.encoder,
        "num_classes": 5,
        "damage_classes": ["background", "no-damage", "minor-damage", "major-damage", "destroyed"],
        "state_dict": model.state_dict(),
        "profile": profile.name,
    }
    torch.save(checkpoint, out_dir / "siamese_unet.pt")

    metrics_out = {
        "model_version": model_version,
        "best_val_xview2_score": best_score,
        "final_val": final_metrics,
        "history": history,
        "hardware_profile": profile.name,
    }
    (out_dir / "siamese_metrics.json").write_text(json.dumps(metrics_out, indent=2))

    print(f"[INFO] Checkpoint: {out_dir / 'siamese_unet.pt'}")
    print(f"[INFO] Metrics:    {out_dir / 'siamese_metrics.json'}")
    print(f"[INFO] Best xView2 score: {best_score:.4f}")

    return metrics_out


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Train Rubicon Siamese U-Net on xBD dataset.")
    ap.add_argument("--synthetic", action="store_true",
                    help="Use procedurally generated synthetic dataset (no real xBD required)")
    ap.add_argument("--xbd-root", default=None,
                    help="Path to xBD dataset root (overrides XBD_ROOT env var)")
    ap.add_argument("--epochs", type=int, default=None,
                    help="Override number of training epochs")
    ap.add_argument("--subset", type=int, default=None,
                    help="Limit training to first N tiles (for quick tests)")
    ap.add_argument("--output-dir", default="data/models",
                    help="Directory to write checkpoint and metrics")
    args = ap.parse_args(argv)

    train(
        xbd_root=args.xbd_root,
        synthetic=args.synthetic,
        epochs=args.epochs,
        subset=args.subset,
        output_dir=args.output_dir,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
