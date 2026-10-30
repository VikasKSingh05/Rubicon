"""Rubicon Siamese U-Net Training Pipeline.

CLI:
    python train.py --epochs 30 --config auto --subset 0.2 --resume <checkpoint_path> [--synthetic]

Features:
    - Auto-hardware detection (local 6GB RTX 3050, Kaggle >=15GB, CPU fallback)
    - Automatic OOM recovery: halves crop size (512 -> 384 -> 256) on CUDA OOM
    - AdamW, lr 3e-4, Cosine schedule with 1 epoch warmup
    - Automatic Mixed Precision (AMP fp16), channels_last memory format
    - Gradient accumulation and gradient clipping (max norm 1.0)
    - Early stopping on validation xView2 score
    - Saves checkpoint_best.pt and checkpoint_last.pt every epoch
    - Fully resumable via --resume
    - TensorBoard and per-epoch duration/ETA logging
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ai_engine.model.hardware import detect_hardware, HardwareProfile
from ai_engine.model.siamese_unet import SiameseUNet
from ai_engine.model.losses import RubiconLoss
from ai_engine.model.metrics import XView2Metrics
from ai_engine.data.splits import load_or_create_splits, get_xbd_root
from ai_engine.data.dataset import XBDDataset, get_weighted_sampler, compute_norm_stats
from ai_engine.data.synthetic import generate_synthetic_dataset


def get_checkpoint_sha256(path: Path | str) -> str:
    """Compute sha256 checksum of checkpoint for provenance."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def parse_args():
    parser = argparse.ArgumentParser(description="Train Rubicon Siamese U-Net Disaster Assessment Model")
    parser.add_argument("--epochs", type=int, default=10, help="Total training epochs")
    parser.add_argument("--config", type=str, default="auto", help="Hardware profile: auto, local_6gb, kaggle, cpu")
    parser.add_argument("--subset", type=str, default="1.0", help="Fraction of training data (e.g. 0.2, or 'tiny')")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--synthetic", action="store_true", help="Train on synthetic procedural dataset")
    parser.add_argument("--xbd-root", type=str, default=None, help="Root path to xBD dataset")
    parser.add_argument("--output-dir", type=str, default="data/models", help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs/rubicon_training", help="TensorBoard log directory")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience epochs")
    return parser.parse_args()


def train():
    args = parse_args()

    # 1. Hardware auto-detection
    profile = detect_hardware()
    device = torch.device(profile.device)

    # 2. Dataset resolution and splitting
    if args.synthetic:
        print("[INFO] --synthetic flag active. Generating procedural dataset...")
        xbd_root = generate_synthetic_dataset()
    else:
        try:
            xbd_root = get_xbd_root(args.xbd_root)
        except FileNotFoundError as err:
            print(f"[WARN] {err}")
            print("[INFO] Falling back to procedural SYNTHETIC dataset for training...")
            xbd_root = generate_synthetic_dataset()

    splits = load_or_create_splits(xbd_root=xbd_root)
    train_manifest = splits["train"]
    val_manifest = splits["val"]

    # Handle subset argument
    if args.subset.lower() == "tiny":
        train_manifest = train_manifest[:min(50, len(train_manifest))]
        val_manifest = val_manifest[:min(20, len(val_manifest))]
        print(f"[INFO] Running in TINY SMOKE TEST mode: {len(train_manifest)} train, {len(val_manifest)} val tiles.")
    else:
        subset_ratio = float(args.subset)
        if subset_ratio < 1.0:
            n_sub = max(2, int(len(train_manifest) * subset_ratio))
            train_manifest = train_manifest[:n_sub]
            print(f"[INFO] Reduced subset training ({subset_ratio*100:.0f}%): {len(train_manifest)} train tiles.")

    # Compute/load normalization statistics
    norm_mean, norm_std = compute_norm_stats(train_manifest)

    # 3. Create Datasets and DataLoaders
    current_crop_size = profile.crop_size
    train_dataset = XBDDataset(
        tiles=train_manifest,
        crop_size=current_crop_size,
        is_train=True,
        norm_stats=(norm_mean, norm_std),
    )
    val_dataset = XBDDataset(
        tiles=val_manifest,
        crop_size=current_crop_size,
        is_train=False,
        norm_stats=(norm_mean, norm_std),
    )

    weighted_sampler = get_weighted_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=profile.batch_size,
        sampler=weighted_sampler,
        num_workers=profile.num_workers,
        pin_memory=profile.pin_memory,
        drop_last=True if len(train_dataset) >= profile.batch_size else False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=max(1, profile.batch_size),
        shuffle=False,
        num_workers=profile.num_workers,
        pin_memory=profile.pin_memory,
    )

    # 4. Initialize Model, Loss, Optimizer, Scheduler, Scaler
    print(f"\n[INFO] Initializing Siamese U-Net with shared '{profile.encoder}' encoder...")
    model = SiameseUNet(
        encoder_name=profile.encoder,
        encoder_weights="imagenet",
        num_classes=5,
    ).to(device)

    if profile.channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = RubiconLoss(loc_weight=0.4, dmg_weight=0.6).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    # Cosine schedule with 1 epoch linear warmup
    total_epochs = args.epochs
    warmup_epochs = 1

    def lr_lambda(epoch_idx: int) -> float:
        if epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(max(1, warmup_epochs))
        progress = float(epoch_idx - warmup_epochs) / float(max(1, total_epochs - warmup_epochs))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=(profile.use_amp and device.type == "cuda"))

    start_epoch = 0
    best_score = -1.0
    patience_counter = 0

    # 5. Handle --resume
    if args.resume and Path(args.resume).is_file():
        print(f"[INFO] Resuming training from checkpoint: {args.resume}")
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler and "scaler_state_dict" in checkpoint and checkpoint["scaler_state_dict"]:
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_score = checkpoint.get("xview2_score", -1.0)
        print(f"[INFO] Resumed from epoch {start_epoch} with best xView2 score: {best_score:.4f}")

    # Output and Logging setup
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    epoch_durations = []

    print("\n" + "=" * 65)
    print(f"STARTING TRAINING: {len(train_dataset)} Train Tiles | {len(val_dataset)} Val Tiles | {total_epochs} Epochs")
    print(f"Device: {device} | AMP: {profile.use_amp} | Effective Batch: {profile.batch_size * profile.grad_accum_steps}")
    print("=" * 65 + "\n")

    # 6. Training Loop
    for epoch in range(start_epoch, total_epochs):
        t_epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        running_loc_loss = 0.0
        running_dmg_loss = 0.0
        num_batches = 0

        optimizer.zero_grad(set_to_none=True)

        try:
            for step, (pre, post, loc_mask, dmg_mask) in enumerate(train_loader):
                pre = pre.to(device, non_blocking=True)
                post = post.to(device, non_blocking=True)
                loc_mask = loc_mask.to(device, non_blocking=True)
                dmg_mask = dmg_mask.to(device, non_blocking=True)

                if profile.channels_last:
                    pre = pre.to(memory_format=torch.channels_last)
                    post = post.to(memory_format=torch.channels_last)

                with torch.amp.autocast("cuda", enabled=(profile.use_amp and device.type == "cuda")):
                    loc_logits, dmg_logits = model(pre, post)
                    loss, breakdown = criterion(loc_logits, dmg_logits, loc_mask, dmg_mask)
                    loss_scaled = loss / profile.grad_accum_steps

                scaler.scale(loss_scaled).backward()

                if (step + 1) % profile.grad_accum_steps == 0 or (step + 1) == len(train_loader):
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                running_loss += breakdown["total_loss"]
                running_loc_loss += breakdown["loc_loss"]
                running_dmg_loss += breakdown["dmg_loss"]
                num_batches += 1

        except (torch.cuda.OutOfMemoryError, RuntimeError) as err:
            if "out of memory" in str(err).lower() or isinstance(err, torch.cuda.OutOfMemoryError):
                torch.cuda.empty_cache()
                if current_crop_size > 256:
                    new_crop = 384 if current_crop_size == 512 else 256
                    print(f"\n[ALERT] CUDA OOM encountered! Halving crop size from {current_crop_size} to {new_crop} and resuming...")
                    current_crop_size = new_crop
                    train_dataset.set_crop_size(new_crop)
                    val_dataset.set_crop_size(new_crop)
                    continue
                else:
                    raise err
            else:
                raise err

        scheduler.step()
        epoch_dur = round(time.perf_counter() - t_epoch_start, 2)
        epoch_durations.append(epoch_dur)

        avg_train_loss = running_loss / max(1, num_batches)
        avg_loc_loss = running_loc_loss / max(1, num_batches)
        avg_dmg_loss = running_dmg_loss / max(1, num_batches)

        # 7. Validation Loop
        model.eval()
        val_metrics_meter = XView2Metrics()
        val_running_loss = 0.0
        val_batches = 0

        with torch.no_grad():
            for pre, post, loc_mask, dmg_mask in val_loader:
                pre = pre.to(device, non_blocking=True)
                post = post.to(device, non_blocking=True)
                loc_mask = loc_mask.to(device, non_blocking=True)
                dmg_mask = dmg_mask.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=(profile.use_amp and device.type == "cuda")):
                    loc_logits, dmg_logits = model(pre, post)
                    v_loss, v_breakdown = criterion(loc_logits, dmg_logits, loc_mask, dmg_mask)

                val_running_loss += v_breakdown["total_loss"]
                val_batches += 1

                # Predictions
                loc_probs = torch.sigmoid(loc_logits).cpu().numpy()
                dmg_preds = torch.argmax(dmg_logits, dim=1).cpu().numpy()

                val_metrics_meter.update(
                    loc_preds=loc_probs,
                    loc_targets=loc_mask.cpu().numpy(),
                    dmg_preds=dmg_preds,
                    dmg_targets=dmg_mask.cpu().numpy(),
                )

        avg_val_loss = val_running_loss / max(1, val_batches)
        metrics = val_metrics_meter.compute()
        xview2_score = metrics["xview2_score"]

        # Estimate remaining training time
        avg_epoch_sec = sum(epoch_durations) / len(epoch_durations)
        remaining_epochs = total_epochs - (epoch + 1)
        est_remaining_sec = round(avg_epoch_sec * remaining_epochs, 1)

        print(
            f"Epoch [{epoch+1:02d}/{total_epochs:02d}] "
            f"Train Loss: {avg_train_loss:.4f} (Loc: {avg_loc_loss:.3f}, Dmg: {avg_dmg_loss:.3f}) | "
            f"Val Loss: {avg_val_loss:.4f} | "
            f"Loc F1: {metrics['loc_f1']:.4f} | "
            f"Dmg Harmonic F1: {metrics['damage_f1_harmonic']:.4f} | "
            f"xView2 Score: {xview2_score:.4f} | "
            f"Time: {epoch_dur:.1f}s (ETA: {est_remaining_sec}s)"
        )

        # TensorBoard Logging
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("xView2/Score", xview2_score, epoch)
        writer.add_scalar("xView2/Loc_F1", metrics["loc_f1"], epoch)
        writer.add_scalar("xView2/Damage_F1_Harmonic", metrics["damage_f1_harmonic"], epoch)
        for dmg_cls, val_f1 in metrics["damage_f1_classes"].items():
            writer.add_scalar(f"Damage_F1/{dmg_cls}", val_f1, epoch)

        # 8. Checkpoint Saving (Every Epoch: last and best)
        checkpoint_data = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "xview2_score": xview2_score,
            "metrics": metrics,
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "config": {
                "encoder": profile.encoder,
                "crop_size": current_crop_size,
                "model_version": f"siamese_unet_{profile.encoder}_v1",
                "hardware_profile": profile.name,
                "epochs_trained": epoch + 1,
            },
        }

        # Save last checkpoint every epoch
        last_ckpt_path = out_dir / "checkpoint_last.pt"
        torch.save(checkpoint_data, last_ckpt_path)

        # Save best checkpoint
        if xview2_score > best_score:
            best_score = xview2_score
            patience_counter = 0
            best_ckpt_path = out_dir / "checkpoint_best.pt"
            torch.save(checkpoint_data, best_ckpt_path)
            # Also mirror directly to ai-engine serving artifact
            serving_model_path = out_dir / "model.pt"
            torch.save(checkpoint_data, serving_model_path)
            print(f"  [*] New best xView2 score achieved: {best_score:.4f}! Checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\n[INFO] Early stopping triggered after {args.patience} epochs without score improvement.")
                break

    writer.close()

    # Final summary report
    print("\n" + "=" * 65)
    print("TRAINING RUN COMPLETE")
    print(f"  Best xView2 Score Achieved: {best_score:.4f}")
    if epoch_durations:
        avg_s = sum(epoch_durations) / len(epoch_durations)
        print(f"  Measured Seconds Per Epoch:  {avg_s:.2f} s/epoch")
        full_run_30_est = round(avg_s * 30 / 60, 1)
        print(f"  Estimated Full Run (30 ep):  {full_run_30_est} minutes")
    print(f"  Checkpoints saved in:        {out_dir.resolve()}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    train()
