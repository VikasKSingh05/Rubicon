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
import time
from pathlib import Path

import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ai_engine.model.hardware import detect_hardware
from ai_engine.model.siamese_unet import SiameseUNet
from ai_engine.model.losses import RubiconLoss, inverse_sqrt_class_weights
from ai_engine.model.metrics import DAMAGE_CLASS_NAMES, XView2Metrics
from ai_engine.model.metrics import decode_damage_logits
from ai_engine.model.checkpoint import DAMAGE_SEVERITY_CLASS_NAMES, load_checkpoint, save_checkpoint
from ai_engine.data.splits import load_or_create_splits, get_xbd_root
from ai_engine.data.dataset import XBDDataset, compute_norm_stats
from ai_engine.data.synthetic import generate_synthetic_dataset


def get_checkpoint_sha256(path: Path | str) -> str:
    """Compute sha256 checksum of checkpoint for provenance."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(65536):
            h.update(chunk)
    return h.hexdigest()[:16]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train Rubicon Siamese U-Net Disaster Assessment Model")
    parser.add_argument("--epochs", type=int, default=10, help="Total training epochs")
    parser.add_argument("--config", type=str, default="auto", help="Hardware profile: auto, local_6gb, kaggle, cpu")
    parser.add_argument("--subset", type=str, default="1.0", help="Fraction of training data (e.g. 0.2, or 'tiny')")
    parser.add_argument("--max-val", type=int, default=None, help="Cap validation tiles (for smoke tests)")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--synthetic", action="store_true", help="Train on synthetic procedural dataset")
    parser.add_argument(
        "--synthetic-dir",
        type=str,
        default=None,
        help="Directory for generated synthetic data (requires --synthetic)",
    )
    parser.add_argument("--xbd-root", type=str, default=None, help="Root path to xBD dataset")
    parser.add_argument("--cache-dir", type=str, default="data/xbd_cache", help="Directory for cached rasterized masks")
    parser.add_argument("--no-pretrained", action="store_true", help="Do not download ImageNet encoder weights")
    parser.add_argument("--output-dir", type=str, default="data/models", help="Directory to save checkpoints")
    parser.add_argument("--log-dir", type=str, default="runs/rubicon_training", help="TensorBoard log directory")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=6, help="Early stopping patience epochs")
    parser.add_argument(
        "--model-version",
        type=str,
        default=None,
        help="Override the model version recorded in checkpoints",
    )
    args = parser.parse_args(argv)
    if args.synthetic_dir and not args.synthetic:
        parser.error("--synthetic-dir requires --synthetic")
    return args


def train(argv=None):
    args = parse_args(argv)
    run_started = time.perf_counter()

    # 1. Hardware auto-detection (respects --config)
    profile = detect_hardware(args.config)
    device = torch.device(profile.device)
    model_version = args.model_version or f"siamese_unet_{profile.encoder}_damage4_v2"

    # 2. Dataset resolution and splitting
    if args.synthetic:
        print("[INFO] --synthetic flag active. Generating procedural dataset...")
        xbd_root = generate_synthetic_dataset(output_dir=args.synthetic_dir or "data/synthetic_xbd")
    else:
        xbd_root = get_xbd_root(args.xbd_root)
    data_mode = "synthetic" if args.synthetic else "real_xbd"

    splits = load_or_create_splits(xbd_root=xbd_root)
    train_manifest = splits["train"]
    val_manifest = splits["val"]
    split_meta = splits.get("metadata", {})
    full_val_tile_count = len(val_manifest)

    print(f"[INFO] Source split: {split_meta.get('source_split')} | "
          f"train events: {', '.join(split_meta.get('train_events', []))} | "
          f"val events: {', '.join(split_meta.get('val_events', []))}")
    for warning in split_meta.get("warnings", []):
        print(f"[WARN] {warning}")

    # Handle subset argument
    if args.subset.lower() == "tiny":
        rng = np.random.default_rng(42)
        n_sub = min(50, len(train_manifest))
        indices = np.sort(rng.choice(len(train_manifest), size=n_sub, replace=False))
        train_manifest = [train_manifest[int(i)] for i in indices]
        print(
            f"[INFO] TINY training subset: {len(train_manifest)} train / "
            f"{len(val_manifest)} full event-disjoint validation tiles."
        )
    else:
        subset_ratio = float(args.subset)
        if not 0 < subset_ratio <= 1:
            raise ValueError(f"--subset must be in (0, 1], got {subset_ratio}")
        if subset_ratio < 1.0:
            n_sub = max(1, int(len(train_manifest) * subset_ratio))
            rng = np.random.default_rng(42)
            indices = np.sort(rng.choice(len(train_manifest), size=n_sub, replace=False))
            train_manifest = [train_manifest[int(i)] for i in indices]
            print(
                f"[INFO] Training subset {subset_ratio:.0%}: "
                f"{len(train_manifest)} train / {len(val_manifest)} full "
                "event-disjoint validation tiles (validation is not subset)."
            )

    if args.max_val is not None:
        if args.max_val < full_val_tile_count:
            print("[WARN] --max-val explicitly caps validation; this is not a full validation result.")
        val_manifest = val_manifest[: max(0, args.max_val)]
        print(f"[INFO] Validation capped to {len(val_manifest)} tiles (--max-val).")

    if not train_manifest or not val_manifest:
        raise SystemExit(
            "[ERROR] Train or validation manifest is empty. Check the event split "
            "and subset flags before training."
        )

    # Compute/load normalization statistics on TRAINING tiles only.
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    norm_mean, norm_std = compute_norm_stats(train_manifest)
    print(f"[INFO] Normalization mean={norm_mean} std={norm_std}")

    # 3. Create Datasets and DataLoaders
    current_crop_size = profile.crop_size
    train_dataset = XBDDataset(
        tiles=train_manifest,
        crop_size=current_crop_size,
        is_train=True,
        cache_dir=cache_dir,
        norm_stats=(norm_mean, norm_std),
    )
    val_dataset = XBDDataset(
        tiles=val_manifest,
        crop_size=current_crop_size,
        is_train=False,
        cache_dir=cache_dir,
        norm_stats=(norm_mean, norm_std),
    )

    class_pixel_counts = train_dataset.damage_class_pixel_counts()
    class_weights = inverse_sqrt_class_weights(class_pixel_counts)
    print(
        "[INFO] Training damage pixel counts "
        f"({', '.join(DAMAGE_SEVERITY_CLASS_NAMES)}): {class_pixel_counts.tolist()}"
    )
    print(f"[INFO] Normalized inverse-sqrt class weights (capped at 10): {class_weights.tolist()}")
    print("[INFO] Sampling: shuffled training tiles only; no weighted oversampling.")

    def build_train_loader() -> DataLoader:
        return DataLoader(
            train_dataset,
            batch_size=profile.batch_size,
            shuffle=True,
            num_workers=profile.num_workers,
            pin_memory=profile.pin_memory,
            drop_last=True if len(train_dataset) >= profile.batch_size else False,
        )

    def build_val_loader() -> DataLoader:
        return DataLoader(
            val_dataset,
            batch_size=max(1, profile.batch_size),
            shuffle=False,
            num_workers=profile.num_workers,
            pin_memory=profile.pin_memory,
        )

    train_loader = build_train_loader()
    val_loader = build_val_loader()
    batches_per_epoch = len(train_loader)
    optimizer_steps_per_epoch = math.ceil(
        batches_per_epoch / profile.grad_accum_steps
    )
    planned_total_optimizer_steps = optimizer_steps_per_epoch * args.epochs

    # 4. Initialize Model, Loss, Optimizer, Scheduler, Scaler
    print(f"\n[INFO] Initializing Siamese U-Net with shared '{profile.encoder}' encoder...")
    model = SiameseUNet(
        encoder_name=profile.encoder,
        encoder_weights=None if args.no_pretrained else "imagenet",
        num_classes=4,
    ).to(device)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    trainable_parameter_count = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )

    if profile.channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = RubiconLoss(
        class_weights=class_weights,
        loc_weight=0.4,
        dmg_weight=0.6,
    ).to(device)
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
        checkpoint = load_checkpoint(args.resume, map_location=device)
        model.load_state_dict(checkpoint["state_dict"])
        if checkpoint.get("optimizer_state_dict"):
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if checkpoint.get("scheduler_state_dict"):
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])
        if scaler is not None and checkpoint.get("scaler_state_dict"):
            scaler.load_state_dict(checkpoint["scaler_state_dict"])
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        previous = checkpoint.get("metrics") or {}
        best_score = float(previous.get("xview2_score", checkpoint.get("xview2_score", -1.0)))
        if checkpoint.get("norm_mean"):
            norm_mean = [float(v) for v in checkpoint["norm_mean"]]
            norm_std = [float(v) for v in checkpoint["norm_std"]]
            train_dataset.mean = np.asarray(norm_mean, dtype=np.float32).reshape(1, 1, 3)
            train_dataset.std = np.asarray(norm_std, dtype=np.float32).reshape(1, 1, 3)
            val_dataset.mean = train_dataset.mean
            val_dataset.std = train_dataset.std
        print(f"[INFO] Resumed from epoch {start_epoch} with best xView2 score: {best_score:.4f}")
    elif args.resume:
        print(f"[WARN] Checkpoint not found at {args.resume}; starting from scratch.")

    # Output and Logging setup
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=args.log_dir)

    epoch_durations = []
    history = []
    collapse_warnings = []
    best_epoch = None
    collapse_warning = None
    stop_reason = None
    total_optimizer_steps_completed = 0

    print("\n" + "=" * 65)
    print(f"STARTING TRAINING: {len(train_dataset)} Train Tiles | {len(val_dataset)} Val Tiles | {total_epochs} Epochs")
    print(f"Device: {device} | AMP: {profile.use_amp} | Effective Batch: {profile.batch_size * profile.grad_accum_steps}")
    print(
        f"Optimization steps: {optimizer_steps_per_epoch} per epoch "
        f"({batches_per_epoch} batches / {profile.grad_accum_steps} accumulation) | "
        f"{planned_total_optimizer_steps} planned total"
    )
    print("=" * 65 + "\n")

    # 6. Training Loop
    for epoch in range(start_epoch, total_epochs):
        t_epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        running_loc_loss = 0.0
        running_dmg_loss = 0.0
        num_batches = 0
        epoch_optimizer_steps = 0

        optimizer.zero_grad(set_to_none=True)

        processed = 0
        while processed < len(train_loader):
            try:
                for step, (pre, post, loc_mask, dmg_mask) in enumerate(train_loader):
                    if step < processed:
                        continue

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
                        epoch_optimizer_steps += 1
                        total_optimizer_steps_completed += 1

                    running_loss += breakdown["total_loss"]
                    running_loc_loss += breakdown["loc_loss"]
                    running_dmg_loss += breakdown["dmg_loss"]
                    num_batches += 1
                    processed = step + 1

            except RuntimeError as err:
                is_oom = isinstance(err, torch.cuda.OutOfMemoryError) or "out of memory" in str(err).lower()
                if not is_oom:
                    raise
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                if current_crop_size <= 256:
                    raise RuntimeError(
                        "CUDA OOM persists at the minimum crop size (256); "
                        "reduce batch size or use the Kaggle profile."
                    ) from err
                new_crop = 384 if current_crop_size >= 512 else 256
                print(
                    f"\n[ALERT] CUDA OOM at batch {processed}/{len(train_loader)}. "
                    f"Reducing crop {current_crop_size} -> {new_crop} and retrying."
                )
                current_crop_size = new_crop
                train_dataset.set_crop_size(new_crop)
                val_dataset.set_crop_size(new_crop)
                train_loader = build_train_loader()
                val_loader = build_val_loader()
                optimizer.zero_grad(set_to_none=True)

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
                    _, v_breakdown = criterion(loc_logits, dmg_logits, loc_mask, dmg_mask)

                val_running_loss += v_breakdown["total_loss"]
                val_batches += 1

                # Predictions
                loc_probs = torch.sigmoid(loc_logits).cpu().numpy()
                dmg_preds = decode_damage_logits(dmg_logits).cpu().numpy()

                val_metrics_meter.update(
                    loc_preds=loc_probs,
                    loc_targets=loc_mask.cpu().numpy(),
                    dmg_preds=dmg_preds,
                    dmg_targets=dmg_mask.cpu().numpy(),
                )

        avg_val_loss = val_running_loss / max(1, val_batches)
        metrics = val_metrics_meter.compute()
        xview2_score = metrics["xview2_score"]
        history.append({
            "epoch": epoch + 1,
            "train_loss": avg_train_loss,
            "train_localization_loss": avg_loc_loss,
            "train_damage_loss": avg_dmg_loss,
            "validation_loss": avg_val_loss,
            "validation_metrics": metrics,
            "optimizer_steps": epoch_optimizer_steps,
            "duration_sec": epoch_dur,
        })

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
            f"Dmg Macro F1: {metrics['damage_f1_macro']:.4f} | "
            f"xView2 Score: {xview2_score:.4f} | "
            f"Optimizer Steps: {epoch_optimizer_steps} | "
            f"Time: {epoch_dur:.1f}s (ETA: {est_remaining_sec}s)"
        )
        print("  Validation damage precision / recall / F1:")
        for name in DAMAGE_CLASS_NAMES:
            print(
                f"    {name:<16}: "
                f"P={metrics['damage_precision_classes'][name]:.4f} "
                f"R={metrics['damage_recall_classes'][name]:.4f} "
                f"F1={metrics['damage_f1_classes'][name]:.4f}"
            )
        histogram = metrics["predicted_building_class_histogram"]
        histogram_pct = metrics["predicted_building_class_histogram_pct"]
        predicted_building_pixels = metrics["predicted_building_pixels"]
        histogram_summary = ", ".join(
            f"{name}={count} ({histogram_pct[name]:.2f}%)"
            for name, count in histogram.items()
        )
        print(
            f"  Validation predicted-building class histogram "
            f"({predicted_building_pixels} pixels): {histogram_summary}"
        )
        ground_truth_histogram = metrics["damage_ground_truth_class_histogram"]
        ground_truth_histogram_pct = metrics["damage_ground_truth_class_histogram_pct"]
        ground_truth_summary = ", ".join(
            f"{name}={ground_truth_histogram[name]} "
            f"({ground_truth_histogram_pct[name]:.2f}%)"
            for name in DAMAGE_CLASS_NAMES
        )
        print(
            f"  Validation ground-truth building class histogram "
            f"({metrics['ground_truth_building_pixels']} pixels): "
            f"{ground_truth_summary}"
        )

        # TensorBoard Logging
        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Loss/Val", avg_val_loss, epoch)
        writer.add_scalar("xView2/Score", xview2_score, epoch)
        writer.add_scalar("xView2/Loc_F1", metrics["loc_f1"], epoch)
        writer.add_scalar("xView2/Damage_F1_Harmonic", metrics["damage_f1_harmonic"], epoch)
        writer.add_scalar("xView2/Damage_F1_Macro", metrics["damage_f1_macro"], epoch)
        for dmg_cls, val_f1 in metrics["damage_f1_classes"].items():
            writer.add_scalar(f"Damage_F1/{dmg_cls}", val_f1, epoch)
            writer.add_scalar(
                f"Damage_Precision/{dmg_cls}",
                metrics["damage_precision_classes"][dmg_cls],
                epoch,
            )
            writer.add_scalar(
                f"Damage_Recall/{dmg_cls}",
                metrics["damage_recall_classes"][dmg_cls],
                epoch,
            )
            writer.add_scalar(
                f"Damage_Ground_Truth_Pixels/{dmg_cls}",
                ground_truth_histogram[dmg_cls],
                epoch,
            )

        # 8. Checkpoint Saving (Every Epoch: last and best)
        metrics_payload = dict(metrics)
        metrics_payload["xview2_score"] = xview2_score

        def write_checkpoint(path: Path, include_training_state: bool = False) -> None:
            save_checkpoint(
                path,
                model,
                model_version=model_version,
                encoder_name=profile.encoder,
                num_classes=4,
                crop_size=current_crop_size,
                norm_mean=norm_mean,
                norm_std=norm_std,
                damage_classes=DAMAGE_SEVERITY_CLASS_NAMES,
                epoch=epoch,
                metrics=metrics_payload,
                optimizer=optimizer if include_training_state else None,
                scheduler=scheduler if include_training_state else None,
                scaler=scaler if include_training_state else None,
                hardware_profile=profile.name,
                extra_config={
                    "train_tiles": len(train_dataset),
                    "val_tiles": len(val_dataset),
                    "val_events": split_meta.get("val_events", []),
                    "train_events": split_meta.get("train_events", []),
                    "source_split": split_meta.get("source_split"),
                    "data_mode": data_mode,
                    "optimizer": "AdamW",
                    "learning_rate": args.lr,
                    "weight_decay": 1e-4,
                    "scheduler": "linear_warmup_cosine",
                    "warmup_epochs": warmup_epochs,
                    "localization_loss_weight": 0.4,
                    "damage_loss_weight": 0.6,
                    "focal_gamma": 2.0,
                    "damage_class_pixel_counts": class_pixel_counts.tolist(),
                    "damage_class_weights": class_weights.cpu().tolist(),
                    "damage_sampling": "shuffle_only_no_weighted_oversampling",
                    "batch_size": profile.batch_size,
                    "gradient_accumulation_steps": profile.grad_accum_steps,
                    "parameter_count": parameter_count,
                    "trainable_parameter_count": trainable_parameter_count,
                    "pretrained_encoder_weights": not args.no_pretrained,
                },
            )

        # Save last checkpoint (resumable) every epoch
        last_ckpt_path = out_dir / "checkpoint_last.pt"
        write_checkpoint(last_ckpt_path, include_training_state=True)

        # Save best checkpoint
        if xview2_score > best_score:
            best_score = xview2_score
            best_epoch = epoch + 1
            patience_counter = 0
            best_ckpt_path = out_dir / "checkpoint_best.pt"
            write_checkpoint(best_ckpt_path)
            # Also mirror the serving artifact expected by ai_engine.model.sat_serving
            write_checkpoint(out_dir / "siamese_unet.pt")
            write_checkpoint(out_dir / "model.pt")
            print(f"  [*] New best xView2 score achieved: {best_score:.4f}! Checkpoint saved.")
        else:
            patience_counter += 1

        if epoch + 1 >= 2 and predicted_building_pixels > 0:
            dominant_class, dominant_count = max(histogram.items(), key=lambda item: item[1])
            dominant_fraction = dominant_count / predicted_building_pixels
            if dominant_fraction > 0.90:
                collapse_warning = (
                    f"epoch {epoch + 1}: {dominant_class} accounts for "
                    f"{dominant_fraction:.1%} of predicted building pixels"
                )
                collapse_warnings.append(collapse_warning)
                print(
                    f"[WARN] Damage-class collapse detected: {collapse_warning}. "
                    "Continuing training; this warning never stops the run."
                )

        if patience_counter >= args.patience:
            stop_reason = "validation_score_patience"
            print(f"\n[INFO] Early stopping triggered after {args.patience} epochs without score improvement.")
            break

    writer.close()

    best_checkpoint_path = out_dir / "checkpoint_best.pt"
    training_summary = {
        "data_mode": data_mode,
        "xbd_root": str(Path(xbd_root).resolve()),
        "source_split": split_meta.get("source_split"),
        "model_version": model_version,
        "encoder": profile.encoder,
        "num_classes": 4,
        "class_names": DAMAGE_SEVERITY_CLASS_NAMES,
        "hardware_profile": profile.name,
        "device": str(device),
        "crop_size": current_crop_size,
        "batch_size": profile.batch_size,
        "gradient_accumulation_steps": profile.grad_accum_steps,
        "effective_batch_size": profile.batch_size * profile.grad_accum_steps,
        "batches_per_epoch": batches_per_epoch,
        "optimizer_steps_per_epoch": optimizer_steps_per_epoch,
        "planned_total_optimizer_steps": planned_total_optimizer_steps,
        "total_optimizer_steps_completed": total_optimizer_steps_completed,
        "mixed_precision": bool(profile.use_amp and device.type == "cuda"),
        "optimizer": "AdamW",
        "learning_rate": args.lr,
        "weight_decay": 1e-4,
        "scheduler": "linear_warmup_cosine",
        "warmup_epochs": warmup_epochs,
        "localization_loss_weight": 0.4,
        "damage_loss_weight": 0.6,
        "focal_gamma": 2.0,
        "damage_class_pixel_counts": class_pixel_counts.tolist(),
        "damage_class_weights": class_weights.cpu().tolist(),
        "damage_sampling": "shuffle_only_no_weighted_oversampling",
        "pretrained_encoder_weights": not args.no_pretrained,
        "parameter_count": parameter_count,
        "trainable_parameter_count": trainable_parameter_count,
        "epochs_requested": total_epochs,
        "epochs_completed": len(history),
        "best_epoch": best_epoch,
        "best_validation_xview2_score": best_score,
        "collapse_warning": collapse_warning,
        "collapse_warnings": collapse_warnings,
        "stop_reason": stop_reason,
        "training_tiles": len(train_dataset),
        "validation_tiles": len(val_dataset),
        "training_events": split_meta.get("train_events", []),
        "validation_events": split_meta.get("val_events", []),
        "split_warnings": split_meta.get("warnings", []),
        "checkpoint": str(best_checkpoint_path.resolve()),
        "checkpoint_hash": (
            get_checkpoint_sha256(best_checkpoint_path)
            if best_checkpoint_path.is_file()
            else None
        ),
        "duration_sec": round(time.perf_counter() - run_started, 2),
        "optimizer_steps_per_epoch_history": [
            item["optimizer_steps"] for item in history
        ],
        "history": history,
    }
    summary_path = out_dir / "training_metrics.json"
    summary_path.write_text(json.dumps(training_summary, indent=2), encoding="utf-8")

    # Final summary report
    print("\n" + "=" * 65)
    print("TRAINING RUN COMPLETE")
    print(f"  Best xView2 Score Achieved: {best_score:.4f}")
    print(f"  Trainable Parameters:       {trainable_parameter_count:,}")
    print(
        f"  Optimizer Steps:            {total_optimizer_steps_completed} completed / "
        f"{planned_total_optimizer_steps} planned"
    )
    print(f"  Training Summary:           {summary_path.resolve()}")
    if epoch_durations:
        avg_s = sum(epoch_durations) / len(epoch_durations)
        print(f"  Measured Seconds Per Epoch:  {avg_s:.2f} s/epoch")
        full_run_30_est = round(avg_s * 30 / 60, 1)
        print(f"  Estimated Full Run (30 ep):  {full_run_30_est} minutes")
    print(f"  Checkpoints saved in:        {out_dir.resolve()}")
    print("=" * 65 + "\n")
    return training_summary


if __name__ == "__main__":
    train()
