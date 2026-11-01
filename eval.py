"""Rubicon Model Evaluation Pipeline.

Usage:
    python eval.py [--checkpoint PATH] [--output-dir docs/results] [--xbd-root PATH] [--synthetic]

Outputs:
    - docs/results/metrics.json: Official xView2 metrics and damage breakdown
    - docs/results/confusion_matrix.png: Normalized confusion matrix heatmap
    - docs/results/eval_overlay_1.png .. eval_overlay_10.png: 10 qualitative evaluation overlays
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any, List

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

from ai_engine.model.metrics import XView2Metrics, DAMAGE_CLASS_NAMES
from ai_engine.model.hardware import detect_hardware
from ai_engine.model.checkpoint import (
    build_model_from_checkpoint,
    load_checkpoint,
    norm_stats_from_checkpoint,
)
from ai_engine.data.splits import discover_xbd_tiles, load_or_create_splits, get_xbd_root
from ai_engine.data.dataset import XBDDataset
from ai_engine.data.synthetic import generate_synthetic_dataset


def create_eval_overlay(
    pre_path: str,
    post_path: str,
    gt_loc: np.ndarray,
    gt_dmg: np.ndarray,
    pred_loc: np.ndarray,
    pred_dmg: np.ndarray,
    tile_id: str,
) -> np.ndarray:
    """Create a 4-panel visual comparison: Pre, Post, Ground Truth Damage, Predicted Damage."""
    pre_bgr = cv2.imread(pre_path)
    post_bgr = cv2.imread(post_path)
    if pre_bgr is None or post_bgr is None:
        raise FileNotFoundError(f"Cannot read overlay images for tile {tile_id}")
    h, w = pre_bgr.shape[:2]

    # Resize predictions/ground truth if needed
    if pred_dmg.shape != (h, w):
        pred_dmg = cv2.resize(pred_dmg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)
        pred_loc = cv2.resize(pred_loc.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
    if gt_dmg.shape != (h, w):
        gt_dmg = cv2.resize(gt_dmg.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST)

    def colorize_damage(base_img: np.ndarray, dmg_mask: np.ndarray) -> np.ndarray:
        res = base_img.copy()
        color_layer = np.zeros_like(base_img)
        color_layer[dmg_mask == 1] = [46, 204, 113]   # No damage (Green)
        color_layer[dmg_mask == 2] = [0, 215, 255]    # Minor damage (Yellow)
        color_layer[dmg_mask == 3] = [0, 140, 255]    # Major damage (Orange)
        color_layer[dmg_mask == 4] = [36, 40, 231]    # Destroyed (Red)

        has_dmg = dmg_mask > 0
        blend = res.copy()
        cv2.addWeighted(color_layer, 0.55, res, 0.45, 0, blend)
        res[has_dmg] = blend[has_dmg]

        # Draw contour boundaries
        for cls_id in (1, 2, 3, 4):
            m = (dmg_mask == cls_id).astype(np.uint8)
            if m.sum() > 0:
                cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                c = [46, 204, 113] if cls_id == 1 else (
                    [0, 215, 255] if cls_id == 2 else (
                        [0, 140, 255] if cls_id == 3 else [36, 40, 231]
                    )
                )
                cv2.drawContours(res, cnts, -1, c, 1)
        return res

    gt_panel = colorize_damage(post_bgr, gt_dmg)
    pred_panel = colorize_damage(post_bgr, pred_dmg)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.75
    thick = 2

    def add_title(img, title, color=(255, 255, 255)):
        r = img.copy()
        cv2.rectangle(r, (0, 0), (w, 38), (15, 15, 15), -1)
        cv2.putText(r, title, (12, 26), font, scale, color, thick, cv2.LINE_AA)
        return r

    p1 = add_title(pre_bgr, f"PRE-DISASTER ({tile_id})")
    p2 = add_title(post_bgr, "POST-DISASTER OPTICAL")
    p3 = add_title(gt_panel, "GROUND TRUTH LABELS")
    p4 = add_title(pred_panel, "SIAMESE U-NET PREDICTION")

    top_row = np.hstack([p1, p2])
    bot_row = np.hstack([p3, p4])
    grid = np.vstack([top_row, bot_row])
    return cv2.resize(grid, (1024, 1024), interpolation=cv2.INTER_AREA)


def evaluate():
    parser = argparse.ArgumentParser(description="Evaluate Rubicon Disaster Model on Hold-Out Dataset")
    parser.add_argument("--checkpoint", type=str, default="data/models/checkpoint_best.pt", help="Path to checkpoint")
    parser.add_argument("--output-dir", type=str, default="docs/results", help="Directory to save eval artifacts")
    parser.add_argument("--xbd-root", type=str, default=None, help="Root path to xBD dataset")
    parser.add_argument("--synthetic", action="store_true", help="Evaluate in synthetic mode")
    parser.add_argument(
        "--split",
        choices=("val", "hold", "test"),
        default="val",
        help="Evaluate event-disjoint validation from train/, or a labeled hold/ or test/ split",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Validation batch size")
    parser.add_argument("--max-tiles", type=int, default=None, help="Cap the number of validation tiles")
    parser.add_argument("--cache-dir", type=str, default="data/xbd_cache", help="Directory for cached rasterized masks")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.is_file():
        # Fallback to checkpoint_last if best not found
        fallback_ckpt = ckpt_path.parent / "checkpoint_last.pt"
        if fallback_ckpt.is_file():
            ckpt_path = fallback_ckpt
        else:
            raise FileNotFoundError(f"Checkpoint not found at '{args.checkpoint}' or '{fallback_ckpt}'.")

    # 1. Device and checkpoint loading
    profile = detect_hardware()
    device = torch.device(profile.device)

    print(f"\n[INFO] Loading checkpoint from '{ckpt_path.resolve()}'...")
    checkpoint = load_checkpoint(ckpt_path, map_location=device, expected_num_classes=5)
    cfg = checkpoint.get("config", {})
    encoder_name = checkpoint.get("encoder_name", cfg.get("encoder", "resnet34"))
    model_version = checkpoint.get("model_version", "siamese_unet-v1")

    model = build_model_from_checkpoint(checkpoint, device=device)
    norm_mean, norm_std = norm_stats_from_checkpoint(checkpoint)
    print(f"[INFO] Model: {model_version} | encoder={encoder_name} | "
          f"hash={checkpoint.get('state_dict_hash', 'n/a')}")
    print(f"[INFO] Normalization mean={norm_mean} std={norm_std}")

    # 2. Dataset resolution
    if args.synthetic:
        xbd_root = generate_synthetic_dataset()
    else:
        xbd_root = get_xbd_root(args.xbd_root)

    splits = load_or_create_splits(xbd_root=xbd_root)
    split_meta = splits.get("metadata", {})
    if args.split == "val":
        eval_manifest = splits["val"]
        eval_events = split_meta.get("val_events", [])
    else:
        eval_manifest = discover_xbd_tiles(xbd_root, split=args.split)
        eval_events = sorted({tile["event"] for tile in eval_manifest})
        if not eval_manifest:
            raise ValueError(
                f"No labeled xBD tiles found in split '{args.split}' under {xbd_root}."
            )
    if args.max_tiles is not None:
        eval_manifest = eval_manifest[: max(0, args.max_tiles)]

    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    eval_dataset = XBDDataset(
        tiles=eval_manifest,
        crop_size=cfg.get("crop_size", 512),
        is_train=False,
        cache_dir=cache_dir,
        norm_stats=(norm_mean, norm_std),
    )
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )

    if args.split == "val":
        print(f"[INFO] Evaluating on {len(eval_manifest)} event-disjoint validation tiles "
              f"({', '.join(eval_events)})...")
        for warning in split_meta.get("warnings", []):
            print(f"[WARN] {warning}")
    else:
        print(f"[INFO] Evaluating on {len(eval_manifest)} labeled '{args.split}' split tiles "
              f"({', '.join(eval_events)})...")
        if args.split == "hold":
            print(
                "[WARN] xBD hold/ shares disaster event names with train/; "
                "treat these tile-level hold metrics as supplemental, not the "
                "primary event-disjoint generalization estimate."
            )

    # 3. Evaluation loop
    metrics_meter = XView2Metrics()
    overlay_candidates = []

    with torch.no_grad():
        for b_idx, (pre, post, loc_mask, dmg_mask) in enumerate(eval_loader):
            pre = pre.to(device)
            post = post.to(device)

            loc_logits, dmg_logits = model(pre, post)

            loc_probs = torch.sigmoid(loc_logits).cpu().numpy()
            dmg_preds = torch.argmax(dmg_logits, dim=1).cpu().numpy()

            loc_gt = loc_mask.cpu().numpy()
            dmg_gt = dmg_mask.cpu().numpy()

            metrics_meter.update(
                loc_preds=loc_probs,
                loc_targets=loc_gt,
                dmg_preds=dmg_preds,
                dmg_targets=dmg_gt,
            )

            # Collect for qualitative overlays
            for i in range(len(pre)):
                tile_idx = b_idx * args.batch_size + i
                if tile_idx < len(eval_manifest):
                    tile_info = eval_manifest[tile_idx]
                    overlay_candidates.append({
                        "tile": tile_info,
                        "pred_loc": loc_probs[i, 0],
                        "pred_dmg": dmg_preds[i],
                        "gt_loc": loc_gt[i, 0],
                        "gt_dmg": dmg_gt[i],
                    })

    # 4. Compute metrics
    metrics = metrics_meter.compute()

    print("\n" + "=" * 65)
    print(f"{args.split.upper()} EVALUATION RESULTS")
    print("=" * 65)
    print(f"  Official xView2 Score:       {metrics['xview2_score']:.4f}")
    print(f"  Localization F1:             {metrics['loc_f1']:.4f}")
    print(f"  Damage F1 (Harmonic Mean):   {metrics['damage_f1_harmonic']:.4f}")
    print("  Per-Class Damage F1:")
    for cls_name, f1_val in metrics["damage_f1_classes"].items():
        print(f"    - {cls_name:<16}: {f1_val:.4f}")
    print("=" * 65)

    # 5. Save metrics JSON
    metrics_path = out_dir / "metrics.json"
    metrics_payload = {
        "checkpoint": str(ckpt_path.resolve()),
        "checkpoint_hash": checkpoint.get("state_dict_hash"),
        "model_version": model_version,
        "encoder": encoder_name,
        "class_names": checkpoint.get("damage_classes", DAMAGE_CLASS_NAMES),
        "data_mode": "synthetic" if args.synthetic else "real_xbd",
        "evaluation_split": args.split,
        "evaluation_events": eval_events,
        "num_evaluation_tiles": len(eval_manifest),
        "source_split": split_meta.get("source_split"),
        "train_events": split_meta.get("train_events", []),
        "validation_events": split_meta.get("val_events", []),
        "num_validation_tiles": len(splits.get("val", [])),
        # Retained for existing notebook consumers; historically these names
        # referred to the event-disjoint validation subset.
        "holdout_events": split_meta.get("val_events", []),
        "num_holdout_tiles": len(splits.get("val", [])),
        "split_warnings": split_meta.get("warnings", []),
        "results": metrics,
    }
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    print(f"\n[+] Saved metrics JSON to: {metrics_path}")

    # 6. Save confusion matrix plot
    cm_plot_path = out_dir / "confusion_matrix.png"
    metrics_meter.plot_confusion_matrix(output_path=str(cm_plot_path))
    print(f"[+] Saved confusion matrix to: {cm_plot_path}")

    # 7. Generate 10 qualitative overlays
    print(f"[INFO] Generating up to 10 qualitative overlays in '{out_dir}'...")
    num_overlays = min(10, len(overlay_candidates))
    for idx in range(num_overlays):
        cand = overlay_candidates[idx]
        overlay_img = create_eval_overlay(
            pre_path=cand["tile"]["pre_image"],
            post_path=cand["tile"]["post_image"],
            gt_loc=cand["gt_loc"],
            gt_dmg=cand["gt_dmg"],
            pred_loc=cand["pred_loc"],
            pred_dmg=cand["pred_dmg"],
            tile_id=cand["tile"]["tile_id"],
        )
        overlay_file = out_dir / f"eval_overlay_{idx+1}.png"
        cv2.imwrite(str(overlay_file), overlay_img)
        print(f"  [+] Saved overlay {idx+1}/{num_overlays}: {overlay_file.name}")

    print("\n" + "=" * 65)
    print("PHASE 3 EVALUATION COMPLETE")
    print("=" * 65)


if __name__ == "__main__":
    evaluate()
