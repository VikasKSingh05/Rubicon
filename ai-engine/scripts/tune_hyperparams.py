"""Hyperparameter Tuning Pipeline for Rubicon Siamese U-Net.

Systematically evaluates architectural and optimization hyperparameters on xBD:
  - Encoder Backbones: resnet18 vs resnet34
  - Learning Rate: [1e-4, 3e-4, 5e-4]
  - Task Loss Alpha (loc_weight): [0.3, 0.4, 0.5] (dmg_weight = 1 - alpha)
  - Focal Loss Gamma: [1.5, 2.0, 2.5]
  - Weight Decay: [1e-4, 1e-3]

Objective:
  Maximize Validation xView2 Metric:
  F1_total = 0.3 * F1_loc + 0.7 * F1_dmg

Usage:
  python scripts/tune_hyperparams.py [--trials 5] [--synthetic] [--dry-run]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Ensure workspace root and ai-engine root are in sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from ai_engine.model.hardware import detect_hardware
from ai_engine.model.siamese_unet import SiameseUNet
from ai_engine.model.losses import RubiconLoss
from ai_engine.model.metrics import XView2Metrics
from ai_engine.data.splits import load_or_create_splits, get_xbd_root
from ai_engine.data.dataset import XBDDataset, compute_norm_stats
from ai_engine.data.synthetic import generate_synthetic_dataset


# Hyperparameter search space
SEARCH_SPACE = {
    "encoder_name": ["resnet34", "resnet18"],
    "lr": [1e-4, 3e-4, 5e-4],
    "loc_weight": [0.3, 0.4, 0.5],
    "focal_gamma": [1.5, 2.0, 2.5],
    "weight_decay": [1e-4, 1e-3],
}


def sample_configurations(n: int, seed: int = 42) -> List[Dict[str, Any]]:
    """Sample n unique hyperparameter configurations from search space."""
    random.seed(seed)
    keys = list(SEARCH_SPACE.keys())
    all_combos = list(itertools.product(*[SEARCH_SPACE[k] for k in keys]))
    random.shuffle(all_combos)
    
    configs = []
    for combo in all_combos[:n]:
        cfg = dict(zip(keys, combo))
        cfg["dmg_weight"] = round(1.0 - cfg["loc_weight"], 2)
        configs.append(cfg)
    return configs


def evaluate_trial(
    trial_idx: int,
    cfg: Dict[str, Any],
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    max_batches: int = 5,
) -> Dict[str, Any]:
    """Train for a small burst and evaluate validation xView2 score."""
    print(f"\n--- [TRIAL {trial_idx+1}] Config: {cfg} ---")
    
    # Instantiate model with candidate encoder
    model = SiameseUNet(encoder_name=cfg["encoder_name"], num_classes=5).to(device)
    loss_fn = RubiconLoss(
        loc_weight=cfg["loc_weight"],
        dmg_weight=cfg["dmg_weight"],
        focal_gamma=cfg["focal_gamma"],
    ).to(device)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )
    
    # 1. Mini training run
    model.train()
    total_train_loss = 0.0
    steps = 0
    start_t = time.time()
    
    for i, batch in enumerate(train_loader):
        if i >= max_batches:
            break
        pre = batch["pre"].to(device, dtype=torch.float32)
        post = batch["post"].to(device, dtype=torch.float32)
        loc_tgt = batch["loc_mask"].to(device, dtype=torch.float32)
        dmg_tgt = batch["dmg_mask"].to(device, dtype=torch.long)
        
        optimizer.zero_grad()
        loc_logits, dmg_logits = model(pre, post)
        loss, loc_l, dmg_l = loss_fn(loc_logits, dmg_logits, loc_tgt, dmg_tgt)
        loss.backward()
        optimizer.step()
        
        total_train_loss += loss.item()
        steps += 1
        
    avg_train_loss = total_train_loss / max(1, steps)
    
    # 2. Validation evaluation
    model.eval()
    metric_evaluator = XView2Metrics(num_classes=5)
    with torch.no_grad():
        for j, batch in enumerate(val_loader):
            if j >= max_batches:
                break
            pre = batch["pre"].to(device, dtype=torch.float32)
            post = batch["post"].to(device, dtype=torch.float32)
            loc_tgt = batch["loc_mask"].to(device, dtype=torch.float32)
            dmg_tgt = batch["dmg_mask"].to(device, dtype=torch.long)
            
            loc_logits, dmg_logits = model(pre, post)
            metric_evaluator.update(loc_logits, dmg_logits, loc_tgt, dmg_tgt)
            
    metrics = metric_evaluator.compute()
    duration = time.time() - start_t
    
    xview2_score = metrics.get("xview2_total_f1", 0.0)
    loc_f1 = metrics.get("loc_f1", 0.0)
    dmg_f1 = metrics.get("dmg_f1", 0.0)
    
    print(f"  Result -> xView2 Score: {xview2_score:.4f} (Loc F1: {loc_f1:.4f}, Dmg F1: {dmg_f1:.4f}) | Duration: {duration:.1f}s")
    
    return {
        "trial": trial_idx + 1,
        "config": cfg,
        "train_loss": round(avg_train_loss, 4),
        "xview2_score": round(xview2_score, 4),
        "loc_f1": round(loc_f1, 4),
        "dmg_f1": round(dmg_f1, 4),
        "duration_sec": round(duration, 2),
    }


def main():
    parser = argparse.ArgumentParser(description="Rubicon Hyperparameter Tuning")
    parser.add_argument("--trials", type=int, default=3, help="Number of hyperparameter configurations to sample")
    parser.add_argument("--synthetic", action="store_true", help="Tune using procedural synthetic dataset")
    parser.add_argument("--dry-run", action="store_true", help="Quick test with 2 batches per trial")
    parser.add_argument("--xbd-root", type=str, default=None, help="Root path of xBD dataset")
    parser.add_argument("--output", type=str, default="runs/hyperparam_tuning_results.json", help="Path to save trial log")
    args = parser.parse_args()

    print("\n=======================================================")
    print("  RUBICON SIAMESE U-NET HYPERPARAMETER TUNING")
    print("=======================================================")
    
    # Detect hardware & device
    hw = detect_hardware()
    device = torch.device(hw.device)
    print(f"[Device] Hardware profile: {hw.name} | Using device: {device}")
    
    # Resolve dataset
    if args.synthetic:
        xbd_root = generate_synthetic_dataset()
    else:
        try:
            xbd_root = get_xbd_root(args.xbd_root)
        except FileNotFoundError:
            print("[INFO] No xBD root found. Auto-generating synthetic procedural dataset for tuning...")
            xbd_root = generate_synthetic_dataset()
            
    splits = load_or_create_splits(xbd_root)
    train_manifest = splits["train"]
    val_manifest = splits["val"]
    
    if not train_manifest or not val_manifest:
        print("[ERROR] Train or validation manifest is empty.")
        return

    # Small crop size for rapid hyperparameter convergence
    crop_size = 256
    norm_mean, norm_std = compute_norm_stats(train_manifest[:10])
    
    train_dataset = XBDDataset(train_manifest, crop_size=crop_size, is_train=True, norm_stats=(norm_mean, norm_std))
    val_dataset = XBDDataset(val_manifest, crop_size=crop_size, is_train=False, norm_stats=(norm_mean, norm_std))
    
    train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=2, shuffle=False)
    
    # Sample configurations
    configs = sample_configurations(args.trials)
    max_batches = 2 if args.dry_run else 5
    
    results = []
    for idx, cfg in enumerate(configs):
        res = evaluate_trial(idx, cfg, train_loader, val_loader, device, max_batches=max_batches)
        results.append(res)
        
    # Rank results by xview2_score descending
    results.sort(key=lambda x: x["xview2_score"], reverse=True)
    best_trial = results[0]
    
    print("\n=======================================================")
    print("  HYPERPARAMETER TUNING LEADERBOARD")
    print("=======================================================")
    for rank, r in enumerate(results, 1):
        cfg_str = f"lr={r['config']['lr']}, enc={r['config']['encoder_name']}, alpha={r['config']['loc_weight']}, gamma={r['config']['focal_gamma']}"
        print(f" #{rank} | xView2: {r['xview2_score']:.4f} (Loc: {r['loc_f1']:.3f}, Dmg: {r['dmg_f1']:.3f}) | {cfg_str}")
        
    print("\n-------------------------------------------------------")
    print(f" OPTIMAL CONFIGURATION (Trial {best_trial['trial']}):")
    print(f"  Encoder:     {best_trial['config']['encoder_name']}")
    print(f"  Learning Rate: {best_trial['config']['lr']}")
    print(f"  Loc / Dmg:   {best_trial['config']['loc_weight']} / {best_trial['config']['dmg_weight']}")
    print(f"  Focal Gamma: {best_trial['config']['focal_gamma']}")
    print(f"  Weight Decay:{best_trial['config']['weight_decay']}")
    print(f"  Best Score:  {best_trial['xview2_score']}")
    print("=======================================================\n")
    
    # Save results
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({
            "trials_count": len(results),
            "best_config": best_trial["config"],
            "best_score": best_trial["xview2_score"],
            "trials": results,
        }, f, indent=2)
    print(f"[Results] Tuning results saved to {out_path.resolve()}\n")


if __name__ == "__main__":
    main()
