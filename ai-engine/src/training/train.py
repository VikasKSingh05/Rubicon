"""Model training CLI for Phase 3.

Reads the Phase 2 npz (X_hsi, X_dem, y_sev, val_mask, test_mask), trains the
configured architecture, and writes a checkpoint + metrics beside a copy of the
fitted PCA so inference only needs the output directory.

Usage:
    python -m src.training.train --config configs/phase3.yaml
"""
from __future__ import annotations

import argparse
import json
import pickle
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Dataset, Subset
from torch.utils.data import WeightedRandomSampler

import yaml

from ..model.factory import create_model, model_version


class NpzDataset(Dataset):
    """Channels-first Phase 2 tensors -> tensors."""

    def __init__(self, npz: dict):
        self.x_hsi = torch.from_numpy(npz["X_hsi"])
        self.x_dem = torch.from_numpy(npz["X_dem"])
        self.y = torch.from_numpy(npz["y_sev"])

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.x_hsi[i], self.x_dem[i], self.y[i]


def make_loaders(npz: dict, batch_size: int, seed: int):
    ds = NpzDataset(npz)
    train_idx = np.flatnonzero(~npz["val_mask"] & ~npz["test_mask"])
    val_idx = np.flatnonzero(npz["val_mask"])
    test_idx = np.flatnonzero(npz["test_mask"])

    train_sev = npz["y_sev"][train_idx]
    weights = 1.0 / np.bincount(train_sev, minlength=3).astype(np.float64)
    sampler = WeightedRandomSampler(
        torch.from_numpy(weights[train_sev]), num_samples=len(train_idx),
        replacement=True, generator=torch.Generator().manual_seed(seed),
    )
    tr = DataLoader(Subset(ds, train_idx), batch_size=batch_size, sampler=sampler)
    va = DataLoader(Subset(ds, val_idx), batch_size=batch_size, shuffle=False)
    te = DataLoader(Subset(ds, test_idx), batch_size=batch_size, shuffle=False)
    return tr, va, te


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for x_hsi, x_dem, y in loader:
            logits = model(x_hsi.to(device), x_dem.to(device))
            y_pred.append(logits.argmax(dim=1).cpu())
            y_true.append(y)
    y_true = torch.cat(y_true).numpy()
    y_pred = torch.cat(y_pred).numpy()
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "per_class_f1": [float(v) for v in f1_score(
            y_true, y_pred, average=None, labels=[0, 1, 2], zero_division=0
        )],
        "counts": [int((y_true == c).sum()) for c in range(3)],
    }


def train(cfg: dict) -> dict:
    torch.manual_seed(cfg["train"]["seed"])
    np.random.seed(cfg["train"]["seed"])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    npz = dict(np.load(cfg["data"]["npz"]))
    pca_components = int(npz.get("pca_n_components", npz["X_hsi"].shape[1]))
    cfg = dict(cfg)
    cfg["data"]["pca_components"] = pca_components

    model = create_model(cfg).to(device)
    tr, va, te = make_loaders(npz, cfg["train"]["batch_size"], cfg["train"]["seed"])

    loss_fn = nn.CrossEntropyLoss(
        weight=torch.tensor(
            [1.0, 1.0, 1.0]
            if not cfg["train"].get("class_weights", True)
            else 1.0 / np.bincount(npz["y_sev"][~npz["val_mask"] & ~npz["test_mask"]],
                                   minlength=3).astype(np.float64),
            dtype=torch.float32,
            device=device,
        )
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )

    best_val, best_state = -1.0, None
    for epoch in range(1, int(cfg["train"]["epochs"]) + 1):
        model.train()
        total, correct, running = 0, 0, 0.0
        for x_hsi, x_dem, y in tr:
            x_hsi, x_dem, y = x_hsi.to(device), x_dem.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x_hsi, x_dem), y)
            loss.backward()
            opt.step()
            running += loss.item() * len(y)
            total += len(y)
            correct += (model(x_hsi, x_dem).argmax(1) == y).sum().item()
        val = evaluate(model, va, device)
        best = val["accuracy"] > best_val
        if best:
            best_val = val["accuracy"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        print(
            f"[train] epoch {epoch:02d} loss {running / total:.4f} "
            f"train acc {correct / total:.4f} val acc {val['accuracy']:.4f}"
            f"{' (best)' if best else ''}"
        )
        if running / total < 0.01 and epoch >= 3:
            print("[train] early break: loss collapsed", file=sys.stderr)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    test = evaluate(model, te, device)

    return {
        "model": model,
        "best_val_accuracy": float(best_val),
        "test": test,
        "pca_components": pca_components,
        "dem_scale": [float(v) for v in npz["dem_scale"]],
    }


def save_artifact(cfg: dict, result: dict, version: str) -> Path:
    out = Path(cfg["output"]["dir"])
    out.mkdir(parents=True, exist_ok=True)

    cp = {
        "config": cfg,
        "model_version": version,
        "class_names": ["None", "Moderate", "Severe Collapse"],
        "pca_components": result["pca_components"],
        "dem_scale": [float(x) for x in result["dem_scale"]],
        "state_dict": result["model"].state_dict(),
    }
    torch.save(cp, out / cfg["output"]["checkpoint"])

    metrics = {
        "model_version": version,
        "pca_components": result["pca_components"],
        "best_val_accuracy": result["best_val_accuracy"],
        "test": result["test"],
    }
    (out / cfg["output"]["metrics"]).write_text(json.dumps(metrics, indent=2))
    print(f"[train] wrote {out / cfg['output']['checkpoint']}")
    print(f"[train] wrote {out / cfg['output']['metrics']}")
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="configs/phase3.yaml")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(Path(args.config).read_text())
    result = train(cfg)

    # copy the fitted PCA beside the model for inference-time reuse
    out_dir = save_artifact(cfg, result, version=model_version(cfg))
    pca_path = Path(cfg["data"].get("pca") or "")
    if pca_path.exists():
        shutil.copy(pca_path, out_dir / "pca.pkl")
        print(f"[train] copied PCA -> {out_dir / 'pca.pkl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())