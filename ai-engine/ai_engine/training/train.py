"""Compatibility API and CLI for the root xBD v3 trainer.

The root ``train.py`` owns the training implementation. This module preserves
the historical programmatic ``train(...)`` call while routing it to the active
four-class damage model.
"""
from __future__ import annotations

import importlib.util
import sys
import tempfile
from functools import lru_cache
from pathlib import Path
from typing import Any, Sequence

from ..data.splits import get_xbd_root, load_or_create_splits
from ..data.synthetic import generate_synthetic_dataset


@lru_cache(maxsize=1)
def _root_trainer():
    trainer_path = Path(__file__).resolve().parents[3] / "train.py"
    spec = importlib.util.spec_from_file_location("_rubicon_xbd_v3_trainer", trainer_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load the xBD v3 trainer at {trainer_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _run_v3(
    xbd_root: Path,
    *,
    synthetic: bool,
    epochs: int | None,
    subset: int | None,
    output_dir: str | Path,
    model_version: str | None,
) -> dict[str, Any]:
    splits = load_or_create_splits(xbd_root)
    available_tiles = len(splits.get("train", []))
    if available_tiles == 0:
        raise ValueError(f"No training tiles found under {xbd_root}")

    args = ["--config", "auto"]
    if synthetic:
        args.extend(["--synthetic", "--synthetic-dir", str(xbd_root)])
    else:
        args.extend(["--xbd-root", str(xbd_root)])
    args.extend([
        "--output-dir",
        str(output_dir),
        "--log-dir",
        str(Path(output_dir) / "logs"),
        "--no-pretrained",
    ])
    if model_version is not None:
        args.extend(["--model-version", model_version])
    if epochs is not None:
        args.extend(["--epochs", str(epochs)])
    if subset is not None:
        if subset <= 0:
            raise ValueError("subset must be a positive tile count")
        subset_count = min(subset, available_tiles)
        subset_fraction = min(1.0, (subset_count + 0.1) / available_tiles)
        args.extend(["--subset", str(subset_fraction)])
    summary = _root_trainer().train(args)
    history = summary["history"]
    if not history:
        raise RuntimeError("The xBD v3 trainer completed without recording an epoch")
    return {
        "model_version": summary["model_version"],
        "checkpoint_hash": summary["checkpoint_hash"],
        "best_val_xview2_score": summary["best_validation_xview2_score"],
        "final_val": history[-1]["validation_metrics"],
        "history": history,
        "hardware_profile": summary["hardware_profile"],
        "training_metrics": summary,
    }


def train(
    xbd_root: str | Path | None = None,
    synthetic: bool = False,
    epochs: int | None = None,
    subset: int | None = None,
    output_dir: str | Path = "data/models",
    model_version: str | None = None,
) -> dict[str, Any]:
    """Train the active v3 model, preserving the legacy call signature."""
    if synthetic:
        with tempfile.TemporaryDirectory(prefix="rubicon-xbd-synthetic-") as temp_dir:
            root = Path(temp_dir) / "xbd"
            generate_synthetic_dataset(output_dir=root)
            return _run_v3(
                root,
                synthetic=True,
                epochs=epochs,
                subset=subset,
                output_dir=output_dir,
                model_version=model_version,
            )

    if xbd_root is None:
        raise ValueError("Provide xbd_root for real-data training or set synthetic=True")
    root = get_xbd_root(xbd_root)
    return _run_v3(
        root,
        synthetic=False,
        epochs=epochs,
        subset=subset,
        output_dir=output_dir,
        model_version=model_version,
    )


def main(argv: Sequence[str] | None = None) -> int:
    _root_trainer().train(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
