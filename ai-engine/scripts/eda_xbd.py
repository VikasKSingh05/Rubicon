#!/usr/bin/env python
"""Explore the xBD training split using the same masks and weighting as training."""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Sequence

import cv2
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ai_engine.data.dataset import XBDDataset
from ai_engine.data.parser import INVERSE_DAMAGE_MAP
from ai_engine.data.splits import get_xbd_root, load_or_create_splits
from ai_engine.model.losses import inverse_sqrt_class_weights
from ai_engine.data.synthetic import generate_synthetic_dataset


def _select_training_tiles(
    tiles: list[dict[str, Any]], subset: float, seed: int = 42
) -> list[dict[str, Any]]:
    if not 0 < subset <= 1:
        raise ValueError(f"--subset must be in (0, 1], got {subset}")
    if subset == 1:
        return list(tiles)
    sample_count = max(1, int(len(tiles) * subset))
    indices = np.sort(
        np.random.default_rng(seed).choice(len(tiles), size=sample_count, replace=False)
    )
    return [tiles[int(index)] for index in indices]


def _count_label_objects(tiles: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for tile in tiles:
        with open(tile["post_label"], "r", encoding="utf-8") as label_file:
            label = json.load(label_file)
        features = label.get("features", {}).get("xy", [])
        for feature in features:
            subtype = feature.get("properties", {}).get("subtype")
            if subtype in ("no-damage", "minor-damage", "major-damage", "destroyed"):
                counts[subtype] += 1
            elif subtype == "un-classified":
                counts["no-damage"] += 1
            else:
                raise ValueError(
                    f"Unexpected damage subtype {subtype!r} in {tile['post_label']}"
                )
    return dict(sorted(counts.items()))


def _image_statistics(tiles: list[dict[str, Any]], max_samples: int) -> dict[str, Any]:
    samples = tiles[:max_samples]
    pre_pixels: list[np.ndarray] = []
    post_pixels: list[np.ndarray] = []
    differences: list[np.ndarray] = []
    shapes: Counter[str] = Counter()

    for tile in samples:
        pre = cv2.imread(tile["pre_image"], cv2.IMREAD_COLOR)
        post = cv2.imread(tile["post_image"], cv2.IMREAD_COLOR)
        if pre is None:
            raise FileNotFoundError(f"Cannot read pre-disaster image: {tile['pre_image']}")
        if post is None:
            raise FileNotFoundError(f"Cannot read post-disaster image: {tile['post_image']}")
        if post.shape[:2] != pre.shape[:2]:
            post = cv2.resize(post, (pre.shape[1], pre.shape[0]), interpolation=cv2.INTER_AREA)
        shapes[f"{pre.shape[1]}x{pre.shape[0]}"] += 1
        pre = cv2.resize(pre, (128, 128), interpolation=cv2.INTER_AREA)
        post = cv2.resize(post, (128, 128), interpolation=cv2.INTER_AREA)
        pre_pixels.append(pre.reshape(-1, 3).astype(np.float64) / 255.0)
        post_pixels.append(post.reshape(-1, 3).astype(np.float64) / 255.0)
        differences.append(
            np.abs(pre.astype(np.int16) - post.astype(np.int16)).reshape(-1, 3)
            .astype(np.float64)
            / 255.0
        )

    def summary(arrays: list[np.ndarray]) -> dict[str, list[float]]:
        pixels = np.concatenate(arrays, axis=0)
        return {
            "mean_bgr": pixels.mean(axis=0).round(6).tolist(),
            "std_bgr": pixels.std(axis=0).round(6).tolist(),
        }

    if not samples:
        return {
            "tiles_sampled": 0,
            "sample_resolution": [128, 128],
            "shapes": {},
            "pre": None,
            "post": None,
            "absolute_pair_difference": None,
        }

    return {
        "tiles_sampled": len(samples),
        "sample_resolution": [128, 128],
        "shapes": dict(sorted(shapes.items())),
        "pre": summary(pre_pixels),
        "post": summary(post_pixels),
        "absolute_pair_difference": {
            "mean_bgr": np.concatenate(differences, axis=0).mean(axis=0).round(6).tolist()
        },
    }


def run_eda(
    xbd_root: str | Path | None = None,
    subset: float = 1.0,
    max_samples: int = 32,
    cache_dir: str | Path = "data/xbd_cache",
    source_split: str | None = None,
) -> dict[str, Any]:
    root = get_xbd_root(xbd_root)
    splits = load_or_create_splits(root, source_split=source_split)
    metadata = splits.get("metadata", {})
    train_tiles = splits.get("train", [])
    if not train_tiles:
        raise ValueError(f"No training tiles found for xBD root {root}")

    selected_tiles = _select_training_tiles(train_tiles, subset)
    dataset = XBDDataset(
        selected_tiles,
        crop_size=None,
        is_train=False,
        cache_dir=cache_dir,
    )
    class_pixel_counts = dataset.damage_class_pixel_counts()
    class_weights = inverse_sqrt_class_weights(class_pixel_counts)
    class_names = [INVERSE_DAMAGE_MAP[class_id] for class_id in range(1, 5)]
    pixel_summary = {
        class_names[index]: int(class_pixel_counts[index])
        for index in range(len(class_names))
    }
    weight_summary = {
        class_names[index]: round(float(class_weights[index]), 6)
        for index in range(len(class_names))
    }

    return {
        "dataset_root": str(root.resolve()),
        "source_split": metadata.get("source_split"),
        "split_manifest": str(
            (root / f"splits_{metadata.get('source_split')}.json")
            if (root / f"splits_{metadata.get('source_split')}.json").exists()
            else (root / "splits.json")
        ),
        "train_tiles_available": len(train_tiles),
        "training_subset_fraction": subset,
        "training_tiles_analyzed": len(selected_tiles),
        "validation_tiles": len(splits.get("val", [])),
        "train_events": metadata.get("train_events", []),
        "validation_events": metadata.get("val_events", []),
        "damage_object_counts": _count_label_objects(selected_tiles),
        "building_pixel_counts": pixel_summary,
        "class_weights": weight_summary,
        "weighting": "normalized inverse-square-root pixel frequency, capped at 10; no weighted sampler",
        "image_statistics": _image_statistics(selected_tiles, max_samples),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Analyze xBD training labels, pixel frequencies, and image pairs."
    )
    parser.add_argument("--xbd-root", type=str, default=None, help="Root path to xBD")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Generate and analyze a clearly labeled synthetic dataset",
    )
    parser.add_argument("--subset", type=float, default=1.0, help="Training tile fraction (0, 1]")
    parser.add_argument(
        "--max-samples", type=int, default=32, help="Maximum image pairs for RGB statistics"
    )
    parser.add_argument("--source-split", type=str, default=None, help="Source split, e.g. train")
    parser.add_argument("--cache-dir", type=str, default="data/xbd_cache")
    parser.add_argument("--save-json", type=str, default="data/eda_summary.json")
    args = parser.parse_args(argv)

    if args.max_samples < 0:
        parser.error("--max-samples must be non-negative")
    xbd_root = args.xbd_root
    if args.synthetic:
        if xbd_root:
            parser.error("--synthetic and --xbd-root cannot be used together")
        xbd_root = str(generate_synthetic_dataset(output_dir="data/synthetic_xbd"))

    report = run_eda(
        xbd_root=xbd_root,
        subset=args.subset,
        max_samples=args.max_samples,
        cache_dir=args.cache_dir,
        source_split=args.source_split,
    )
    print(json.dumps(report, indent=2))
    output_path = Path(args.save_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nEDA summary saved to {output_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
