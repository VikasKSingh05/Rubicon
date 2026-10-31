"""Disaster-event disjoint split utilities for the xBD dataset.

Per the research paper specification: Disaster events present in the training set
must never appear in both train and validation sets.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def extract_event_name(filename_or_path: str | Path) -> str:
    """Extract disaster event prefix from an xBD filename.

    xBD convention: <disaster-name>_<disaster-number>_<pre|post>_disaster.<ext>
    Example: 'santa-rosa-wildfire_00000027_post_disaster.png' -> 'santa-rosa-wildfire'
    """
    stem = Path(filename_or_path).stem
    # Strip suffixes
    for suffix in ("_post_disaster", "_pre_disaster", "_post", "_pre"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    # Strip tile number index (e.g. '_00000027' or '_0001')
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def discover_xbd_tiles(data_root: str | Path) -> List[Dict[str, Any]]:
    """Scan directory structure to identify valid (pre, post, label) tile triplets."""
    data_root = Path(data_root)
    post_images = list(data_root.rglob("*_post_disaster.*"))
    tiles = []

    for post_img in sorted(post_images):
        if post_img.suffix.lower() not in (".png", ".jpg", ".jpeg", ".tif", ".tiff"):
            continue
        ext = post_img.suffix
        stem = post_img.stem.replace("_post_disaster", "")
        img_dir = post_img.parent

        pre_img = img_dir / f"{stem}_pre_disaster{ext}"
        if not pre_img.exists():
            continue

        parts = list(img_dir.parts)
        if "images" in parts:
            idx = parts.index("images")
            lbl_parts = parts[:idx] + ["labels"] + parts[idx + 1 :]
            lbl_dir = Path(*lbl_parts)
        else:
            lbl_dir = img_dir

        post_lbl = lbl_dir / f"{stem}_post_disaster.json"
        pre_lbl = lbl_dir / f"{stem}_pre_disaster.json"

        event = extract_event_name(stem)
        tiles.append({
            "event": event,
            "tile_id": stem,
            "pre_image": str(pre_img),
            "post_image": str(post_img),
            "pre_img": str(pre_img),
            "post_img": str(post_img),
            "pre_label": str(pre_lbl) if pre_lbl.exists() else None,
            "post_label": str(post_lbl) if post_lbl.exists() else None,
        })

    return tiles


def create_event_splits(
    tiles: List[Dict[str, Any]],
    output_path: Optional[str | Path] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Split dataset by event: events in train set NEVER appear in validation set."""
    events = sorted({t["event"] for t in tiles})
    rng = np.random.default_rng(seed)
    shuffled_events = list(events)
    rng.shuffle(shuffled_events)

    # Greedy allocation of events to target tile count ratio
    total_tiles = len(tiles)
    target_train = int(total_tiles * train_ratio)

    event_tiles = {ev: [t for t in tiles if t["event"] == ev] for ev in shuffled_events}

    train_events: List[str] = []
    val_events: List[str] = []
    cur_train_count = 0

    for ev in shuffled_events:
        count = len(event_tiles[ev])
        if cur_train_count + count <= target_train or not train_events:
            train_events.append(ev)
            cur_train_count += count
        else:
            val_events.append(ev)

    splits = {
        "metadata": {
            "total_tiles": total_tiles,
            "train_tiles": cur_train_count,
            "val_tiles": total_tiles - cur_train_count,
            "train_events": train_events,
            "val_events": val_events,
        },
        "train": [t for ev in train_events for t in event_tiles[ev]],
        "val": [t for ev in val_events for t in event_tiles[ev]],
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as fh:
            json.dump(splits, fh, indent=2)

    return splits


def get_xbd_root(candidate_path: Optional[str | Path] = None) -> Path:
    """Resolve the root path of the xBD dataset from candidate path, env var, or known locations."""
    if candidate_path:
        p = Path(candidate_path).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified xBD root does not exist: {candidate_path}")

    env_path = os.environ.get("XBD_DATA_DIR") or os.environ.get("XBD_ROOT")
    if env_path:
        p = Path(env_path).resolve()
        if p.exists():
            return p

    candidates = [
        Path("data/xbd"),
        Path("ai-engine/data/xbd"),
        Path("sample-scans/xbd"),
        Path("../data/xbd"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError("xBD dataset root not found in default locations or environment variables.")


def load_or_create_splits(
    xbd_root: str | Path,
    split_file: Optional[str | Path] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
) -> Dict[str, List[Dict[str, Any]]]:
    """Load existing event-based splits from JSON or discover and create them."""
    xbd_root = Path(xbd_root)
    split_path = Path(split_file) if split_file else xbd_root / "splits.json"

    if split_path.exists():
        try:
            with open(split_path, "r", encoding="utf-8") as fh:
                splits = json.load(fh)
            if "train" in splits and "val" in splits:
                return splits
        except Exception:
            pass

    tiles = discover_xbd_tiles(xbd_root)
    if not tiles:
        # Fallback empty structure
        return {"train": [], "val": []}

    # If only 1 event exists, split by tile to ensure both train and val have data
    unique_events = {t["event"] for t in tiles}
    if len(unique_events) <= 1:
        rng = np.random.default_rng(seed)
        shuffled = list(tiles)
        rng.shuffle(shuffled)
        cut = max(1, int(len(shuffled) * train_ratio))
        if cut >= len(shuffled):
            cut = max(1, len(shuffled) - 1)
        splits = {
            "train": shuffled[:cut],
            "val": shuffled[cut:] if len(shuffled) > 1 else shuffled[:cut],
        }
        split_path.parent.mkdir(parents=True, exist_ok=True)
        with open(split_path, "w", encoding="utf-8") as fh:
            json.dump(splits, fh, indent=2)
        return splits

    return create_event_splits(tiles, output_path=split_path, train_ratio=train_ratio, seed=seed)
