"""Disaster-event disjoint split utilities for the xBD dataset.

Splitting rules
---------------
* Tiles are **never** split randomly.  Every tile of an event lives entirely
  in either train or validation.
* ``data/xbd/train`` is used as the primary source when present; ``hold`` is
  kept as a secondary tile-level holdout (it contains the same event names,
  so it is *not* event-disjoint from ``train``).
* At least two whole events are held out for validation when the source split
  has enough events (default: ``mexico-earthquake`` and ``palu-tsunami``).
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

SPLIT_SCHEMA_VERSION = 2
KNOWN_SPLITS = ("train", "hold", "test")
DEFAULT_VAL_EVENTS: Tuple[str, ...] = ("mexico-earthquake", "palu-tsunami")
MIN_VAL_EVENTS = 2
MIN_TRUSTWORTHY_VAL_TILES = 100


def extract_event_name(filename_or_path: str | Path) -> str:
    """Extract the disaster event prefix from an xBD filename.

    Example: ``santa-rosa-wildfire_00000027_post_disaster.png`` →
    ``santa-rosa-wildfire``.
    """
    stem = Path(filename_or_path).stem
    for suffix in ("_post_disaster_target", "_pre_disaster_target",
                   "_post_disaster", "_pre_disaster", "_post", "_pre"):
        if stem.endswith(suffix):
            stem = stem[: -len(suffix)]
            break
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0]
    return stem


def _sibling_dir(directory: Path, old_name: str, new_name: str) -> Path:
    """Replace the last occurrence of *old_name* in a path with *new_name*."""
    parts = list(directory.parts)
    for idx in range(len(parts) - 1, -1, -1):
        if parts[idx] == old_name:
            parts[idx] = new_name
            return Path(*parts)
    return directory.parent / new_name


def _tile_split_name(post_img: Path, data_root: Path) -> str:
    try:
        rel = post_img.relative_to(data_root)
    except ValueError:
        return data_root.name
    if len(rel.parts) > 1 and rel.parts[0] in KNOWN_SPLITS:
        return rel.parts[0]
    if data_root.name in KNOWN_SPLITS:
        return data_root.name
    return data_root.name


def discover_xbd_tiles(
    data_root: str | Path,
    split: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """Scan a directory for valid (pre, post, label, target) tile quadruplets.

    Supports both layouts:

    * ``<root>/{train,hold,test}/{images,labels,targets}/...``
    * ``<root>/{images,labels,targets}/...`` (hold-only extraction)
    * per-event subfolders under ``images/`` and ``labels/``
    """
    data_root = Path(data_root)
    if split:
        candidate = data_root / split
        if candidate.is_dir():
            data_root = candidate
        elif data_root.name != split:
            raise FileNotFoundError(f"xBD split '{split}' not found under {data_root}")

    image_exts = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
    tiles: List[Dict[str, Any]] = []
    seen: set[Tuple[str, str]] = set()

    for post_img in sorted(data_root.rglob("*_post_disaster.*")):
        if not post_img.is_file() or post_img.suffix.lower() not in image_exts:
            continue
        if post_img.stem.endswith("_target"):
            continue

        stem = post_img.stem[: -len("_post_disaster")]
        ext = post_img.suffix
        img_dir = post_img.parent
        pre_img = img_dir / f"{stem}_pre_disaster{ext}"
        if not pre_img.exists():
            continue

        lbl_dir = _sibling_dir(img_dir, "images", "labels")
        tgt_dir = _sibling_dir(img_dir, "images", "targets")
        post_lbl = lbl_dir / f"{stem}_post_disaster.json"
        pre_lbl = lbl_dir / f"{stem}_pre_disaster.json"
        if not post_lbl.exists():
            continue

        pre_tgt = tgt_dir / f"{stem}_pre_disaster_target.png"
        post_tgt = tgt_dir / f"{stem}_post_disaster_target.png"

        split_name = _tile_split_name(post_img, data_root)
        key = (split_name, stem)
        if key in seen:
            continue
        seen.add(key)

        tiles.append({
            "event": extract_event_name(stem),
            "tile_id": stem,
            "split": split_name,
            "pre_image": str(pre_img),
            "post_image": str(post_img),
            "pre_img": str(pre_img),
            "post_img": str(post_img),
            "pre_label": str(pre_lbl) if pre_lbl.exists() else None,
            "post_label": str(post_lbl),
            "pre_target": str(pre_tgt) if pre_tgt.exists() else None,
            "post_target": str(post_tgt) if post_tgt.exists() else None,
        })

    return tiles


def _choose_val_events(
    events: List[str],
    event_tiles: Dict[str, List[Dict[str, Any]]],
    train_ratio: float,
    seed: int,
    val_events: Optional[Iterable[str]],
    min_val_events: int,
) -> List[str]:
    if len(events) < 2:
        return []

    explicit = [e for e in (val_events or []) if e in event_tiles]
    if explicit:
        return explicit

    if all(e in event_tiles for e in DEFAULT_VAL_EVENTS):
        return list(DEFAULT_VAL_EVENTS)

    # Deterministic seeded choice for datasets that do not contain the default
    # holdout events (e.g. the small synthetic CI set).
    max_val_events = max(1, min_val_events)
    max_val_events = min(max_val_events, len(events) - 1)
    target_val = int(round(sum(len(v) for v in event_tiles.values()) * (1.0 - train_ratio)))

    rng = np.random.default_rng(seed)
    shuffled = list(events)
    rng.shuffle(shuffled)

    chosen: List[str] = []
    count = 0
    for ev in shuffled:
        if len(chosen) >= max_val_events and count >= target_val:
            break
        if len(chosen) >= len(events) - 1:
            break
        chosen.append(ev)
        count += len(event_tiles[ev])
        if len(chosen) >= max_val_events and count >= target_val:
            break
    return chosen


def create_event_splits(
    tiles: List[Dict[str, Any]],
    output_path: Optional[str | Path] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
    val_events: Optional[Iterable[str]] = None,
    min_val_events: int = MIN_VAL_EVENTS,
    source_split: Optional[str] = None,
) -> Dict[str, Any]:
    """Split tiles by whole disaster events with zero event overlap."""
    events = sorted({t["event"] for t in tiles})
    event_tiles = {ev: [t for t in tiles if t["event"] == ev] for ev in events}

    chosen_val = _choose_val_events(
        events=events,
        event_tiles=event_tiles,
        train_ratio=train_ratio,
        seed=seed,
        val_events=val_events,
        min_val_events=min_val_events,
    )
    val_set = set(chosen_val)
    train_events = [ev for ev in events if ev not in val_set]
    val_events_sorted = [ev for ev in events if ev in val_set]

    train_tiles = [t for ev in train_events for t in event_tiles[ev]]
    val_tiles = [t for ev in val_events_sorted for t in event_tiles[ev]]

    warnings: List[str] = []
    if len(events) < 2:
        warnings.append(
            "Only one disaster event is available; no event-disjoint validation "
            "split is possible. Add another event before trusting metrics."
        )
    elif len(val_events_sorted) < min_val_events:
        warnings.append(
            f"Validation holds only {len(val_events_sorted)} event(s); "
            f"{min_val_events} are recommended for a trustworthy estimate."
        )
    if 0 < len(val_tiles) < MIN_TRUSTWORTHY_VAL_TILES:
        warnings.append(
            f"Validation has {len(val_tiles)} tiles (<{MIN_TRUSTWORTHY_VAL_TILES}); "
            "treat validation metrics as noisy."
        )
    if not val_tiles:
        warnings.append("Validation split is empty; metrics cannot be computed.")

    total_tiles = len(tiles)
    splits: Dict[str, Any] = {
        "metadata": {
            "schema_version": SPLIT_SCHEMA_VERSION,
            "split_mode": "event",
            "event_disjoint": True,
            "source_split": source_split,
            "total_tiles": total_tiles,
            "total_events": len(events),
            "train_count": len(train_tiles),
            "val_count": len(val_tiles),
            "train_tiles": len(train_tiles),
            "val_tiles": len(val_tiles),
            "train_events": train_events,
            "val_events": val_events_sorted,
            "train_ratio_target": train_ratio,
            "val_fraction": round(len(val_tiles) / total_tiles, 4) if total_tiles else 0.0,
            "seed": seed,
            "warnings": warnings,
        },
        "train": train_tiles,
        "val": val_tiles,
    }

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(splits, fh, indent=2)

    return splits


def get_xbd_root(candidate_path: Optional[str | Path] = None) -> Path:
    """Resolve the xBD root from an explicit path, env var, or known locations."""
    if candidate_path:
        p = Path(candidate_path).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Specified xBD root does not exist: {candidate_path}")

    env_path = os.environ.get("XBD_DATA_DIR") or os.environ.get("XBD_ROOT")
    if env_path:
        p = Path(env_path).expanduser().resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"XBD_ROOT points to a missing directory: {env_path}")

    candidates = [
        Path("data/xbd"),
        Path("ai-engine/data/xbd"),
        Path("../data/xbd"),
    ]
    for c in candidates:
        if c.exists():
            return c.resolve()

    raise FileNotFoundError(
        "xBD dataset root not found. Set XBD_ROOT=./data/xbd or pass --xbd-root."
    )


def resolve_source_root(
    xbd_root: str | Path,
    source_split: Optional[str] = None,
) -> Tuple[Path, str]:
    """Return (split_root, split_name) for the primary train/val source."""
    root = Path(xbd_root)
    if source_split:
        candidate = root / source_split
        if candidate.is_dir():
            return candidate, source_split
        if root.name == source_split:
            return root, source_split
        raise FileNotFoundError(f"xBD split '{source_split}' not found under {root}")

    for name in ("train", "hold"):
        candidate = root / name
        if candidate.is_dir():
            return candidate, name
    return root, root.name


def _default_split_file(xbd_root: Path, source_name: str) -> Path:
    if (xbd_root / source_name).is_dir():
        return xbd_root / f"splits_{source_name}.json"
    return xbd_root / "splits.json"


def validate_splits(
    splits: Dict[str, Any],
    source_split: Optional[str] = None,
) -> bool:
    """Return True only for a usable event-disjoint split manifest."""
    meta = splits.get("metadata") or {}
    if meta.get("split_mode") != "event" or not meta.get("event_disjoint"):
        return False
    if source_split and meta.get("source_split") not in (None, source_split):
        return False
    train = splits.get("train") or []
    val = splits.get("val") or []
    train_events = set(meta.get("train_events") or [])
    val_events = set(meta.get("val_events") or [])
    if train_events & val_events:
        return False
    if not train:
        return False
    if len(train_events) + len(val_events) > 1 and not val:
        return False
    for tile in train:
        if tile.get("event") not in train_events:
            return False
    for tile in val:
        if tile.get("event") not in val_events:
            return False
    train_ids = {t.get("tile_id") for t in train}
    val_ids = {t.get("tile_id") for t in val}
    if train_ids & val_ids:
        return False
    return True


def load_or_create_splits(
    xbd_root: str | Path,
    split_file: Optional[str | Path] = None,
    train_ratio: float = 0.8,
    seed: int = 42,
    force_recreate: bool = False,
    val_events: Optional[Iterable[str]] = None,
    min_val_events: int = MIN_VAL_EVENTS,
    source_split: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a valid event split manifest or discover tiles and create one."""
    xbd_root = Path(xbd_root)
    source_root, source_name = resolve_source_root(xbd_root, source_split)
    split_path = Path(split_file) if split_file else _default_split_file(xbd_root, source_name)

    if split_path.exists() and not force_recreate:
        try:
            with open(split_path, "r", encoding="utf-8") as fh:
                splits = json.load(fh)
            if validate_splits(splits, source_split=source_name):
                return splits
        except (OSError, json.JSONDecodeError):
            pass

    tiles = discover_xbd_tiles(source_root)
    if not tiles:
        return {
            "metadata": {
                "schema_version": SPLIT_SCHEMA_VERSION,
                "split_mode": "event",
                "event_disjoint": True,
                "source_split": source_name,
                "total_tiles": 0,
                "total_events": 0,
                "train_count": 0,
                "val_count": 0,
                "train_events": [],
                "val_events": [],
                "warnings": [f"No labelled pre/post tile pairs found under {source_root}"],
            },
            "train": [],
            "val": [],
        }

    return create_event_splits(
        tiles,
        output_path=split_path,
        train_ratio=train_ratio,
        seed=seed,
        val_events=val_events,
        min_val_events=min_val_events,
        source_split=source_name,
    )
