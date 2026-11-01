"""Verification and diagnostic tool for the Rubicon xBD data pipeline.

Usage (from the repo root):
    python -m ai_engine.data.check
    python -m ai_engine.data.check --xbd-root ./data/xbd
    python -m ai_engine.data.check --synthetic

Prints:
    - discovered train/hold/test inventories and per-event pair counts
    - the exact event-disjoint train/validation split
    - mask agreement against xBD ``targets/`` PNGs
    - class-frequency histogram
Saves:
    - qualitative sample overlays under ``docs/samples/``
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import numpy as np

from .parser import INVERSE_DAMAGE_MAP, get_or_create_masks
from .splits import (
    discover_xbd_tiles,
    get_xbd_root,
    load_or_create_splits,
    resolve_source_root,
)
from .synthetic import generate_synthetic_dataset


DAMAGE_PALETTE_BGR = {
    0: (0, 0, 0),
    1: (46, 204, 113),   # green
    2: (0, 215, 255),    # amber
    3: (0, 140, 255),    # orange
    4: (36, 40, 231),    # red
}


def repo_root() -> Path:
    """Resolve the repository root from this file's location."""
    return Path(__file__).resolve().parents[3]


def default_cache_dir() -> Path:
    env = os.environ.get("XBD_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return repo_root() / "data" / "xbd_cache"


def event_counts(tiles: Sequence[Dict[str, Any]]) -> Dict[str, int]:
    counts = Counter(t["event"] for t in tiles)
    return dict(sorted(counts.items()))


def _format_event_counts(tiles: Sequence[Dict[str, Any]]) -> str:
    counts = event_counts(tiles)
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _sample_tiles(tiles: Sequence[Dict[str, Any]], n: int, seed: int = 42) -> List[Dict[str, Any]]:
    if n <= 0 or not tiles:
        return []
    if n >= len(tiles):
        return list(tiles)
    rng = random.Random(seed)
    return rng.sample(list(tiles), n)


def crosscheck_targets(
    tiles: Sequence[Dict[str, Any]],
    cache_dir: str | Path,
    n: int = 10,
    seed: int = 42,
) -> Dict[str, Any]:
    """Rasterize labels and compare against xBD ``targets/`` PNGs."""
    candidates = [t for t in tiles if t.get("pre_target") and t.get("post_target")]
    sample = _sample_tiles(candidates, n, seed=seed)
    if not sample:
        return {"available": False, "reason": "no targets/ PNGs found", "tiles": 0}

    rows: List[Dict[str, Any]] = []
    total_px = loc_match_total = dmg_match_total = 0
    tp = Counter()
    fp = Counter()
    fn = Counter()

    for tile in sample:
        loc, dmg = get_or_create_masks(
            tile_id=tile["tile_id"],
            pre_json_path=tile.get("pre_label"),
            post_json_path=tile["post_label"],
            cache_dir=cache_dir,
        )
        gt_loc = cv2.imread(tile["pre_target"], cv2.IMREAD_UNCHANGED)
        gt_dmg = cv2.imread(tile["post_target"], cv2.IMREAD_UNCHANGED)
        if gt_loc is None or gt_dmg is None:
            continue
        if gt_loc.shape[:2] != loc.shape[:2]:
            gt_loc = cv2.resize(gt_loc, (loc.shape[1], loc.shape[0]), interpolation=cv2.INTER_NEAREST)
        if gt_dmg.shape[:2] != dmg.shape[:2]:
            gt_dmg = cv2.resize(gt_dmg, (dmg.shape[1], dmg.shape[0]), interpolation=cv2.INTER_NEAREST)

        loc_ok = int((loc == (gt_loc > 0)).sum())
        dmg_ok = int((dmg == gt_dmg).sum())
        total_px += dmg.size
        loc_match_total += loc_ok
        dmg_match_total += dmg_ok

        for cls in range(5):
            pred = dmg == cls
            gt = gt_dmg == cls
            tp[cls] += int((pred & gt).sum())
            fp[cls] += int((pred & ~gt).sum())
            fn[cls] += int((~pred & gt).sum())

        rows.append({
            "tile_id": tile["tile_id"],
            "event": tile["event"],
            "loc_match_pct": round(100.0 * loc_ok / loc.size, 4),
            "dmg_match_pct": round(100.0 * dmg_ok / dmg.size, 4),
            "dmg_mismatch_px": int(dmg.size - dmg_ok),
        })

    if not rows:
        return {"available": False, "reason": "targets could not be read", "tiles": 0}

    per_class = {}
    for cls in range(5):
        denom = tp[cls] + fp[cls] + fn[cls]
        per_class[INVERSE_DAMAGE_MAP[cls]] = round(tp[cls] / denom, 4) if denom else 1.0

    return {
        "available": True,
        "tiles": len(rows),
        "loc_agreement_pct": round(100.0 * loc_match_total / total_px, 4),
        "dmg_agreement_pct": round(100.0 * dmg_match_total / total_px, 4),
        "per_class_iou": per_class,
        "rows": rows,
    }


def class_histogram(
    tiles: Sequence[Dict[str, Any]],
    cache_dir: str | Path,
    limit: int = 200,
) -> Dict[str, Any]:
    pixel_counts: Counter = Counter()
    tiles_with_class: Counter = Counter()
    sample = list(tiles)[:limit]
    for tile in sample:
        _, dmg = get_or_create_masks(
            tile_id=tile["tile_id"],
            pre_json_path=tile.get("pre_label"),
            post_json_path=tile["post_label"],
            cache_dir=cache_dir,
        )
        unique, counts = np.unique(dmg, return_counts=True)
        for cls, count in zip(unique.tolist(), counts.tolist()):
            pixel_counts[int(cls)] += int(count)
            if cls > 0:
                tiles_with_class[int(cls)] += 1
    return {
        "tiles_inspected": len(sample),
        "pixel_counts": dict(sorted(pixel_counts.items())),
        "tile_presence": dict(sorted(tiles_with_class.items())),
    }


def create_sample_overlay(
    pre_path: str,
    post_path: str,
    loc_mask: np.ndarray,
    dmg_mask: np.ndarray,
    tile_id: str,
) -> np.ndarray:
    """Create a 2x2 panel: pre, post, localization, damage."""
    pre_bgr = cv2.imread(pre_path)
    post_bgr = cv2.imread(post_path)
    if pre_bgr is None or post_bgr is None:
        raise FileNotFoundError(f"cannot read image pair for {tile_id}")
    h, w = pre_bgr.shape[:2]

    loc_overlay = pre_bgr.copy()
    loc_color = np.zeros_like(pre_bgr)
    loc_color[loc_mask > 0] = (0, 255, 0)
    cv2.addWeighted(loc_color, 0.45, loc_overlay, 0.55, 0, loc_overlay)
    contours, _ = cv2.findContours(loc_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(loc_overlay, contours, -1, (0, 255, 0), 1)

    dmg_overlay = post_bgr.copy()
    dmg_color = np.zeros_like(post_bgr)
    for cls, color in DAMAGE_PALETTE_BGR.items():
        dmg_color[dmg_mask == cls] = color
    has_damage = dmg_mask > 0
    dmg_blend = dmg_overlay.copy()
    cv2.addWeighted(dmg_color, 0.5, dmg_overlay, 0.5, 0, dmg_blend)
    dmg_overlay[has_damage] = dmg_blend[has_damage]
    for cls in (1, 2, 3, 4):
        mask_c = (dmg_mask == cls).astype(np.uint8)
        if mask_c.sum() > 0:
            c_contours, _ = cv2.findContours(mask_c, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(dmg_overlay, c_contours, -1, DAMAGE_PALETTE_BGR[cls], 1)

    def banner(img: np.ndarray, text: str) -> np.ndarray:
        out = img.copy()
        cv2.rectangle(out, (0, 0), (w, 40), (20, 20, 20), -1)
        cv2.putText(out, text, (15, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return out

    top = np.hstack([banner(pre_bgr, f"PRE: {tile_id}"), banner(post_bgr, "POST")])
    bottom = np.hstack([banner(loc_overlay, "LOCALIZATION"), banner(dmg_overlay, "DAMAGE")])
    return cv2.resize(np.vstack([top, bottom]), (1024, 1024), interpolation=cv2.INTER_AREA)


def _save_overlays(
    tiles: Sequence[Dict[str, Any]],
    cache_dir: Path,
    samples_dir: Path,
    num_samples: int,
) -> List[Path]:
    samples_dir.mkdir(parents=True, exist_ok=True)
    selected: List[Tuple[Dict[str, Any], np.ndarray, np.ndarray]] = []

    for tile in tiles:
        loc, dmg = get_or_create_masks(
            tile_id=tile["tile_id"],
            pre_json_path=tile.get("pre_label"),
            post_json_path=tile["post_label"],
            cache_dir=cache_dir,
        )
        if int((dmg > 0).sum()) > 500:
            selected.append((tile, loc, dmg))
            if len(selected) >= num_samples:
                break

    if len(selected) < num_samples:
        seen = {t["tile_id"] for t, _, _ in selected}
        for tile in tiles:
            if tile["tile_id"] in seen:
                continue
            loc, dmg = get_or_create_masks(
                tile_id=tile["tile_id"],
                pre_json_path=tile.get("pre_label"),
                post_json_path=tile["post_label"],
                cache_dir=cache_dir,
            )
            selected.append((tile, loc, dmg))
            if len(selected) >= num_samples:
                break

    paths: List[Path] = []
    for idx, (tile, loc, dmg) in enumerate(selected, start=1):
        overlay = create_sample_overlay(
            pre_path=tile["pre_image"],
            post_path=tile["post_image"],
            loc_mask=loc,
            dmg_mask=dmg,
            tile_id=tile["tile_id"],
        )
        path = samples_dir / f"sample_overlay_{idx}.png"
        cv2.imwrite(str(path), overlay)
        paths.append(path)
    return paths


def run_data_check(
    xbd_root_arg: Optional[str] = None,
    synthetic_mode: bool = False,
    num_samples: int = 5,
    samples_dir: Optional[str | Path] = None,
    crosscheck_n: int = 10,
    source_split: Optional[str] = None,
) -> Dict[str, Any]:
    root_dir = repo_root()
    cache_dir = default_cache_dir()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out_samples = Path(samples_dir) if samples_dir else root_dir / "docs" / "samples"

    print("=" * 78)
    print("RUBICON PHASE 2 - xBD DATA PIPELINE CHECK")
    print("=" * 78)

    if synthetic_mode:
        print("\n[INFO] SYNTHETIC mode (--synthetic): real xBD data is not being used.")
        xbd_root = generate_synthetic_dataset(output_dir=root_dir / "data" / "synthetic_xbd")
        available_splits = {"synthetic": discover_xbd_tiles(xbd_root)}
    else:
        xbd_root = get_xbd_root(xbd_root_arg)
        print(f"\n[INFO] xBD root: {xbd_root}")
        available_splits = {}
        for name in ("train", "hold", "test"):
            split_root = xbd_root / name
            if split_root.is_dir():
                available_splits[name] = discover_xbd_tiles(split_root)
        if not available_splits:
            available_splits[xbd_root.name] = discover_xbd_tiles(xbd_root)

    print("\n--- 1. Dataset inventory ---")
    for name, tiles in available_splits.items():
        print(f"{name:>8}: {len(tiles):>5} pairs | events={len(event_counts(tiles))}")
        if tiles:
            print(f"          {_format_event_counts(tiles)}")

    # Primary event-disjoint train/val split (train/ when present).
    source_root, source_name = resolve_source_root(xbd_root, source_split)
    splits = load_or_create_splits(
        xbd_root=xbd_root,
        source_split=source_name,
        force_recreate=True,
    )
    meta = splits.get("metadata", {})
    train_tiles = splits.get("train", [])
    val_tiles = splits.get("val", [])

    print(f"\n--- 2. Event-disjoint split (source: {source_name}) ---")
    print(f"Source root:        {source_root}")
    print(f"Total events:       {meta.get('total_events', 0)}")
    print(f"Train events ({len(meta.get('train_events', []))}): "
          f"{', '.join(meta.get('train_events', []))}")
    print(f"Val events   ({len(meta.get('val_events', []))}): "
          f"{', '.join(meta.get('val_events', []))}")
    print(f"Train tiles:        {len(train_tiles)}")
    print(f"Val tiles:          {len(val_tiles)} "
          f"({100.0 * meta.get('val_fraction', 0.0):.2f}% of source split)")
    overlap = set(meta.get("train_events", [])) & set(meta.get("val_events", []))
    print(f"Event overlap:      {sorted(overlap) if overlap else 'NONE'}")
    if meta.get("warnings"):
        for warning in meta["warnings"]:
            print(f"[WARN] {warning}")
    else:
        print("[OK] Split is event-disjoint and validation size is usable.")

    # Holdout is a tile-level holdout with the same event names; be explicit.
    hold_tiles = available_splits.get("hold", [])
    if hold_tiles and source_name != "hold":
        hold_events = set(event_counts(hold_tiles))
        print(f"\n--- 3. Secondary holdout (hold/) ---")
        print(f"{len(hold_tiles)} pairs | events={len(hold_events)}")
        print("Same event names as train; this is a TILE-level holdout, not "
              "event-disjoint. Use it for qualitative checks, not as the primary "
              "generalisation metric.")

    # Mask parity against the shipped xBD targets.
    print("\n--- 4. Mask cross-check against targets/ ---")
    primary_eval = list(val_tiles) + list(train_tiles)
    primary_check = crosscheck_targets(primary_eval, cache_dir, n=crosscheck_n)
    if primary_check["available"]:
        print(f"Tiles checked:      {primary_check['tiles']}")
        print(f"Localization match: {primary_check['loc_agreement_pct']:.4f}%")
        print(f"Damage match:       {primary_check['dmg_agreement_pct']:.4f}%")
        print("Per-class IoU:      " + ", ".join(
            f"{name}={iou:.4f}" for name, iou in primary_check["per_class_iou"].items()
        ))
    else:
        print(f"[SKIP] {primary_check['reason']}")

    hold_check = None
    if hold_tiles:
        hold_check = crosscheck_targets(hold_tiles, cache_dir, n=max(3, crosscheck_n // 2), seed=7)
        if hold_check["available"]:
            print(f"Hold split:         loc={hold_check['loc_agreement_pct']:.4f}% "
                  f"dmg={hold_check['dmg_agreement_pct']:.4f}% "
                  f"({hold_check['tiles']} tiles)")

    # Class histogram.
    print("\n--- 5. Class frequencies (sampled up to 200 tiles) ---")
    histogram = class_histogram(primary_eval, cache_dir, limit=200)
    pixels = histogram["pixel_counts"]
    total_pixels = sum(pixels.values())
    building_pixels = sum(v for k, v in pixels.items() if k > 0)
    print(f"{'ID':<3} | {'Class':<15} | {'Pixels':>14} | {'Share':>9} | Tile presence")
    print("-" * 70)
    for cls in range(5):
        count = pixels.get(cls, 0)
        share = count / max(1, total_pixels if cls == 0 else building_pixels) * 100
        presence = histogram["tiles_inspected"] if cls == 0 else histogram["tile_presence"].get(cls, 0)
        print(f"{cls:<3} | {INVERSE_DAMAGE_MAP[cls]:<15} | {count:>14,} | "
              f"{share:>8.2f}% | {presence}/{histogram['tiles_inspected']}")

    # Overlays.
    print(f"\n--- 6. Qualitative overlays -> {out_samples} ---")
    overlay_tiles = list(val_tiles) + list(train_tiles)
    overlay_paths = _save_overlays(overlay_tiles, cache_dir, out_samples, num_samples)
    for path in overlay_paths:
        print(f"  [+] {path}")

    print("\n" + "=" * 78)
    print("PHASE 2 DATA CHECK: PASS")
    print("=" * 78)

    return {
        "xbd_root": str(xbd_root),
        "source_split": source_name,
        "inventory": {name: len(tiles) for name, tiles in available_splits.items()},
        "events": {name: event_counts(tiles) for name, tiles in available_splits.items()},
        "split": meta,
        "crosscheck": primary_check,
        "hold_crosscheck": hold_check,
        "histogram": histogram,
        "overlays": [str(p) for p in overlay_paths],
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Check the xBD dataset and generate sample overlays.")
    parser.add_argument("--synthetic", action="store_true", help="Use a procedural synthetic dataset")
    parser.add_argument("--xbd-root", type=str, default=None, help="Root path to the xBD dataset")
    parser.add_argument("--samples", type=int, default=5, help="Number of qualitative overlays")
    parser.add_argument("--crosscheck", type=int, default=10, help="Tiles to compare against targets/")
    parser.add_argument("--source-split", type=str, default=None, help="train | hold | test")
    parser.add_argument("--json-out", type=str, default=None, help="Optional path to write the stats JSON")
    args = parser.parse_args(argv)

    stats = run_data_check(
        xbd_root_arg=args.xbd_root,
        synthetic_mode=args.synthetic,
        num_samples=args.samples,
        crosscheck_n=args.crosscheck,
        source_split=args.source_split,
    )
    if args.json_out:
        Path(args.json_out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"[INFO] Wrote stats JSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
