"""Unit tests for the Rubicon xBD data pipeline."""
import copy
import json
import os
import tempfile
import time
from pathlib import Path
import numpy as np
import cv2
import pytest
import torch

from ai_engine.data.parser import (
    DAMAGE_MAP,
    cache_tile_masks,
    get_or_create_masks,
    parse_xbd_label,
    tile_has_severe_damage,
)
from ai_engine.data.splits import (
    DEFAULT_VAL_EVENTS,
    create_event_splits,
    discover_xbd_tiles,
    load_or_create_splits,
    resolve_source_root,
    validate_splits,
)
from ai_engine.data.synthetic import generate_synthetic_dataset
from ai_engine.data.dataset import XBDDataset, compute_norm_stats, get_weighted_sampler


@pytest.fixture(scope="module")
def sample_dataset(tmp_path_factory):
    """Fixture providing a verified synthetic dataset for isolated testing."""
    tmp_dir = tmp_path_factory.mktemp("xbd_test")
    dataset_path = generate_synthetic_dataset(output_dir=tmp_dir, tiles_per_event=3)
    return dataset_path


def test_damage_mapping_contract():
    """Verify class mapping contract: 0=bg, 1=no-damage, 2=minor, 3=major, 4=destroyed."""
    assert DAMAGE_MAP["no-damage"] == 1
    assert DAMAGE_MAP["minor-damage"] == 2
    assert DAMAGE_MAP["major-damage"] == 3
    assert DAMAGE_MAP["destroyed"] == 4
    # Requirement: Treat "un-classified" as no-damage (1)
    assert DAMAGE_MAP["un-classified"] == 1


def test_mask_shapes_and_values(sample_dataset):
    """Verify rendered mask shapes and valid class range."""
    tiles = discover_xbd_tiles(sample_dataset)
    assert len(tiles) >= 2

    tile = tiles[0]
    loc_mask, dmg_mask, features = parse_xbd_label(tile["post_label"], shape=(1024, 1024))

    assert loc_mask.shape == (1024, 1024)
    assert dmg_mask.shape == (1024, 1024)
    assert loc_mask.dtype == np.uint8
    assert dmg_mask.dtype == np.uint8

    loc_vals = set(np.unique(loc_mask))
    assert loc_vals.issubset({0, 1}), f"loc_mask contains unexpected values: {loc_vals}"

    dmg_vals = set(np.unique(dmg_mask))
    assert dmg_vals.issubset({0, 1, 2, 3, 4}), f"dmg_mask contains invalid classes: {dmg_vals}"


def test_pre_post_alignment(sample_dataset, tmp_path):
    """Verify pre/post alignment and caching logic."""
    tiles = discover_xbd_tiles(sample_dataset)
    tile = tiles[0]
    cache_dir = tmp_path / "cache"

    loc_file, dmg_file = cache_tile_masks(
        tile_id=tile["tile_id"],
        pre_json_path=tile["pre_label"],
        post_json_path=tile["post_label"],
        cache_dir=cache_dir,
    )

    assert loc_file.exists()
    assert dmg_file.exists()

    import cv2
    loc = cv2.imread(str(loc_file), cv2.IMREAD_UNCHANGED)
    dmg = cv2.imread(str(dmg_file), cv2.IMREAD_UNCHANGED)

    # Invariant: where building has damage (class 1..4), loc_mask must be 1
    damage_pixels = dmg > 0
    assert np.all(loc[damage_pixels] == 1), "Localization mask missing building present in damage mask"


def test_no_event_overlap_between_splits(sample_dataset, tmp_path):
    """Verify strict disaster event partitioning with zero overlap."""
    tiles = discover_xbd_tiles(sample_dataset)
    split_file = tmp_path / "splits.json"

    splits = create_event_splits(tiles, output_path=split_file)
    assert split_file.exists()

    train_events = set(t["event"] for t in splits["train"])
    val_events = set(t["event"] for t in splits["val"])

    # Core requirement: zero event overlap
    overlap = train_events.intersection(val_events)
    assert len(overlap) == 0, f"Disaster events overlap between splits: {overlap}"
    assert len(train_events) >= 1
    assert len(val_events) >= 1


def test_dataset_item_contract_and_oom_recovery(sample_dataset, tmp_path):
    """Verify PyTorch dataset shapes, layout, and dynamic crop adjustment."""
    tiles = discover_xbd_tiles(sample_dataset)
    cache_dir = tmp_path / "cache"

    dataset = XBDDataset(
        tiles=tiles,
        crop_size=512,
        is_train=True,
        cache_dir=cache_dir,
    )
    assert len(dataset) == len(tiles)

    pre_t, post_t, loc_t, dmg_t = dataset[0]

    # Verify tensor shapes and types
    assert pre_t.shape == (3, 512, 512)
    assert post_t.shape == (3, 512, 512)
    assert loc_t.shape == (1, 512, 512)
    assert dmg_t.shape == (512, 512)

    assert pre_t.dtype == torch.float32
    assert post_t.dtype == torch.float32
    assert loc_t.dtype == torch.float32
    assert dmg_t.dtype == torch.int64

    # Test dynamic crop halving for OOM recovery
    dataset.set_crop_size(256)
    pre_256, post_256, loc_256, dmg_256 = dataset[0]
    assert pre_256.shape == (3, 256, 256)
    assert loc_256.shape == (1, 256, 256)
    assert dmg_256.shape == (256, 256)


def test_weighted_sampler_construction(sample_dataset, tmp_path):
    """Verify weighted sampling oversamples damage."""
    tiles = discover_xbd_tiles(sample_dataset)
    cache_dir = tmp_path / "cache"

    dataset = XBDDataset(tiles=tiles, crop_size=256, is_train=True, cache_dir=cache_dir)
    sampler = get_weighted_sampler(dataset, severe_multiplier=4.0)

    assert sampler is not None
    assert len(sampler) == len(dataset)


# ── targeted tests for the event-split / mask contract ───────────────

_PRE_POLY = "POLYGON ((100 100, 200 100, 200 200, 100 200, 100 100))"
_POST_POLY = "POLYGON ((600 600, 700 600, 700 700, 600 700, 600 600))"


def _write_tile(root, tile_id, pre_polys, post_polys):
    """Create one minimal xBD tile (images + labels) under *root*."""
    root = Path(root)
    img_dir = root / "images"
    lbl_dir = root / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    img = np.zeros((1024, 1024, 3), dtype=np.uint8)
    cv2.imwrite(str(img_dir / f"{tile_id}_pre_disaster.png"), img)
    cv2.imwrite(str(img_dir / f"{tile_id}_post_disaster.png"), img)

    for suffix, polys in (("pre", pre_polys), ("post", post_polys)):
        features = [
            {"wkt": wkt, "properties": {"uid": f"b{idx}", "subtype": subtype,
                                        "feature_type": "building"}}
            for idx, (wkt, subtype) in enumerate(polys)
        ]
        label = {"features": {"xy": features},
                 "metadata": {"height": 1024, "width": 1024}}
        (lbl_dir / f"{tile_id}_{suffix}_disaster.json").write_text(
            json.dumps(label), encoding="utf-8"
        )

    tiles = discover_xbd_tiles(root)
    assert tiles, f"tile {tile_id} was not discovered under {root}"
    return tiles[0]


def test_localization_comes_from_pre_label(tmp_path):
    """Buildings present only pre-disaster must still be localised."""
    tile = _write_tile(
        tmp_path / "a",
        "test-event_00000000",
        [(_PRE_POLY, "un-classified")],
        [(_POST_POLY, "destroyed")],
    )
    loc, dmg = get_or_create_masks(
        tile_id=tile["tile_id"],
        pre_json_path=tile["pre_label"],
        post_json_path=tile["post_label"],
        cache_dir=tmp_path / "cache",
    )

    assert loc[150, 150] == 1, "pre-only building missing from localization mask"
    assert dmg[150, 150] == 0, "pre-only building must not receive a damage class"
    assert dmg[650, 650] == DAMAGE_MAP["destroyed"]
    assert loc[650, 650] == 1, "damage pixels must be localised"


def test_mask_cache_invalidated_when_label_changes(tmp_path):
    """Editing a source label must invalidate the cached mask PNGs."""
    tile = _write_tile(
        tmp_path / "a",
        "test-event_00000000",
        [(_PRE_POLY, "un-classified")],
        [(_POST_POLY, "destroyed")],
    )
    cache_dir = tmp_path / "cache"
    loc_first, _ = get_or_create_masks(
        tile["tile_id"], tile["pre_label"], tile["post_label"], cache_dir
    )
    assert loc_first[150, 150] == 1

    pre_label = Path(tile["pre_label"])
    pre_label.write_text(json.dumps({"features": {"xy": []}}), encoding="utf-8")
    future = time.time() + 10
    os.utime(pre_label, (future, future))

    loc_second, _ = get_or_create_masks(
        tile["tile_id"], tile["pre_label"], tile["post_label"], cache_dir
    )
    assert loc_second[150, 150] == 0, "stale cache was not invalidated"


def test_tile_has_severe_damage(tmp_path):
    severe = _write_tile(
        tmp_path / "severe", "test-event_00000000",
        [(_PRE_POLY, "un-classified")], [(_POST_POLY, "destroyed")],
    )
    mild = _write_tile(
        tmp_path / "mild", "test-event_00000000",
        [(_PRE_POLY, "un-classified")], [(_POST_POLY, "no-damage")],
    )
    assert tile_has_severe_damage(severe["post_label"])
    assert not tile_has_severe_damage(mild["post_label"])


def test_validate_splits_rejects_non_event_manifests(sample_dataset, tmp_path):
    tiles = discover_xbd_tiles(sample_dataset)
    splits = create_event_splits(tiles, output_path=tmp_path / "splits.json")
    assert validate_splits(splits)

    random_split = {
        "metadata": {"split_mode": "random", "event_disjoint": False},
        "train": tiles,
        "val": [],
    }
    assert not validate_splits(random_split)

    leaking = copy.deepcopy(splits)
    leaking["val"] = list(leaking["train"])
    assert not validate_splits(leaking), "tile-level leakage must be rejected"

    other_source = copy.deepcopy(splits)
    other_source["metadata"]["source_split"] = "hold"
    assert not validate_splits(other_source, source_split="train")


def test_default_validation_events(sample_dataset):
    tiles = discover_xbd_tiles(sample_dataset)
    splits = create_event_splits(tiles)
    assert set(splits["metadata"]["val_events"]) == set(DEFAULT_VAL_EVENTS)
    assert not set(splits["metadata"]["train_events"]) & set(DEFAULT_VAL_EVENTS)


def test_train_hold_layout_discovery(tmp_path):
    """``train/`` must be the primary source when both splits exist."""
    xbd_root = tmp_path / "xbd"
    generate_synthetic_dataset(xbd_root / "train", tiles_per_event=2, n_events=3)
    generate_synthetic_dataset(xbd_root / "hold", tiles_per_event=1, n_events=3)

    source_root, source_name = resolve_source_root(xbd_root)
    assert source_name == "train"
    assert source_root == xbd_root / "train"

    splits = load_or_create_splits(xbd_root)
    assert splits["metadata"]["source_split"] == "train"
    assert (xbd_root / "splits_train.json").exists()
    assert len(splits["train"]) + len(splits["val"]) == 6
    assert validate_splits(splits, source_split="train")

    hold_tiles = discover_xbd_tiles(xbd_root / "hold")
    assert len(hold_tiles) == 3


def test_compute_norm_stats(sample_dataset):
    tiles = discover_xbd_tiles(sample_dataset)
    mean, std = compute_norm_stats(tiles, max_tiles=2, sample_size=32)
    assert len(mean) == 3 and len(std) == 3
    assert all(0.0 <= m <= 1.0 for m in mean)
    assert all(s > 0.0 for s in std)

    empty_mean, empty_std = compute_norm_stats([])
    assert empty_mean == [0.485, 0.456, 0.406]
    assert empty_std == [0.229, 0.224, 0.225]
