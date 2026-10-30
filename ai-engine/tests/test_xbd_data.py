"""Unit tests for the Rubicon xBD data pipeline."""
import json
import tempfile
from pathlib import Path
import numpy as np
import pytest
import torch

from ai_engine.data.parser import parse_xbd_label, cache_tile_masks, DAMAGE_MAP
from ai_engine.data.splits import create_event_splits, discover_xbd_tiles
from ai_engine.data.synthetic import generate_synthetic_dataset
from ai_engine.data.dataset import XBDDataset, get_weighted_sampler


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
