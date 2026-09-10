#!/usr/bin/env python
"""Fetch the Houston 2013 (IEEE GRSS DFC) dataset into data/houston2013/.

The canonical host occasionally rate-limits or changes; this script tries the
known mirrors in order, then prints manual-download instructions.

Expected files (put them in ai-engine/data/houston2013/):
  Houston_2013_HSI.tif    349 x 1905 x 144 bands, CASI, 2.5 m GSD
  Houston_2013_DSM.tif    LiDAR-derived surface height (rasterized)
  Houston_2013_GT.tif     ground truth, 15 land-cover classes (0 = background)

Then run:
  python -m src.preprocessing.dataset --config configs/phase2.yaml

Sources (documented, not exhaustively relied upon):
  - IEEE GRSS Data Fusion Contest 2013 page (University of Houston site)
  - Community mirrors used by Hyperspectral image repositories
"""
from __future__ import annotations

import shutil
import sys
import zipfile
from pathlib import Path

import requests

MIRRORS = [
    "https://hyperspectral.ee.uh.edu/wp-content/uploads/2015/06/Houston2013.zip",
    "https://www2.cs.uh.edu/~txie/ares/dataset/houston2013/Houston2013.zip",
]

# Common renames when archives contain differently-named files.
RENAMES = {
    "GRSS2013/houston_hsi.tif": "Houston_2013_HSI.tif",
    "houston_hsi.tif": "Houston_2013_HSI.tif",
    "GRSS2013/houston_lidar.tif": "Houston_2013_DSM.tif",
    "houston_lidar.tif": "Houston_2013_DSM.tif",
    "GRSS2013/houston_gt.tif": "Houston_2013_GT.tif",
    "houston_gt.tif": "Houston_2013_GT.tif",
}

CHUNK = 1024 * 256


def download(url: str, dest: Path, timeout: int = 30):
    with requests.get(url, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(CHUNK):
                fh.write(chunk)
    print(f"downloaded {dest.name} ({dest.stat().st_size / 1e6:.0f} MB)")


def main() -> int:
    root = Path("data/houston2013")
    root.mkdir(parents=True, exist_ok=True)

    target = root / "Houston2013.zip"
    for url in MIRRORS:
        try:
            download(url, target)
            break
        except Exception as exc:  # noqa: BLE001 - try next mirror
            print(f"mirror failed ({url}): {exc}", file=sys.stderr)
    else:
        print(
            "All mirrors failed. Download 'Houston2013.zip' manually and drop it "
            "in data/houston2013/ then re-run this script.",
            file=sys.stderr,
        )
        return 1

    with zipfile.ZipFile(target) as zf:
        zf.extractall(root)
    target.unlink()

    for src, rename in RENAMES.items():
        p = root / src
        if p.exists() and not (root / rename).exists():
            p.rename(root / rename)

    missing = [
        n
        for n in ("Houston_2013_HSI.tif", "Houston_2013_DSM.tif", "Houston_2013_GT.tif")
        if not (root / n).exists()
    ]
    if missing:
        print(f"Expected files missing after extraction: {missing}", file=sys.stderr)
        for f in root.rglob("*.tif"):
            print("  found:", f.relative_to(root))
        return 1
    print("Houston 2013 ready in", root)
    return 0


if __name__ == "__main__":
    sys.exit(main())