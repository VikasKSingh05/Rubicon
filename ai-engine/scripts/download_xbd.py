#!/usr/bin/env python
"""Download or prepare the xBD (xView2) Satellite Disaster Assessment Dataset.

Dataset structure expected in data/xbd/:
  data/xbd/
    train/
      images/
        <disaster>_<id>_pre_disaster.png
        <disaster>_<id>_post_disaster.png
      labels/
        <disaster>_<id>_pre_disaster.json
        <disaster>_<id>_post_disaster.json
    holdout/
    test/

Usage:
  # Generate synthetic dataset for immediate testing:
  python scripts/download_xbd.py --synthetic --num-pairs 10

  # Print download links and instructions for full xBD archive:
  python scripts/download_xbd.py --info
"""
import argparse
import sys
from pathlib import Path


def print_info():
    print("=" * 70)
    print("xBD (xView2) Dataset Access Guide")
    print("=" * 70)
    print(
        "xBD is the premier high-resolution satellite imagery dataset for\n"
        "post-disaster building damage assessment, covering 19+ natural\n"
        "disaster events across 850,000+ building polygons worldwide.\n"
    )
    print("Official Download Sources:")
    print("  1. Official xView2 Challenge Site:")
    print("     https://xview2.org/download")
    print("     Files: train.tar.gz (~20GB), test.tar.gz (~5GB), holdout.tar.gz (~5GB)")
    print()
    print("  2. Hugging Face Mirrors:")
    print("     https://huggingface.co/datasets/dioptra-ai/xbd")
    print()
    print("Local Target Directory:")
    print("  ai-engine/data/xbd/")
    print()
    print("Quick Start (No download needed - Synthetic Data):")
    print("  python scripts/download_xbd.py --synthetic --num-pairs 10")
    print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description="xBD Dataset Setup & Download Helper")
    parser.add_argument("--info", action="store_true", help="Print download instructions")
    parser.add_argument("--synthetic", action="store_true", help="Generate synthetic xBD pairs locally")
    parser.add_argument("--num-pairs", type=int, default=10, help="Number of synthetic tile pairs to generate")
    parser.add_argument("--dest", type=str, default="data/synthetic_xbd", help="Destination path for synthetic dataset")
    args = parser.parse_args()

    if args.synthetic:
        from ai_engine.data.synthetic import generate_synthetic_dataset
        dest = Path(args.dest)
        print(f"Generating {args.num_pairs} synthetic xBD image pairs in '{dest}'...")
        generate_synthetic_dataset(output_dir=str(dest), num_pairs=args.num_pairs)
        print(f"Synthetic xBD dataset generated successfully at '{dest}'")
        return 0

    print_info()
    return 0


if __name__ == "__main__":
    sys.exit(main())
