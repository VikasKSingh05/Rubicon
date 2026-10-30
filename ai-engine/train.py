"""Rubicon Siamese U-Net Training Pipeline (ai-engine entrypoint)."""
import sys
from pathlib import Path

# Ensure root workspace is in sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from train import train

if __name__ == "__main__":
    train()
