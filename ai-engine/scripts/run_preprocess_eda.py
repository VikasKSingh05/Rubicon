"""CLI script to run Rubicon data loading, preprocessing, and export preprocessed dataset for EDA."""
import sys
from pathlib import Path

# Ensure ai-engine root is in sys.path
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.preprocessing.preprocess_eda_pipeline import run_preprocessing_pipeline

if __name__ == "__main__":
    config_path = ROOT / "configs" / "preprocess_eda.yaml"
    print(f"[run_preprocess_eda] Using configuration: {config_path}")
    run_preprocessing_pipeline(config_path)
