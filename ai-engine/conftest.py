"""pytest configuration for ai-engine tests."""
import sys
from pathlib import Path

ai_engine_dir = Path(__file__).resolve().parent
repo_root = ai_engine_dir.parent

for p in (ai_engine_dir, repo_root):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))
