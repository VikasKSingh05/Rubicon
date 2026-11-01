"""Compatibility shim for running Rubicon scripts from the repository root.

The canonical package lives in ``ai-engine/ai_engine``.  Root-level entry
points (``train.py``, ``eval.py``) and ``python -m ai_engine.data.check``
import ``ai_engine``; this shim extends the package search path so there is
exactly one implementation instead of two drifting copies.
"""
from pathlib import Path

_CANONICAL = Path(__file__).resolve().parent.parent / "ai-engine" / "ai_engine"
if _CANONICAL.is_dir():
    __path__.append(str(_CANONICAL))
