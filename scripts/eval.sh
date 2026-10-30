#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DIR/ai-engine"
python3 eval.py "$@" || python eval.py "$@"
