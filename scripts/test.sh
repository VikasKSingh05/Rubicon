#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Running Rubicon full test suite..."

echo -e "\n1/4: AI Engine (pytest)..."
cd "$DIR/ai-engine"
python3 -m pytest || pytest

echo -e "\n2/4: Backend tests (node --test)..."
cd "$DIR/backend"
npm test

echo -e "\n3/4: Agent Service tests (node --test)..."
cd "$DIR/agent-service"
npm test

echo -e "\n4/4: Frontend tests (vitest)..."
cd "$DIR/frontend"
npm test

cd "$DIR"
echo -e "\nAll test suites passed cleanly!"
