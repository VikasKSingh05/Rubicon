#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "Setting up Rubicon environment..."

echo -e "\n--- Installing Backend Dependencies ---"
cd "$DIR/backend"
npm install

echo -e "\n--- Installing Agent Service Dependencies ---"
cd "$DIR/agent-service"
npm install

echo -e "\n--- Installing Frontend Dependencies ---"
cd "$DIR/frontend"
npm install

echo -e "\n--- Installing AI Engine Python Dependencies ---"
cd "$DIR/ai-engine"
python3 -m pip install -r requirements.txt || pip install -r requirements.txt

cd "$DIR"
echo -e "\nRubicon setup complete!"
