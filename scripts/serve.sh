#!/usr/bin/env bash
set -e

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [ "$1" == "--local" ]; then
    echo "Starting Rubicon services locally..."
    (cd "$DIR/ai-engine" && python3 -m uvicorn src.inference:app --host 0.0.0.0 --port 8000) &
    (cd "$DIR/backend" && npm start) &
    (cd "$DIR/agent-service" && npm start) &
    (cd "$DIR/frontend" && npm run dev) &
    wait
else
    echo "Starting Rubicon stack with Docker Compose..."
    cd "$DIR"
    if [ "$1" == "--build" ] || [ "$2" == "--build" ]; then
        docker compose up --build
    else
        docker compose up
    fi
fi
