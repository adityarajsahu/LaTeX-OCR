#!/usr/bin/env bash
# Usage: ./scripts/train.sh [config-path]
set -euo pipefail

CONFIG="${1:-config/config.yaml}"
LOG_DIR="logs"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="${LOG_DIR}/train_${TIMESTAMP}.log"
LATEST_LOG="${LOG_DIR}/train.log"

mkdir -p "$LOG_DIR"

echo "== Starting training in background (nohup) =="
echo "Config:   $CONFIG"
echo "Log file: $LOG_FILE"

nohup python -m src.train --config "$CONFIG" > "$LOG_FILE" 2>&1 &
PID=$!

ln -sf "train_${TIMESTAMP}.log" "$LATEST_LOG"

echo "Training process started with PID: $PID"
echo "To monitor live logs, run: tail -f $LATEST_LOG"
