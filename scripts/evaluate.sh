#!/usr/bin/env bash
# Usage: ./scripts/evaluate.sh [config-path] [checkpoint-path] [split]
set -euo pipefail

PYTHON="${PYTHON:-python}"
CONFIG="${1:-config/config.yaml}"
CKPT="${2:-outputs/checkpoint-best}"
SPLIT="${3:-test}"

echo "== Evaluating on $SPLIT split =="
"$PYTHON" -m src.evaluate --config "$CONFIG" --checkpoint "$CKPT" --split "$SPLIT"

echo "Done."
