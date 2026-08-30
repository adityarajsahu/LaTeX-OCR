#!/usr/bin/env bash
# Usage: ./scripts/evaluate.sh [config-path] [checkpoint-path] [split]
set -euo pipefail

CONFIG="${1:-config/config.yaml}"
CKPT="${2:-outputs/checkpoint-best}"
SPLIT="${3:-test}"

echo "== Evaluating on $SPLIT split =="
python -m src.evaluate --config "$CONFIG" --checkpoint "$CKPT" --split "$SPLIT"

echo "Done."
