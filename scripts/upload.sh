#!/usr/bin/env bash
# Usage: ./scripts/upload.sh <hf-repo-id> [checkpoint-path] [hf-token]
set -euo pipefail

PYTHON="${PYTHON:-python}"
REPO_ID="${1:?Usage: ./scripts/upload.sh <hf-repo-id> [checkpoint-path] [hf-token]}"
CKPT="${2:-outputs/checkpoint-best}"
TOKEN="${3:-${HF_TOKEN:-}}"

echo "== Uploading to the Hub: $REPO_ID =="
if [ -n "$TOKEN" ]; then
    "$PYTHON" -m src.upload_to_hub --checkpoint "$CKPT" --repo-id "$REPO_ID" --token "$TOKEN"
else
    "$PYTHON" -m src.upload_to_hub --checkpoint "$CKPT" --repo-id "$REPO_ID"
fi

echo "Done."
