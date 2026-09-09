#!/usr/bin/env bash
# Upload a built flex-pi snapshot to a Hugging Face dataset repo (private by default).
# Usage: bash scripts/upload_flexpi_dataset.sh <org-or-user>/<repo> /home/evan/yam_data/<snapshot_dir> [--public]
# Requires: `hf auth login` done once on this machine (write token).
set -euo pipefail
REPO="${1:?repo id, e.g. LoriCai99/put_pen_in_bag}"
DIR="${2:?snapshot directory}"
VIS="--private"; [[ "${3:-}" == "--public" ]] && VIS=""
[[ -f "$DIR/meta/info.json" ]] || { echo "not a dataset dir: $DIR"; exit 1; }
hf auth whoami >/dev/null 2>&1 || { echo "not logged in: run 'hf auth login' first"; exit 1; }
hf repo create "$REPO" --repo-type dataset $VIS 2>/dev/null || true   # idempotent
# Resumable, parallel, chunked; safe to re-run after an interruption.
hf upload-large-folder "$REPO" "$DIR" --repo-type dataset
echo "uploaded -> https://huggingface.co/datasets/$REPO"
