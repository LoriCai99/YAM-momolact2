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
hf repo create "$REPO" --repo-type dataset $VIS --exist-ok
# Enforce the visibility explicitly: on 2026-09-08 the create step yielded a PUBLIC repo.
python - "$REPO" "$VIS" <<'PY'
import sys
from huggingface_hub import HfApi
repo, vis = sys.argv[1], sys.argv[2]
api = HfApi(); api.update_repo_settings(repo, repo_type="dataset", private=(vis == "--private"))
print("visibility:", "private" if api.repo_info(repo, repo_type="dataset").private else "PUBLIC")
PY
# Resumable, parallel, chunked; safe to re-run after an interruption.
hf upload-large-folder "$REPO" "$DIR" --repo-type dataset
echo "uploaded -> https://huggingface.co/datasets/$REPO"
