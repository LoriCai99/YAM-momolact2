#!/usr/bin/env bash
# Upload a built flex-pi snapshot to UW Kopah with s3cmd (credentials in ~/.s3cfg).
#
#   bash scripts/upload_flexpi_dataset_kopah.sh /home/evan/yam_data/<snapshot_dir> [s3://bucket/prefix]
#
# Default prefix is s3://rselab/datasets/yam/ ; the snapshot's own directory name becomes
# the last path element, so the team pulls it back with
#   s3cmd get -r s3://rselab/datasets/yam/<snapshot_dir>/ ./
# Re-running only transfers changed files (sync), and never deletes anything remote.
set -euo pipefail
DIR="${1:?snapshot directory}"; DIR="${DIR%/}"
PREFIX="${2:-s3://rselab/datasets/yam}"; PREFIX="${PREFIX%/}"
S3CMD="${S3CMD:-$HOME/.local/bin/s3cmd}"
[[ -f "$DIR/meta/info.json" ]] || { echo "not a dataset dir: $DIR"; exit 1; }
[[ -f "$HOME/.s3cfg" ]] || { echo "no ~/.s3cfg (Kopah credentials)"; exit 1; }
NAME="$(basename "$DIR")"
DEST="$PREFIX/$NAME/"
echo "-> $DEST"
"$S3CMD" sync --no-delete-removed --multipart-chunk-size-mb=64 --exclude '.cache/*' "$DIR/" "$DEST"
echo "verifying..."
LOCAL=$(find "$DIR" -type f -not -path '*/.cache/*' | wc -l)
REMOTE=$("$S3CMD" ls -r "$DEST" | wc -l)
echo "  local files: $LOCAL | objects on Kopah: $REMOTE"
[[ "$LOCAL" -eq "$REMOTE" ]] && echo "OK - pull with: s3cmd get -r $DEST ./" || { echo "MISMATCH - re-run to finish the sync"; exit 1; }
