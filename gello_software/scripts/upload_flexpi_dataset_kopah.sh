#!/usr/bin/env bash
# Upload a built flex-pi snapshot to UW Kopah (S3-compatible) with rclone.
# Usage:
#   export KOPAH_ACCESS_KEY=... KOPAH_SECRET_KEY=...        # never commit these
#   bash scripts/upload_flexpi_dataset_kopah.sh <bucket>[/prefix] /home/evan/yam_data/<snapshot_dir>
# Example: bash scripts/upload_flexpi_dataset_kopah.sh mylab-bucket/yam/put_pen_in_bag \
#              /home/evan/yam_data/put_pen_in_bag_flexpi_v21_2026-09-08_v2
# Resumable: re-running only transfers files that are missing or differ (size+checksum).
set -euo pipefail
DEST="${1:?bucket[/prefix]}"
DIR="${2:?snapshot directory}"
: "${KOPAH_ACCESS_KEY:?export KOPAH_ACCESS_KEY}"; : "${KOPAH_SECRET_KEY:?export KOPAH_SECRET_KEY}"
ENDPOINT="${KOPAH_ENDPOINT:-https://s3.kopah.uw.edu}"
[[ -f "$DIR/meta/info.json" ]] || { echo "not a dataset dir: $DIR"; exit 1; }
RCLONE="${RCLONE:-$HOME/.local/bin/rclone}"
# On-the-fly remote (no config file, keys stay in env): S3 provider "Other", path-style.
R=":s3,provider=Other,endpoint=${ENDPOINT},access_key_id=${KOPAH_ACCESS_KEY},secret_access_key=${KOPAH_SECRET_KEY},force_path_style=true:"
NAME="$(basename "$DIR")"
echo "-> ${ENDPOINT}  ${DEST}/${NAME}"
"$RCLONE" copy "$DIR" "${R}${DEST}/${NAME}" --exclude ".cache/**" \
    --transfers 8 --checkers 16 --s3-chunk-size 64M --s3-upload-concurrency 4 \
    --progress --stats 30s --stats-one-line
echo "verifying..."
"$RCLONE" check "$DIR" "${R}${DEST}/${NAME}" --exclude ".cache/**" --one-way && echo "OK: every local file is on Kopah with matching size/hash"
echo "listing: $RCLONE lsd \"${R}${DEST}\""
