#!/usr/bin/env bash
# Download ONLY the ds006780 Restingstate EEG task directly from OpenNeuro's public S3 bucket.
# Uses AWS CLI include/exclude filters; this avoids downloading the whole multi-paradigm dataset
# and avoids Git-Annex pointer files entirely.
set -Eeuo pipefail

ROOT="${1:-$HOME/neuroq-eeg-downloads}"
DEST="$ROOT/ds006780"

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: AWS CLI is missing. Run: bash install_neuroq_download_tools.sh"
  exit 1
fi

mkdir -p "$ROOT"

if [[ -e "$DEST/.git" ]]; then
  echo "ERROR: $DEST is an existing Git/DataLad clone."
  echo "Use a fresh destination for this direct-file download, e.g.:"
  echo "  bash $0 \"$HOME/neuroq-eeg-downloads-direct\""
  exit 1
fi

mkdir -p "$DEST"

echo "Dataset: ds006780 (SFARI_EEG multi-paradigm dataset)"
echo "Task:    Restingstate"
echo "Source:  s3://openneuro.org/ds006780/"
echo "Output:  $DEST"
echo
echo "Downloading real BDF payloads and matching task sidecars."
echo "Only paths containing task-Restingstate are selected, plus root-level BIDS metadata."
echo "This is a resumable sync; rerun the same command to continue."
echo

aws s3 sync "s3://openneuro.org/ds006780/" "$DEST" \
  --no-sign-request \
  --region us-east-1 \
  --exclude "*" \
  --include "*task-Restingstate*" \
  --include "dataset_description.json" \
  --include "participants.tsv" \
  --include "participants.json" \
  --include "README*" \
  --include "CHANGES" \
  --include ".bidsignore" \
  --no-progress

if [[ ! -s "$DEST/dataset_description.json" ]]; then
  echo "ERROR: dataset_description.json is missing or empty in $DEST."
  exit 2
fi

BDF_COUNT="$(find "$DEST" -type f -path '*/eeg/*' -name '*task-Restingstate*_eeg.bdf' | wc -l | tr -d ' ')"
SUBJECT_COUNT="$(find "$DEST" -mindepth 1 -maxdepth 1 -type d -name 'sub-*' | wc -l | tr -d ' ')"

echo
echo "Verification:"
echo "  Subject folders found: $SUBJECT_COUNT"
echo "  Restingstate BDF payloads: $BDF_COUNT"

if [[ "$BDF_COUNT" -eq 0 ]]; then
  echo "ERROR: No Restingstate BDF payloads were downloaded."
  echo "Check internet access to OpenNeuro's public S3 bucket and the AWS CLI error above."
  exit 3
fi

echo "SUCCESS: ds006780 Restingstate subset is at:"
echo "  $DEST"
