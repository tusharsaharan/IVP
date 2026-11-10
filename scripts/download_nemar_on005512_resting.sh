#!/usr/bin/env bash
# Download ONLY HBN Release 8 RestingState EEG from NEMAR.
# Uses NEMAR CLI --http so git, git-annex, DataLad and a NEMAR login are not required.
set -Eeuo pipefail

ROOT="${1:-$HOME/neuroq-eeg-downloads}"
DEST="$ROOT/on005512"
JOBS="${JOBS:-4}"

export BUN_INSTALL="${BUN_INSTALL:-$HOME/.bun}"
export PATH="$BUN_INSTALL/bin:$HOME/.local/bin:$PATH"

if ! command -v nemar >/dev/null 2>&1; then
  echo "ERROR: NEMAR CLI is not installed."
  echo "Run: bash install_neuroq_download_tools.sh"
  exit 1
fi

mkdir -p "$ROOT"

if [[ -e "$DEST/.git" ]]; then
  echo "ERROR: $DEST is an existing Git/DataLad clone."
  echo "Use a fresh destination for the HTTP snapshot, e.g.:"
  echo "  bash $0 \"$HOME/neuroq-eeg-downloads-http\""
  exit 1
fi

echo "Dataset: on005512 (Healthy Brain Network EEG - Release 8)"
echo "Task:    RestingState"
echo "Output:  $DEST"
echo "Workers: $JOBS"
echo
echo "Downloading only the RestingState EEG BIDS files plus required metadata."
echo "The full dataset is about 157 GB; this command filters to the RestingState EEG task."
echo "The HTTP snapshot can resume: rerun this same command after an interruption."
echo

nemar dataset download on005512 \
  --tasks RestingState \
  --datatypes eeg \
  --http \
  --output "$DEST" \
  --jobs "$JOBS"

if [[ ! -s "$DEST/dataset_description.json" ]]; then
  echo "ERROR: dataset_description.json is missing or empty in $DEST."
  exit 2
fi

SET_COUNT="$(find "$DEST" -type f -path '*/eeg/*' -name '*task-RestingState*_eeg.set' | wc -l | tr -d ' ')"
SUBJECT_COUNT="$(find "$DEST" -mindepth 1 -maxdepth 1 -type d -name 'sub-*' | wc -l | tr -d ' ')"

echo
echo "Verification:"
echo "  Subject folders found: $SUBJECT_COUNT"
echo "  RestingState EEGLAB SET payloads: $SET_COUNT"

if [[ "$SET_COUNT" -eq 0 ]]; then
  echo "ERROR: No RestingState SET payloads were downloaded."
  echo "Check the NEMAR CLI output above and confirm that the task filter matched."
  exit 3
fi

echo "SUCCESS: on005512 RestingState subset is at:"
echo "  $DEST"
