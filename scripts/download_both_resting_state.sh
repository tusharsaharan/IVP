#!/usr/bin/env bash
# Install required tools (if needed), then download both NeuroQ-Connect resting-state subsets.
set -Eeuo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ROOT="${1:-$HOME/neuroq-eeg-downloads}"

export BUN_INSTALL="${BUN_INSTALL:-$HOME/.bun}"
export PATH="$BUN_INSTALL/bin:$HOME/.local/bin:$PATH"

if ! command -v aws >/dev/null 2>&1 || ! command -v nemar >/dev/null 2>&1; then
  bash "$SCRIPT_DIR/install_neuroq_download_tools.sh"
  export PATH="$BUN_INSTALL/bin:$HOME/.local/bin:$PATH"
fi

bash "$SCRIPT_DIR/download_openneuro_ds006780_resting.sh" "$ROOT"
bash "$SCRIPT_DIR/download_nemar_on005512_resting.sh" "$ROOT"

echo
echo "Both resting-state subsets have been downloaded:"
echo "  $ROOT/ds006780"
echo "  $ROOT/on005512"
