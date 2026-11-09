#!/usr/bin/env bash
# Install the tools used by the NeuroQ-Connect public EEG download scripts.
# Intended for WSL (Ubuntu/Kali/Debian) or another Debian-based Linux.
set -Eeuo pipefail

SUDO=""
if [[ "$EUID" -ne 0 ]]; then
  if ! command -v sudo >/dev/null 2>&1; then
    echo "ERROR: sudo is required to install system packages."
    exit 1
  fi
  SUDO="sudo"
fi

if ! command -v curl >/dev/null 2>&1 || ! command -v aws >/dev/null 2>&1; then
  echo "Installing curl and AWS CLI..."
  if command -v apt-get >/dev/null 2>&1; then
    $SUDO apt-get update
    $SUDO apt-get install -y curl awscli
  else
    echo "ERROR: this installer currently supports apt-based Linux."
    echo "Install curl and AWS CLI v2 manually, then rerun."
    exit 1
  fi
fi

if ! command -v aws >/dev/null 2>&1; then
  echo "ERROR: AWS CLI was not found after installation."
  echo "Install AWS CLI using: https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html"
  exit 1
fi

if ! command -v bun >/dev/null 2>&1; then
  echo "Installing Bun..."
  curl -fsSL https://bun.sh/install | bash
fi

export BUN_INSTALL="${BUN_INSTALL:-$HOME/.bun}"
export PATH="$BUN_INSTALL/bin:$HOME/.local/bin:$PATH"

if ! command -v bun >/dev/null 2>&1; then
  echo "ERROR: Bun installation did not complete. Open a new WSL terminal and rerun."
  exit 1
fi

if ! command -v nemar >/dev/null 2>&1; then
  echo "Installing NEMAR CLI..."
  bun install -g nemar-cli
fi

# Persist Bun's PATH for future WSL terminals.
BASHRC="$HOME/.bashrc"
if ! grep -Fq 'export BUN_INSTALL="$HOME/.bun"' "$BASHRC" 2>/dev/null; then
  {
    echo ''
    echo '# Bun / NEMAR CLI'
    echo 'export BUN_INSTALL="$HOME/.bun"'
    echo 'export PATH="$BUN_INSTALL/bin:$HOME/.local/bin:$PATH"'
  } >> "$BASHRC"
fi

echo
echo "Installed tools:"
aws --version
bun --version
nemar --version
echo
echo "Ready. No AWS credentials or NEMAR login are needed for these public datasets."
