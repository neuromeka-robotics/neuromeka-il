#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
# Activate env_il in the calling shell. No machine-specific pyenv paths.
exec python perception/detection_server.py "$@"
