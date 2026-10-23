#!/usr/bin/env bash
# Activate the pace environment first. Data and training settings live in YAML.
set -euo pipefail
train_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec python "$train_dir/correction_pipeline.py" "$@"
