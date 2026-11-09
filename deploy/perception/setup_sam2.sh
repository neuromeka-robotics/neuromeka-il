#!/usr/bin/env bash
set -euo pipefail
perception_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
sam2_dir="${SAM2_REPO:-$perception_dir/../third_party/sam2}"
sam2_revision=e3623cda3f748df999f76db663f08da7a884a820
if [[ -d "$sam2_dir/sam2" ]]; then
    echo "Using existing SAM2 source: $sam2_dir"
else
    git clone https://github.com/Gy920/segment-anything-2-real-time.git "$sam2_dir"
    git -C "$sam2_dir" checkout "$sam2_revision"
fi
python -m pip install -r "$perception_dir/requirements.txt"
echo "SAM2 source is ready. Set SAM2_CHECKPOINT to existing weights; this script downloads no weights."
