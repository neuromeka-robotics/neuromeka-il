#!/usr/bin/env bash
# Generate/normalize a shared joint-command dataset, then train YAML-selected designs.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash run_correction.sh {repair|generate|preprocess|train|all} [--run NAME] [options]

Run this after activating your Python environment (for example, conda activate pace).
Generate/preprocess once into teleop_common. --run selects the checkpoint name.

Options:
  --run NAME                  Checkpoint name (default: eir_correction).
  --data PATH                 Source teleoperation H5/PT file or directory.
                              Default: ~/Desktop/pace_eir_data/data/nami_joint_abs
  --repair-timing             Infer recording gaps and insert H5 samples before generate/all.
                              Requires raw H5 with q/qdot/qdes/qdotdes; timing is estimated.
  --repaired-dir PATH         Repair output (new/empty directory).
                              Default: PACE/genesis_bridge/logs/timing_repaired/teleop_common
  --pace-dir PATH             pace-sim2real checkout (default: sibling checkout).
  --replay-config PATH        Default: PACE/genesis_bridge/configs/eir_train.yaml
  --train-config PATH         Select model inputs/outputs for train.
                              Default: train/config/correction/eir.yaml
                              Position: train/config/correction/eir_position.yaml
  --raw-dir PATH              Override PACE/genesis_bridge/logs/teleop_residual/teleop_common
  --processed-dir PATH        Override train/processed_data/teleop_common
  --checkpoint-dir PATH       Override train/weights/NAME
  --python EXECUTABLE         Python executable (default: python from active environment).
  --validation-fraction FLOAT Preprocessing validation fraction (default: 0; all recordings train).
  --seed INTEGER              Preprocessing split seed (default: 0).
  --epochs INTEGER            Override training epochs; otherwise use the YAML.
  --device DEVICE             Training device, e.g. cuda, cuda:0, cpu (default: auto).
  --wandb                     Enable W&B training logs; settings come from the YAML.
  --dry-run                   Print commands without executing or creating outputs.
  -h, --help                  Show this help.

Examples:
  bash run_correction.sh generate --dry-run
  bash run_correction.sh repair --repaired-dir /path/to/repaired
  bash run_correction.sh generate --repair-timing --raw-dir /path/to/new_labels
  bash run_correction.sh generate --data /path/to/repaired --raw-dir /path/to/new_labels
  bash run_correction.sh all --run eir_run1
  bash run_correction.sh generate --run eir_run1 --data /path/to/recordings
  bash run_correction.sh preprocess --run eir_run1
  bash run_correction.sh train --run eir_run1 --device cuda
  bash run_correction.sh all --run smoke --epochs 1 --device cpu

Generation uses --prepare-only (no Genesis simulation). Each selected output
directory must be new or empty. Training requires the normalized processed data.
The repair stage only writes repaired H5; --repair-timing chains repair before
generation. Reuse repaired H5 with --data. Choose fresh raw/processed directories
when regenerating labels so they stay separate from existing training data.
EOF
}

die() { printf 'Error: %s\n' "$*" >&2; exit 1; }

stage="${1:-}"
case "$stage" in
    -h|--help|"") usage; exit 0 ;;
    repair|generate|preprocess|train|all) shift ;;
    *) die "Unknown stage '$stage'; choose repair, generate, preprocess, train, or all." ;;
esac

train_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
pace_dir="$(dirname -- "$(dirname -- "$train_dir")")/pace-sim2real"
data_path="${HOME}/Desktop/pace_eir_data/data/nami_joint_abs"
run_name="eir_correction"
replay_config=""
train_config="$train_dir/config/correction/eir.yaml"
raw_dir=""
processed_dir=""
checkpoint_dir=""
repaired_dir=""
python_bin="python"
validation_fraction="0."
split_seed="0"
epochs=""
device="auto"
dry_run=false
wandb_logging=false
repair_timing=false

while (($#)); do
    case "$1" in
        -h|--help) usage; exit 0 ;;
        --dry-run) dry_run=true; shift; continue ;;
        --wandb) wandb_logging=true; shift; continue ;;
        --repair-timing) repair_timing=true; shift; continue ;;
        --run|--data|--pace-dir|--replay-config|--train-config|--raw-dir|--processed-dir|--checkpoint-dir|--repaired-dir|--python|--validation-fraction|--seed|--epochs|--device)
            (($# >= 2)) && [[ -n "$2" && "$2" != --* ]] || die "$1 requires a value."
            ;;
        *) die "Unknown option '$1'; use --help." ;;
    esac
    case "$1" in
        --run) run_name="$2" ;;
        --data) data_path="$2" ;;
        --pace-dir) pace_dir="$2" ;;
        --replay-config) replay_config="$2" ;;
        --train-config) train_config="$2" ;;
        --raw-dir) raw_dir="$2" ;;
        --processed-dir) processed_dir="$2" ;;
        --checkpoint-dir) checkpoint_dir="$2" ;;
        --repaired-dir) repaired_dir="$2" ;;
        --python) python_bin="$2" ;;
        --validation-fraction) validation_fraction="$2" ;;
        --seed) split_seed="$2" ;;
        --epochs) epochs="$2" ;;
        --device) device="$2" ;;
    esac
    shift 2
done

[[ "$run_name" =~ ^[a-zA-Z0-9_][a-zA-Z0-9_.-]*$ ]] ||
    die "--run must start with a letter, digit, or underscore and contain only letters, digits, _, -, or ."
[[ -z "$epochs" || "$epochs" =~ ^[1-9][0-9]*$ ]] || die "--epochs must be a positive integer."
[[ "$split_seed" =~ ^[0-9]+$ ]] || die "--seed must be a nonnegative integer."
replay_config="${replay_config:-$pace_dir/genesis_bridge/configs/eir_train.yaml}"
raw_dir="${raw_dir:-$pace_dir/genesis_bridge/logs/teleop_residual/teleop_common}"
processed_dir="${processed_dir:-$train_dir/processed_data/teleop_common}"
checkpoint_dir="${checkpoint_dir:-$train_dir/weights/$run_name}"
repaired_dir="${repaired_dir:-$pace_dir/genesis_bridge/logs/timing_repaired/teleop_common}"
if "$repair_timing" && [[ "$stage" != repair && "$stage" != generate && "$stage" != all ]]; then
    die "--repair-timing applies to repair, generate, or all; use --data to reuse repaired H5."
fi

require_file() { [[ -f "$1" ]] || die "Missing file: $1"; }
require_empty_output() (
    [[ ! -e "$1" ]] && exit 0
    [[ -d "$1" ]] || die "Output is not a directory: $1"
    shopt -s nullglob dotglob
    entries=("$1"/*)
    ((${#entries[@]} == 0)) || die "Output directory is not empty: $1. Choose a new output directory (or --run for checkpoints)."
)

# Check all selected destinations before starting a multi-stage run.
if ! "$dry_run"; then
    command -v "$python_bin" >/dev/null 2>&1 || die "Python not found: $python_bin. Activate pace or pass --python."
    if [[ "$stage" == repair ]] || "$repair_timing"; then
        require_file "$pace_dir/genesis_bridge/repair_recording_timing.py"
        require_file "$replay_config"
        [[ -e "$data_path" ]] || die "Source data does not exist: $data_path"
        require_empty_output "$repaired_dir"
        if [[ "$stage" == generate || "$stage" == all ]]; then
            destinations=("$raw_dir")
            if [[ "$stage" == all ]]; then destinations+=("$processed_dir" "$checkpoint_dir"); fi
            "$python_bin" - "$repaired_dir" "${destinations[@]}" <<'PY'
from pathlib import Path
import sys
repaired = Path(sys.argv[1]).expanduser().resolve()
for value in sys.argv[2:]:
    other = Path(value).expanduser().resolve()
    if repaired == other or repaired in other.parents or other in repaired.parents:
        raise SystemExit("Error: repaired H5 and generated/processed/checkpoint outputs must be separate directories.")
PY
        fi
    fi
    if [[ "$stage" == generate || "$stage" == all ]]; then
        require_file "$pace_dir/genesis_bridge/replay_teleop.py"
        require_file "$replay_config"
        [[ -e "$data_path" ]] || die "Source data does not exist: $data_path"
        require_empty_output "$raw_dir"
    fi
    if [[ "$stage" == preprocess || "$stage" == all ]]; then
        require_file "$train_dir/preprocess_correction.py"
        if [[ "$stage" == preprocess ]]; then
            [[ -d "$raw_dir" ]] || die "Raw data directory does not exist: $raw_dir"
        fi
        require_empty_output "$processed_dir"
    fi
    if [[ "$stage" == train || "$stage" == all ]]; then
        require_file "$train_dir/imitate_correction.py"
        require_file "$train_config"
        if [[ "$stage" == train ]]; then
            require_file "$processed_dir/normalization.json"
            require_file "$processed_dir/split.json"
        fi
        require_empty_output "$checkpoint_dir"
    fi
fi

run_command() {
    printf '$'
    printf ' %q' "$@"
    printf '\n'
    if ! "$dry_run"; then "$@"; fi
}

printf 'Run: %s | Stage: %s\n' "$run_name" "$stage"
if [[ "$stage" == repair ]] || "$repair_timing"; then
    run_command "$python_bin" "$pace_dir/genesis_bridge/repair_recording_timing.py" \
        --config "$replay_config" --data "$data_path" --output "$repaired_dir"
    data_path="$repaired_dir"
fi
if [[ "$stage" == generate || "$stage" == all ]]; then
    run_command "$python_bin" "$pace_dir/genesis_bridge/replay_teleop.py" \
        --config "$replay_config" --data "$data_path" --correction recorded_acc \
        --prepare-only --output "$raw_dir"
fi
if [[ "$stage" == preprocess || "$stage" == all ]]; then
    run_command "$python_bin" "$train_dir/preprocess_correction.py" \
        --input "$raw_dir" --output "$processed_dir" \
        --validation-fraction "$validation_fraction" --seed "$split_seed" --all-fields
fi
if [[ "$stage" == train || "$stage" == all ]]; then
    train_args=(--config "$train_config" --data "$processed_dir" --output "$checkpoint_dir")
    if [[ -n "$epochs" ]]; then train_args+=(--epochs "$epochs"); fi
    if [[ "$device" != auto ]]; then train_args+=(--device "$device"); fi
    if "$wandb_logging"; then train_args+=(--wandb); fi
    run_command "$python_bin" "$train_dir/imitate_correction.py" "${train_args[@]}"
fi
