"""Normalize replay H5 corrections; no rotations, poses, images, or resampling."""
import argparse
from pathlib import Path
import sys

import yaml

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "nrmk_il/src"))

from nrmk_il.dataset.correction_dataset import preprocess_corrections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Directory of raw replay H5 files")
    parser.add_argument("--output", type=Path, required=True, help="New/empty normalized dataset directory")
    parser.add_argument("--validation-fraction", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--all-joints", action="store_true", help="Include inactive joints (default: active only)")
    parser.add_argument("--config", type=Path, help="Training YAML selecting inputs/output; default: legacy acceleration design")
    parser.add_argument("--all-fields", action="store_true",
                        help="Normalize every exported field once; select model features at training time")
    args = parser.parse_args()
    policy = None if args.config is None else yaml.safe_load(args.config.read_text())["policy"]
    manifest, stats = preprocess_corrections(args.input, args.output, args.validation_fraction,
                                            args.seed, active_only=not args.all_joints, policy_config=policy, all_fields=args.all_fields)
    print(f"Saved {len(manifest['train'])} training and {len(manifest['validation'])} validation recordings")
    print(f"{len(stats['joint_names'])} joints, dt={stats['physics_dt_s']:g} s; normalized float32 inputs/outputs")
    print(args.output)


if __name__ == "__main__":
    main()
