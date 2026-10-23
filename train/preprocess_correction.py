"""Normalize the five nominal-command arrays; paths/split settings come from YAML."""
import argparse
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent / "nrmk_il/src"))

from correction_config import DEFAULT_DATA_CONFIG, load_data_settings
from nrmk_il.dataset.correction_dataset import preprocess_corrections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_DATA_CONFIG)
    args = parser.parse_args()
    settings, paths = load_data_settings(args.config)
    data = settings["data"]
    manifest, stats = preprocess_corrections(paths["labels"], paths["processed"],
        validation_fraction=data["validation_fraction"], seed=data.get("seed", 0))
    print(f"Saved {len(manifest['train'])} training and {len(manifest['validation'])} validation episodes")
    print(f"{len(stats['joint_names'])} joints, dt={stats['physics_dt_s']:g} s; normalized float32 inputs/outputs")
    print(paths["processed"])


if __name__ == "__main__":
    main()
