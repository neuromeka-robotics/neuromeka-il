"""Generate/preprocess with eir_data.yaml; train with eir_nominal_command.yaml.

Generate preserves timing-gap inference, converts task recordings with arm-only
IK, and keeps contiguous successful segments >=5 s. Original files are unchanged.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import yaml

from correction_config import DEFAULT_CONFIG, DEFAULT_DATA_CONFIG, load_settings, load_data_settings, require_empty


def generate_commands(paths, snapshot):
    pace, work = paths["pace"], paths["work"]
    post, ik = work / "postprocessed", work / "ik/nami_task_abs"
    def command(script, *arguments):
        return [sys.executable, str(pace / "genesis_bridge" / script), "--config", str(snapshot), *map(str, arguments)]
    stages = [("joint_timing", command("repair_recording_timing.py", "--data", paths["joint"],
                                       "--output", post / "nami_joint_abs"))]
    stages.extend((f"task_ik_{i}", command("convert_task_to_joint.py", "--data", source,
                   "--output", ik, "--ik-joints", "arms")) for i, source in enumerate(paths["tasks"]))
    if paths["tasks"]:
        stages.append(("task_timing", command("repair_recording_timing.py", "--data", ik,
                                              "--output", post / "nami_task_abs")))
    stages.append(("labels", command("replay_teleop.py", "--data", post, "--prepare-only",
                                    "--output", paths["labels"])))
    return stages


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("generate", "preprocess", "train", "all"))
    parser.add_argument("--config", type=Path,
                        help="Stage config: eir_data.yaml for generate/preprocess, eir_nominal_command.yaml for train/all")
    parser.add_argument("--data-config", type=Path, help="Override eir_data.yaml when using all")
    parser.add_argument("--run", help="Override train.run_name (train/all only)")
    parser.add_argument("--device", help="Override train.device (train/all only)")
    parser.add_argument("--dry-run", action="store_true", help="Print resolved paths/commands without writing")
    args = parser.parse_args()
    if (args.run or args.device) and args.stage not in ("train", "all"):
        parser.error("--run and --device apply to train/all")
    if args.data_config and args.stage != "all":
        parser.error("--data-config applies to all; use --config for a single stage")
    try:
        paths = {}
        data_config = (args.data_config or DEFAULT_DATA_CONFIG) if args.stage == "all" else (args.config or DEFAULT_DATA_CONFIG)
        data_config = data_config.expanduser().resolve()
        if args.stage in ("generate", "preprocess", "all"):
            settings, paths = load_data_settings(data_config)
        config = (args.config or DEFAULT_CONFIG).expanduser().resolve()
        if args.stage in ("train", "all"):
            _, training_paths = load_settings(config, args.run)
            if args.stage == "all" and paths["processed"] != training_paths["processed"]:
                raise ValueError("Data preparation and training configs must point to the same processed_dir for all")
            paths.update(training_paths)
        train_dir = Path(__file__).resolve().parent
        stages = []
        if args.stage in ("generate", "all"):
            snapshot = paths["work"] / "configs/eir_train.yaml"
            stages += generate_commands(paths, snapshot)
        if args.stage in ("preprocess", "all"):
            stages.append(("preprocess", [sys.executable, str(train_dir / "preprocess_correction.py"), "--config", str(data_config)]))
        if args.stage in ("train", "all"):
            command = [sys.executable, str(train_dir / "imitate_correction.py"), "--config", str(config)]
            if args.run:
                command += ["--run", args.run]
            if args.device:
                command += ["--device", args.device]
            stages.append(("train", command))
        for key in ("labels", "processed", "weights"):
            if key in paths:
                print(f"{key.capitalize()}: {paths[key]}", flush=True)
        if args.dry_run:
            for name, command in stages:
                print(f"{name}: {shlex.join(command)}")
            return
        # Validate all destinations before any multi-stage work starts.
        if args.stage in ("generate", "all"):
            for output in (paths["work"] / "ik", paths["work"] / "postprocessed", paths["labels"], paths["work"] / "configs", paths["work"] / "reports"):
                require_empty(output)
            # Export/preprocessing must not write into source directories or each other.
            outputs = [paths["work"] / "ik", paths["work"] / "postprocessed", paths["labels"], paths["processed"]]
            sources = [paths["joint"], *paths["tasks"]]
            for i, output in enumerate(outputs):
                for other in [*sources, *outputs[i+1:]]:
                    if output == other or output in other.parents or other in output.parents:
                        raise ValueError(f"Data paths must not overlap: {output}, {other}")
            files = sorted(paths["joint"].rglob("*.h5")) + sorted(paths["joint"].rglob("*.hdf5"))
            if not files:
                raise ValueError(f"No joint recordings in {paths['joint']}")
            files += paths["tasks"]
            for source in [*files, paths["robot_config"]]:
                if not source.is_file():
                    raise ValueError(f"Missing source: {source}")
        if args.stage in ("preprocess", "all"):
            require_empty(paths["processed"])
        if args.stage in ("train", "all"):
            require_empty(paths["weights"])
        manifest = None
        if args.stage in ("generate", "all"):
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            (paths["work"] / "reports").mkdir(parents=True, exist_ok=True)
            robot = yaml.safe_load(paths["robot_config"].read_text())
            robot["urdf"] = str((paths["robot_config"].parent / Path(robot["urdf"]).expanduser()).resolve())
            snapshot.write_text(yaml.safe_dump(robot, sort_keys=False))
            # Snapshot absolute data roots so relocating the config cannot change paths.
            settings["data"].update(root=str(paths["root"]), work_dir=str(paths["work"]), pace_dir=str(paths["pace"]),
                joint_dir=str(paths["joint"]), task_files=[str(p) for p in paths["tasks"]],
                robot_config=str(paths["robot_config"]), labels_dir=str(paths["labels"]), processed_dir=str(paths["processed"]))
            (snapshot.parent / "eir_data.yaml").write_text(yaml.safe_dump(settings, sort_keys=False))
            manifest = {"status": "generating", "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
                        "ik_joints": "arms", "minimum_task_segment_duration_s": 5., "stages": []}
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        for name, command in stages:
            print(f"{name}: {shlex.join(command)}", flush=True)
            # Retain conversion/timing diagnostics alongside the dataset.
            if manifest is not None and name not in ("preprocess", "train"):
                with (paths["work"] / "reports" / f"{name}.log").open("w") as log:
                    process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                    code = process.wait()
                manifest["stages"].append({"name": name, "command": command, "exit_code": code})
                manifest["status"] = "failed" if code else "generated" if name == "labels" else "generating"
                (paths["work"] / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
                if code:
                    raise RuntimeError(f"{name} failed; see {paths['work'] / 'reports'}")
            else:
                subprocess.run(command, env=env, check=True)
    except (ValueError, KeyError, OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
