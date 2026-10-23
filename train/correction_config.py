"""Resolve correction data/training paths once, independently of the shell cwd."""
from pathlib import Path
import re

import yaml

DEFAULT_CONFIG = Path(__file__).parent / "config/correction/eir_nominal_command.yaml"
DEFAULT_DATA_CONFIG = Path(__file__).parent / "config/correction/eir_data.yaml"


def resolve_path(value, base):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else base / path).resolve()


def load_settings(config=DEFAULT_CONFIG, run=None):
    """Training needs only processed data, policy settings and training settings."""
    config = Path(config).expanduser().resolve()
    settings = yaml.safe_load(config.read_text())
    data, train = settings["data"], settings["train"]
    if settings["policy"].get("control_mode") != "nominal_command":
        raise ValueError("Use control_mode: nominal_command")
    run = run or train["run_name"]
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", run):
        raise ValueError("run_name must start with a letter, digit or underscore and contain only letters, digits, _, -, .")
    train["run_name"] = run
    paths = {"processed": resolve_path(data["processed_dir"], config.parent),
             "weights": resolve_path(train["output_dir"], config.parent) / run}
    # Saved training configs remain usable when moved into the weight directory.
    data["processed_dir"] = str(paths["processed"])
    train["output_dir"] = str(paths["weights"].parent)
    return settings, paths


def load_data_settings(config=DEFAULT_DATA_CONFIG):
    """Generation/preprocessing have a separate config and never need a model."""
    config = Path(config).expanduser().resolve()
    settings = yaml.safe_load(config.read_text())
    data = settings["data"]
    root = resolve_path(data["root"], config.parent)
    work = resolve_path(data["work_dir"], root)
    paths = {"root": root, "work": work,
             "pace": resolve_path(data["pace_dir"], config.parent),
             "joint": resolve_path(data["joint_dir"], root),
             "tasks": [resolve_path(p, root) for p in data["task_files"]],
             "labels": resolve_path(data["labels_dir"], work),
             "processed": resolve_path(data["processed_dir"], work)}
    paths["robot_config"] = resolve_path(data["robot_config"], paths["pace"])
    if not 0 <= data["validation_fraction"] < 1:
        raise ValueError("data.validation_fraction must be in [0, 1)")
    if not isinstance(data.get("seed", 0), int) or data.get("seed", 0) < 0:
        raise ValueError("data.seed must be a nonnegative integer")
    return settings, paths


def require_empty(path):
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise ValueError(f"Output must be new/empty: {path}. Change its path in the YAML or choose a new --run for training.")
