"""Shared normalized joint-command data with YAML-selected features."""
from pathlib import Path
import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


RAW_TO_KEY = {
    "answer_q": "observation.answer_q",
    "answer_v": "observation.answer_v",
    "nominal_on_answer": "observation.nominal_on_answer",
    "correction": "action.correction",
}
STATE_KEYS = list(RAW_TO_KEY.values())[:-1]
ACTION_KEY = RAW_TO_KEY["correction"]


# Each mode defines available causal inputs and the meaning of its outputs.
# State subsets/order may change in YAML; output order follows output_shapes.
MODE_FEATURES = {
    "acceleration_correction": (
        {"answer_q", "answer_v", "nominal_on_answer", "teleop_error", "teleop_target"},
        {"correction"}),
    "position_correction": (
        {"answer_q", "answer_v", "nominal_on_answer", "teleop_error", "teleop_target"},
        {"position_correction"}),
    "position_velocity_correction": (
        {"reference_q", "reference_v", "reference_teleop_error", "nominal_on_reference", "teleop_target"},
        {"delta_q", "delta_v"}),
    "nominal_command": (
        {"nominal_q", "nominal_v", "nominal_teleop_error", "teleop_target"},
        {"command_delta_q", "command_delta_v"}),
    "nominal_acceleration_command": (
        {"nominal_q", "nominal_v", "nominal_teleop_error", "teleop_target"},
        {"command_delta_acc"}),
    "direct_command": (
        {"robot_q", "robot_v", "robot_teleop_error", "teleop_target"},
        {"qdes_minus_teleop", "qdotdes"}),
}
ACTION_FIELDS = {field for _, outputs in MODE_FEATURES.values() for field in outputs}


def action_keys(action):
    return [action] if isinstance(action, str) else list(action)


def policy_features(settings):
    """Validate causal features and action semantics, keeping YAML token order."""
    mode = settings.get("control_mode")
    if mode not in MODE_FEATURES:
        raise ValueError(f"Unsupported joint-command control_mode: {mode}")
    available, expected = MODE_FEATURES[mode]
    states = settings.get("state_token_keys")
    outputs = list(settings.get("output_shapes", {}))
    if (not isinstance(states, list) or not states or len(set(states)) != len(states)
            or not set(states) <= {f"observation.{key}" for key in available}
            or set(settings.get("input_shapes", {})) != set(states)):
        raise ValueError(f"{mode}: state_token_keys/input_shapes must select from {sorted(available)}")
    if set(outputs) != {f"action.{key}" for key in expected}:
        raise ValueError(f"{mode}: output_shapes must contain {sorted(expected)}")
    return list(states), outputs[0] if len(outputs) == 1 else outputs


def dataset_features(stats):
    return stats.get("state_keys", STATE_KEYS), stats.get("action_keys", stats.get("action_key", ACTION_KEY))


def select_normalization(stats, settings):
    """Extract a model's statistics from a shared dataset without refitting them."""
    states, action = policy_features(settings)
    if (settings["control_mode"] in ("nominal_command", "nominal_acceleration_command")
            and stats.get("nominal_command_timing") != "post_integration"):
        raise ValueError("nominal_command requires post-integration labels; regenerate/preprocess the common dataset")
    if (settings["control_mode"] == "nominal_acceleration_command"
            and stats.get("command_delta_acc_target") != "position"):
        raise ValueError("nominal_acceleration_command requires position-matching command_delta_acc labels")
    keys = [*states, *action_keys(action)]
    missing = set(keys) - set(stats["features"])
    if missing:
        raise ValueError(f"Processed data missing {sorted(missing)}; generate/preprocess the common dataset")
    selected = {key: value for key, value in stats.items()
                if key not in ("features", "state_keys", "action_key", "action_keys", "common_data")}
    selected.update(state_keys=states, features={key: stats["features"][key] for key in keys})
    selected["action_key" if isinstance(action, str) else "action_keys"] = action
    return selected


class CorrectionNormalizer:
    """The only transform: per-joint (value - mean) / std, and its inverse."""

    def __init__(self, stats):
        self.stats = stats

    @classmethod
    def load(cls, path):
        return cls(json.loads(Path(path).read_text()))

    def _parameters(self, key, value):
        parameters = self.stats["features"][key]
        if isinstance(value, torch.Tensor):
            return tuple(torch.as_tensor(parameters[k], device=value.device, dtype=value.dtype)
                         for k in ("mean", "std"))
        return tuple(np.asarray(parameters[k], dtype=np.float64) for k in ("mean", "std"))

    def normalize(self, key, value):
        mean, std = self._parameters(key, value)
        return (value - mean) / std

    def denormalize(self, key, value):
        mean, std = self._parameters(key, value)
        return value * std + mean


def read_raw(path, active_only=True, raw_to_key=None):
    raw_to_key = RAW_TO_KEY if raw_to_key is None else raw_to_key
    with h5py.File(path, "r") as file:
        missing = set(raw_to_key) - set(file)
        if missing:
            raise ValueError(f"{path}: missing {sorted(missing)}; regenerate replay H5 with the current exporter")
        arrays = {key: np.asarray(file[key], dtype=np.float64) for key in raw_to_key}
        shape = next(iter(arrays.values())).shape
        if (len(shape) != 2 or min(shape) < 1
                or any(value.shape != shape or not np.isfinite(value).all() for value in arrays.values())):
            raise ValueError(f"{path}: all selected keys must have the same finite [transitions, joints] shape")
        attrs = dict(file.attrs)
    dt = float(attrs["physics_dt_s"])
    names = [x.decode() if isinstance(x, bytes) else str(x) for x in attrs["joint_names"]]
    active = np.asarray(attrs["active_joints"], dtype=bool)
    if not np.isfinite(dt) or dt <= 0 or len(names) != shape[1] or active.shape != (shape[1],):
        raise ValueError(f"{path}: invalid joint/timestep metadata")
    indices = np.flatnonzero(active) if active_only else np.arange(shape[1])
    if len(indices) == 0:
        raise ValueError(f"{path}: no selected joints")
    ticks = np.asarray(attrs["recorded_ticks"], dtype=np.int64)
    if ticks.ndim != 1 or len(ticks) < 2 or ticks[0] != 0 or ticks[-1] != shape[0] or np.any(np.diff(ticks) < 1):
        raise ValueError(f"{path}: invalid recorded interval boundaries")
    metadata = {"physics_dt_s": dt, "joint_names": [names[i] for i in indices],
                "source_joint_indices": indices.tolist(), "active_only": active_only}
    if {"command_delta_q", "command_delta_v", "command_delta_acc"} & set(raw_to_key):
        timing = str(attrs.get("nominal_command_timing", "pre_integration"))
        if timing not in ("pre_integration", "post_integration"):
            raise ValueError(f"{path}: invalid nominal_command_timing {timing!r}")
        metadata["nominal_command_timing"] = timing
    if "command_delta_acc" in raw_to_key:
        if attrs.get("command_delta_acc_target") != "position":
            raise ValueError(f"{path}: command_delta_acc must contain position-matching labels")
        metadata["command_delta_acc_target"] = "position"
    return {raw_to_key[key]: value[:, indices] for key, value in arrays.items()}, attrs, metadata


def preprocess_corrections(input_dir, output_dir, validation_fraction=0.2, seed=0, active_only=True,
                           policy_config=None, all_fields=False):
    """Fit statistics on whole training recordings, then normalize every split."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    states, action = (STATE_KEYS, ACTION_KEY) if policy_config is None else policy_features(policy_config)
    mapping = {key.split(".", 1)[1]: key for key in (*states, *action_keys(action))}
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    paths = sorted(input_dir.glob("*.h5"))
    if not paths:
        raise ValueError(f"No H5 correction recordings in {input_dir}")
    if all_fields:
        with h5py.File(paths[0], "r") as file:
            mapping = {key: f"{'action' if key in ACTION_FIELDS else 'observation'}.{key}" for key in file}
        states = [key for key in mapping.values() if key.startswith("observation.")]
        action = [key for key in mapping.values() if key.startswith("action.")]
        if not states or not action:
            raise ValueError("Common data must include observation and action fields")
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError("Output must differ from raw input directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {output_dir}")
    if len(paths) == 1 and validation_fraction:
        raise ValueError("Need two recordings for a held-out split; use --validation-fraction 0 for training only")
    order = np.random.default_rng(seed).permutation(len(paths))
    num_val = min(len(paths) - 1, max(1, round(len(paths) * validation_fraction))) if validation_fraction else 0
    validation = set(order[:num_val].tolist())
    train = [i for i in range(len(paths)) if i not in validation]
    metadata = None
    moments = {}
    # Validate every episode before writing, and fit stable pooled moments on train only.
    for i, path in enumerate(paths):
        arrays, _, current = read_raw(path, active_only, mapping)
        if metadata is None:
            metadata = current
        elif metadata != current:
            raise ValueError(f"{path}: joint order, active joints, physics_dt, or command timing differs between episodes")
        if i not in train:
            continue
        for key, values in arrays.items():
            count, mean, m2 = moments.get(key, (0, np.zeros(values.shape[1]), np.zeros(values.shape[1])))
            n = len(values)
            local_mean = values.mean(axis=0)
            delta = local_mean - mean
            total = count + n
            m2 = m2 + ((values - local_mean) ** 2).sum(axis=0) + delta**2 * count * n / total
            mean = mean + delta * n / total
            moments[key] = (total, mean, m2)
    stats = {**metadata, "schema": "normalized_correction_v1", "features": {},
             "training_sources": [str(paths[i].resolve()) for i in train]}
    if policy_config is not None or all_fields:
        stats["state_keys"] = list(states)
        stats["action_key" if isinstance(action, str) else "action_keys"] = action
    if all_fields:
        stats["common_data"] = True
    for key, (count, mean, m2) in moments.items():
        std = np.maximum(np.sqrt(m2 / count), 1e-6)
        stats["features"][key] = {"mean": mean.tolist(), "std": std.tolist()}
    normalizer = CorrectionNormalizer(stats)
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest = {"train": [], "validation": [], "seed": seed, "episodes": {}}
    for i, path in enumerate(paths):
        arrays, attrs, _ = read_raw(path, active_only, mapping)
        filename = f"{i}.h5"
        split = "validation" if i in validation else "train"
        manifest[split].append(filename)
        manifest["episodes"][filename] = str(path.resolve())
        with h5py.File(output_dir / filename, "w") as file:
            for key, value in arrays.items():
                normalized = normalizer.normalize(key, value).astype(np.float32)
                if not np.isfinite(normalized).all():
                    raise ValueError(f"{path}: non-finite normalized {key}")
                file.create_dataset(key, data=normalized, compression="gzip")
            file.attrs["normalized"] = True
            for key in ("nominal_command_timing", "command_delta_acc_target"):
                if key in metadata:
                    file.attrs[key] = metadata[key]
            file.attrs["physics_dt_s"] = metadata["physics_dt_s"]
            file.attrs["joint_names"] = np.asarray(metadata["joint_names"], dtype=h5py.string_dtype())
            file.attrs["source_joint_indices"] = metadata["source_joint_indices"]
            file.attrs["recorded_ticks"] = attrs["recorded_ticks"]
            file.attrs["source"] = str(path.resolve())
            file.attrs["split"] = split
    (output_dir / "normalization.json").write_text(json.dumps(stats, indent=2) + "\n")
    (output_dir / "split.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest, stats


class CorrectionDataset(Dataset):
    """One current observation and a future correction chunk, within an episode.

    Also mask steps past the next recorded command boundary: the current input
    does not encode the teleoperator's next command. The current step is always valid.
    """

    def __init__(self, directory, split="train", chunk_size=2, *, state_keys=None, action_keys=None):
        self.directory = Path(directory)
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        stats = json.loads((self.directory / "normalization.json").read_text())
        default_states, default_action = dataset_features(stats)
        self.state_keys = list(default_states if state_keys is None else state_keys)
        default_actions = [default_action] if isinstance(default_action, str) else list(default_action)
        self.action_keys = list(default_actions if action_keys is None else action_keys)
        self.action_key = self.action_keys[0] if len(self.action_keys) == 1 else None
        manifest = json.loads((self.directory / "split.json").read_text())
        self.episodes, self.boundaries, lengths = [], [], []
        for name in manifest[split]:
            with h5py.File(self.directory / name, "r") as file:
                if not file.attrs.get("normalized", False):
                    raise ValueError(f"{name}: run preprocess_correction.py first")
                arrays = {key: np.asarray(file[key], dtype=np.float32)
                          for key in (*self.state_keys, *self.action_keys)}
                shape = arrays[self.action_keys[0]].shape
                if len(shape) != 2 or shape[0] < 1 or any(x.shape != shape for x in arrays.values()):
                    raise ValueError(f"{name}: inconsistent correction array shapes")
                ticks = np.asarray(file.attrs["recorded_ticks"], dtype=np.int64)
                if ticks[0] != 0 or ticks[-1] != shape[0] or np.any(np.diff(ticks) < 1):
                    raise ValueError(f"{name}: invalid recorded interval boundaries")
                self.episodes.append(arrays)
                self.boundaries.append(ticks)
                lengths.append(shape[0])
        self.ends = np.cumsum(lengths)

    def __len__(self):
        return int(self.ends[-1]) if len(self.ends) else 0

    def __getitem__(self, index):
        if index < 0 or index >= len(self):
            raise IndexError(index)
        episode = int(np.searchsorted(self.ends, index, side="right"))
        start = int(index - (self.ends[episode - 1] if episode else 0))
        data, ticks = self.episodes[episode], self.boundaries[episode]
        end = min(start + self.chunk_size, int(ticks[np.searchsorted(ticks, start, side="right")]))
        valid = end - start
        result = {key: torch.from_numpy(data[key][start].copy()) for key in self.state_keys}
        for key in self.action_keys:
            action = np.repeat(data[key][end-1:end], self.chunk_size, axis=0)
            action[:valid] = data[key][start:end]
            result[key] = torch.from_numpy(action)
        result["is_pad"] = torch.arange(self.chunk_size) >= valid
        return result
