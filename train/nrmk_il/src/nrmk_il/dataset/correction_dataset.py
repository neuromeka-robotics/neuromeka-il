"""Normalized nominal-command q/v residuals and three state tokens."""
from pathlib import Path
import json

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


STATE_KEYS = ["observation.nominal_q", "observation.nominal_v", "observation.nominal_teleop_error"]
ACTION_KEYS = ["action.command_delta_q", "action.command_delta_v"]
RAW_TO_KEY = {key.split(".", 1)[1]: key for key in (*STATE_KEYS, *ACTION_KEYS)}
COMMAND_TIMING = "next_step_immediate"


def policy_features(settings):
    """Keep YAML token/output order while requiring the nominal-command design."""
    states = settings.get("state_token_keys")
    outputs = list(settings.get("output_shapes", {}))
    if settings.get("control_mode") != "nominal_command":
        raise ValueError("Correction training/replay supports only control_mode: nominal_command")
    if (not isinstance(states, list) or len(states) != len(STATE_KEYS)
            or set(states) != set(STATE_KEYS) or set(settings.get("input_shapes", {})) != set(STATE_KEYS)):
        raise ValueError(f"state_token_keys/input_shapes must contain {STATE_KEYS}")
    if set(outputs) != set(ACTION_KEYS):
        raise ValueError(f"output_shapes must contain {ACTION_KEYS}")
    return list(states), outputs


def select_normalization(stats, settings):
    """Select nominal-command statistics with the same timing as inference."""
    states, outputs = policy_features(settings)
    if stats.get("nominal_command_timing") != COMMAND_TIMING:
        raise ValueError("nominal_command requires next_step_immediate labels; regenerate/preprocess data")
    keys = [*states, *outputs]
    missing = set(keys) - set(stats["features"])
    if missing:
        raise ValueError(f"Processed data missing {sorted(missing)}")
    selected = {key: stats[key] for key in ("physics_dt_s", "joint_names", "source_joint_indices",
                "active_only", "nominal_command_timing", "schema", "training_sources") if key in stats}
    selected.update(state_keys=states, action_keys=outputs, features={key: stats["features"][key] for key in keys})
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
    if {"command_delta_q", "command_delta_v"} & set(raw_to_key):
        timing = str(attrs.get("nominal_command_timing", ""))
        if timing != COMMAND_TIMING:
            raise ValueError(f"{path}: requires next_step_immediate nominal-command labels")
        metadata["nominal_command_timing"] = timing
    return {raw_to_key[key]: value[:, indices] for key, value in arrays.items()}, attrs, metadata


def preprocess_corrections(input_dir, output_dir, validation_fraction=0.2, seed=0, active_only=True,
                           policy_config=None):
    """Fit statistics on whole training recordings, then normalize every split."""
    if not 0 <= validation_fraction < 1:
        raise ValueError("validation_fraction must be in [0, 1)")
    states, action = (STATE_KEYS, ACTION_KEYS) if policy_config is None else policy_features(policy_config)
    mapping = {key.split(".", 1)[1]: key for key in (*states, *action)}
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    paths = sorted(input_dir.glob("*.h5"))
    if not paths:
        raise ValueError(f"No H5 correction recordings in {input_dir}")
    if output_dir.resolve() == input_dir.resolve():
        raise ValueError("Output must differ from raw input directory")
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output directory must be empty: {output_dir}")
    if len(paths) == 1 and validation_fraction:
        raise ValueError("Need two recordings for a held-out split; set data.validation_fraction: 0 for training only")
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
    stats = {**metadata, "schema": "normalized_correction_v2", "features": {},
             "training_sources": [str(paths[i].resolve()) for i in train]}
    stats["state_keys"], stats["action_keys"] = list(states), list(action)
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
            for key in ("nominal_command_timing",):
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
    """Past observations ending at the current tick and a future correction chunk.

    Also mask steps past the next recorded command boundary: the current input
    does not encode the teleoperator's next command. The current step is always valid.
    Action row k predicts reference[k+1]. The endpoint at the next recorded
    boundary is valid; transitions starting at that boundary need its new input.
    History repeats the first frame at episode start and never crosses episodes.
    """

    def __init__(self, directory, split="train", chunk_size=2, *, state_keys=None, action_keys=None,
                 n_obs_steps=1):
        self.directory = Path(directory)
        if chunk_size < 1:
            raise ValueError("chunk_size must be positive")
        self.chunk_size = chunk_size
        if type(n_obs_steps) is not int or n_obs_steps < 1:
            raise ValueError("n_obs_steps must be a positive integer")
        self.n_obs_steps = n_obs_steps
        stats = json.loads((self.directory / "normalization.json").read_text())
        if stats.get("nominal_command_timing") != COMMAND_TIMING:
            raise ValueError("Dataset requires next_step_immediate nominal-command labels")
        self.state_keys = list(STATE_KEYS if state_keys is None else state_keys)
        self.action_keys = list(ACTION_KEYS if action_keys is None else action_keys)
        if set(self.state_keys) != set(STATE_KEYS) or set(self.action_keys) != set(ACTION_KEYS):
            raise ValueError("Dataset requires the three nominal-command states and two q/v residuals")
        manifest = json.loads((self.directory / "split.json").read_text())
        self.episodes, self.boundaries, lengths = [], [], []
        for name in manifest[split]:
            with h5py.File(self.directory / name, "r") as file:
                if not file.attrs.get("normalized", False):
                    raise ValueError(f"{name}: run preprocess_correction.py first")
                if file.attrs.get("nominal_command_timing") != COMMAND_TIMING:
                    raise ValueError(f"{name}: requires next_step_immediate nominal-command labels")
                arrays = {key: np.asarray(file[key], dtype=np.float32)
                          for key in (*self.state_keys, *self.action_keys)}
                shape = arrays[self.action_keys[0]].shape
                if len(shape) != 2 or shape[0] < 1 or any(x.shape != shape or not np.isfinite(x).all() for x in arrays.values()):
                    raise ValueError(f"{name}: inconsistent correction array shapes")
                ticks = np.asarray(file.attrs["recorded_ticks"], dtype=np.int64)
                if ticks.ndim != 1 or len(ticks) < 2 or ticks[0] != 0 or ticks[-1] != shape[0] or np.any(np.diff(ticks) < 1):
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
        # A single observation needs no history dimension.
        indices = start if self.n_obs_steps == 1 else np.maximum(
            np.arange(start - self.n_obs_steps + 1, start + 1), 0)
        result = {key: torch.from_numpy(data[key][indices].copy()) for key in self.state_keys}
        for key in self.action_keys:
            action = np.repeat(data[key][end-1:end], self.chunk_size, axis=0)
            action[:valid] = data[key][start:end]
            result[key] = torch.from_numpy(action)
        result["is_pad"] = torch.arange(self.chunk_size) >= valid
        return result
