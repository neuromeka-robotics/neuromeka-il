#!/usr/bin/env python3
"""Plot gripper state and command from a Neuromeka HDF5 recording.

    python deploy/plot_gripper.py path/to/0.h5
    python deploy/plot_gripper.py path/to/0.h5 --show --output comparison.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np


def read_column(hf, name, column=0):
    if name not in hf:
        raise ValueError(f"Missing dataset {name!r}; available keys: {', '.join(hf.keys())}")
    values = np.asarray(hf[name], dtype=float)
    if values.ndim == 1 and column == 0:
        return values
    if values.ndim == 2 and 0 <= column < values.shape[1]:
        return values[:, column]
    raise ValueError(f"Cannot read column {column} from {name!r} with shape {values.shape}")


def load_trajectory(path, robot_id=None, gripper_index=0, fps=None):
    with h5py.File(path, "r") as hf:
        robot_id = int(hf.attrs.get("robot_id", 0)) if robot_id is None else robot_id
        command = read_column(hf, f"gripper_command_{robot_id}", gripper_index)
        state = read_column(hf, f"gripper_position_{robot_id}", gripper_index)
        if len(command) == 0 or len(state) == 0:
            raise ValueError("Recording contains no command or state samples")

        def sample_times(name, count):
            if name in hf:
                times = read_column(hf, name)
                if len(times) != count:
                    raise ValueError(f"{name!r} has {len(times)} timestamps for {count} samples")
                return times
            rate = fps
            if rate is None:
                if "frequency_hz" in hf.attrs:
                    rate = float(hf.attrs["frequency_hz"])
                elif "control_dt" in hf.attrs:
                    dt = float(hf.attrs["control_dt"])
                    if not np.isfinite(dt) or dt <= 0.:
                        raise ValueError("File attribute control_dt must be finite and positive")
                    rate = 1. / dt
                else:
                    rate = 20.
            if not np.isfinite(rate) or rate <= 0.:
                raise ValueError("Sample rate must be finite and positive")
            return np.arange(count, dtype=float) / rate

        command_time = sample_times("time", len(command))
        state_time = sample_times("state_record_time" if "state_record_time" in hf else "time", len(state))
    return command_time, command, state_time, state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("h5_file", type=Path)
    parser.add_argument("--output", "-o", type=Path, help="Default: <input_stem>_gripper.png")
    parser.add_argument("--show", action="store_true", help="Also open an interactive plot window")
    parser.add_argument("--robot-id", type=int, help="Dataset suffix; defaults to file metadata or 0")
    parser.add_argument("--gripper-index", type=int, default=0, help="Column for recordings with multiple grippers")
    parser.add_argument("--fps", type=float, help="Rate for files without timestamps; defaults to file metadata or 20 Hz")
    args = parser.parse_args()
    if args.gripper_index < 0 or (args.robot_id is not None and args.robot_id < 0):
        parser.error("--gripper-index and --robot-id must be nonnegative")
    if args.fps is not None and (not np.isfinite(args.fps) or args.fps <= 0.):
        parser.error("--fps must be finite and positive")
    try:
        command_time, command, state_time, state = load_trajectory(
            args.h5_file, args.robot_id, args.gripper_index, args.fps)
    except (OSError, ValueError) as error:
        parser.error(str(error))

    import matplotlib
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.step(command_time, command, where="post", label="Command", linewidth=1.5)
    ax.plot(state_time, state, label="Gripper state", linewidth=1.5)
    ax.set(xlabel="Time (s)", ylabel="Position (0=closed, 1=open)",
           title=args.h5_file.name)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    output = args.output or args.h5_file.with_name(f"{args.h5_file.stem}_gripper.png")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"Saved plot: {output}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
