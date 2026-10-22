#!/usr/bin/env python3
"""Record Nami DH gripper excitation for simulation replay and identification.

Run from any directory with the repository's env_il Python environment:
    python deploy/gripper_sysid.py --robot-ip 192.168.0.120 --mode binary
    python deploy/gripper_sysid.py --robot-ip 192.168.0.120 --mode continuous

Commands and state snapshots run at 20 Hz. DH hardware feedback may update much
slower; a separate connection polls it without blocking commands. Both the raw
feedback stream and its timing are saved alongside the 20 Hz snapshots.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path
from threading import Event, Lock, Thread
import time

import h5py
import numpy as np


DT = 0.05
BINARY_START_HOLD_S = 2.
BINARY_END_HOLD_S = 0.5


def make_profile(name, duration, seed):
    """Return the exact command plan on the nominal 20 Hz time grid."""
    if name not in ("binary", "continuous"):
        raise ValueError(f"Unknown command mode: {name}")
    times = np.arange(round(duration / DT), dtype=np.float64) * DT
    periods, starts = [], []
    commands = np.empty(len(times))
    if name == "binary":
        index = 0
        while index < len(times):
            start = index * DT
            switch_rate = (1. / BINARY_START_HOLD_S +
                           (1. / BINARY_END_HOLD_S - 1. / BINARY_START_HOLD_S) * start / duration)
            hold_steps = max(1, int(round(1. / (switch_rate * DT))))
            commands[index:index + hold_steps] = len(periods) % 2
            starts.append(start)
            periods.append(hold_steps * DT)
            index += hold_steps
        return times, commands, np.asarray(starts), np.asarray(periods)
    rng = np.random.default_rng(seed)
    start = 0.
    while start < duration:
        period = float(rng.uniform(5., 10.))
        mask = (times >= start) & (times < start + period)
        # Each complete cycle starts/ends fully open, with continuous phase.
        commands[mask] = 0.5 * (1. + np.cos(2. * np.pi * (times[mask] - start) / period))
        starts.append(start)
        periods.append(period)
        start += period
    return times, commands, np.asarray(starts), np.asarray(periods)


class FeedbackReader:
    """Poll fresh hardware data; retain query bounds, sequence IDs, and raw samples."""

    def __init__(self, read_state):
        self.read_state = read_state
        self.lock = Lock()
        self.stop = Event()
        self.samples = []
        self.error = None
        self.thread = Thread(target=self._run, daemon=True)

    def _run(self):
        try:
            while not self.stop.is_set():
                start = time.monotonic()
                state = self.read_state()
                end = time.monotonic()
                position = float(state["gripper_position"]) / 1000.
                if not np.isfinite(position) or not 0. <= position <= 1.:
                    raise ValueError(f"Invalid normalized gripper feedback: {position}")
                with self.lock:
                    self.samples.append((start, end, position, len(self.samples)))
                # Bound polling to 20 Hz if a different firmware responds quickly.
                self.stop.wait(max(0., DT - (end - start)))
        except Exception as error:
            with self.lock:
                self.error = error

    def latest(self):
        with self.lock:
            if self.error is not None:
                raise RuntimeError("Gripper feedback polling failed") from self.error
            return self.samples[-1] if self.samples else None

    def since(self, start):
        with self.lock:
            first = next((i for i, sample in enumerate(self.samples) if sample[1] >= start),
                         len(self.samples))
            # Retain the initial held measurement, even if it predates recording.
            return self.samples[max(0, first - 1):]


def wait_until_open(robot, reader, args):
    """Require two fresh, near-open measurements after the opening command."""
    print(f"Opening arm {args.arm_index} fully (1.0) before recording...", flush=True)
    robot.move_gripper(mode="no_thread", value=1., arm_index=args.arm_index)
    sent = time.monotonic()
    deadline = sent + args.open_timeout
    previous_id, consecutive = -1, 0
    while time.monotonic() < deadline:
        sample = reader.latest()
        if sample is not None and sample[0] >= sent and sample[3] != previous_id:
            previous_id = sample[3]
            consecutive = consecutive + 1 if sample[2] >= 1. - args.open_tolerance else 0
            if consecutive >= 2:
                print(f"Open confirmed: {sample[2]:.4f}", flush=True)
                return
        time.sleep(DT)
    raise TimeoutError(f"Gripper did not reach open within {args.open_timeout:g} s")


def record(robot, reader, commands, arm_index, rows, origin):
    """Send every planned command; overruns extend the rollout, without bursts."""
    for index, command in enumerate(commands):
        start = time.monotonic()
        robot.move_gripper(mode="no_thread", value=float(command), arm_index=arm_index)
        sent = time.monotonic()
        feedback_start, feedback_end, position, sequence = reader.latest()
        sampled = time.monotonic()
        work = sampled - start
        rows.append((start - origin, sent - origin, sampled - origin,
                     float(command), position, feedback_start - origin,
                     feedback_end - origin, sequence, sampled - feedback_end, work))
        remaining = DT - work
        if remaining < 0.:
            print(f"OVERRUN sample {index}: command + state took {work * 1000:.2f} ms "
                  f"(budget 50 ms; exceeded by {-remaining * 1000:.2f} ms)", flush=True)
        # Include reporting overhead in the sleep budget.
        time.sleep(max(0., DT - (time.monotonic() - start)))


def save_recording(path, name, args, plan, rows, raw_feedback, origin, started_at, status):
    planned_time, planned_commands, cycle_starts, periods = plan
    data = np.asarray(rows, dtype=np.float64).reshape(-1, 10)
    raw = np.asarray(raw_feedback, dtype=np.float64).reshape(-1, 4)
    with h5py.File(path, "x") as hf:
        hf.attrs.update(profile=name, mode=name, frequency_hz=1. / DT, control_dt=DT,
                        duration_s=args.duration, robot_ip=args.robot_ip,
                        arm_index=args.arm_index, tool_index=args.tool_index,
                        robot_id=args.robot_id, gripper_arm_indices=[args.arm_index],
                        speed=args.speed, force=args.force, seed=args.seed,
                        period_min_s=BINARY_END_HOLD_S if name == "binary" else 5.,
                        period_max_s=BINARY_START_HOLD_S if name == "binary" else 10.,
                        profile_schedule="increasing_switching_frequency" if name == "binary" else "random_sine_periods",
                        period_semantics="hold_interval" if name == "binary" else "full_sine_cycle",
                        status=status,
                        open_value=1., closed_value=0.,
                        feedback_mode="background hardware polling; 20 Hz sample-and-hold",
                        time_reference="monotonic seconds relative to recording start",
                        started_at_utc=started_at,
                        timing_note="feedback query bounds are host times, not device measurement timestamps")
        if name == "binary":
            hf.attrs.update(binary_start_hold_s=BINARY_START_HOLD_S,
                            binary_end_hold_s=BINARY_END_HOLD_S,
                            binary_start_frequency_hz=1. / (2. * BINARY_START_HOLD_S),
                            binary_end_frequency_hz=1. / (2. * BINARY_END_HOLD_S))
        # Match collector.py: root datasets, robot suffix, float32 (T, G).
        for key, column in (("gripper_command", 3), ("gripper_position", 4)):
            hf.create_dataset(f"{key}_{args.robot_id}", data=data[:, column:column + 1].astype(np.float32))
        timing = {"time": 0, "command_end_time": 1, "state_record_time": 2,
                  "feedback_read_start_time": 5, "feedback_read_end_time": 6,
                  "feedback_sequence": 7, "feedback_age": 8, "work_duration": 9}
        for key, column in timing.items():
            values = data[:, column:column + 1]
            hf.create_dataset(key, data=values.astype(np.int64) if key == "feedback_sequence" else values)
        hf.create_dataset("overrun", data=(data[:, 9:10] > DT))
        plan_group = hf.create_group("command_plan")
        plan_group.create_dataset("time", data=planned_time)
        plan_group.create_dataset("value", data=planned_commands)
        plan_group.create_dataset("cycle_start_time", data=cycle_starts)
        plan_group.create_dataset("period", data=periods)
        feedback = hf.create_group("raw_feedback")
        feedback.create_dataset("read_start_time", data=raw[:, 0] - origin)
        feedback.create_dataset("read_end_time", data=raw[:, 1] - origin)
        feedback.create_dataset("position", data=raw[:, 2].astype(np.float32))
        feedback.create_dataset("sequence", data=raw[:, 3].astype(np.int64))


def plot_recording(path, robot_id):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with h5py.File(path, "r") as hf:
        command_time = hf["time"][:, 0]
        command = hf[f"gripper_command_{robot_id}"][:, 0]
        state_time = hf["state_record_time"][:, 0]
        position = hf[f"gripper_position_{robot_id}"][:, 0]
        raw_time = hf["raw_feedback/read_end_time"][:]
        raw_position = hf["raw_feedback/position"][:]
        work = hf["work_duration"][:, 0]
        age = hf["feedback_age"][:, 0]
        profile = hf.attrs["profile"]
    fig, (ax, timing) = plt.subplots(2, 1, figsize=(11, 6), sharex=True,
                                     gridspec_kw={"height_ratios": [3, 1]})
    ax.step(command_time, command, where="post", label="Command")
    ax.plot(state_time, position, label="State (20 Hz snapshots)")
    ax.scatter(raw_time, raw_position, s=12, color="black", label="Hardware replies")
    ax.set(ylabel="Position (0=closed, 1=open)", ylim=(-0.05, 1.05), title=f"Gripper: {profile}")
    ax.legend()
    ax.grid(alpha=0.3)
    timing.plot(command_time, work * 1000., label="Command + state")
    timing.plot(state_time, age * 1000., label="Feedback age")
    timing.axhline(50., color="red", linestyle="--", label="50 ms budget")
    timing.set(xlabel="Time since recording start (s)", ylabel="Time (ms)")
    timing.legend()
    timing.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path.with_suffix(".png"), dpi=160)
    plt.close(fig)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot-ip", default="192.168.0.120")
    parser.add_argument("--arm-index", type=int, choices=(1, 2), default=2)
    parser.add_argument("--tool-index", type=int, default=None,
                        help="Physical endport; defaults to arm-index minus one")
    parser.add_argument("--robot-id", type=int, default=0, help="HDF5 dataset suffix")
    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument("--mode", choices=("binary", "continuous"),
                            help="Command mode (default: binary)")
    mode_group.add_argument("--profile", choices=("both", "toggle", "sine"),
                            help="Legacy selector: toggle=binary, sine=continuous, both=record each")
    parser.add_argument("--duration", type=float, default=60., help="Nominal seconds per profile (default: 60)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for continuous sine periods")
    parser.add_argument("--speed", type=int, default=100, choices=range(1, 101), metavar="1..100")
    parser.add_argument("--force", type=int, default=100, choices=range(20, 101), metavar="20..100")
    parser.add_argument("--open-timeout", type=float, default=15.)
    parser.add_argument("--open-tolerance", type=float, default=0.02)
    parser.add_argument("--output-dir", type=Path,
                        default=Path(__file__).resolve().parents[1] / "train/data/gripper_sysid" /
                        datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ"))
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    if (not np.isfinite(args.duration) or args.duration < DT or
            not np.isclose(args.duration / DT, round(args.duration / DT), rtol=0., atol=1e-8)):
        parser.error("--duration must be a positive multiple of 0.05 s")
    if not np.isfinite(args.open_timeout) or args.open_timeout <= 0.:
        parser.error("--open-timeout must be finite and positive")
    if not np.isfinite(args.open_tolerance) or not 0. < args.open_tolerance < 1.:
        parser.error("--open-tolerance must be between 0 and 1")
    args.tool_index = args.arm_index - 1 if args.tool_index is None else args.tool_index
    if args.tool_index < 0 or args.robot_id < 0 or args.seed < 0:
        parser.error("--tool-index, --robot-id, and --seed must be nonnegative")
    if args.profile == "both":
        profiles = ("binary", "continuous")
    else:
        profiles = (args.mode or {"toggle": "binary", "sine": "continuous"}.get(args.profile, "binary"),)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    paths = [args.output_dir / f"{index}.h5" for index in range(len(profiles))]
    if any(path.exists() or path.with_suffix(".png").exists() for path in paths):
        parser.error("Output files already exist; choose a new --output-dir")

    from neuromeka import IndyDCP3
    from communication.robot import NamiRobot

    robot = NamiRobot(robot_ip=args.robot_ip, gripper_config={"per_arm": {
        args.arm_index: {"enable": True, "type": "EndportDHGripperClient",
                         "params": {"tool_index": args.tool_index,
                                    "speed": args.speed, "force": args.force}}
    }})
    feedback_client = IndyDCP3(robot_ip=args.robot_ip)
    reader = FeedbackReader(lambda: feedback_client.get_gripper_data_for(args.tool_index))
    reader.thread.start()
    print("Recording at 20 Hz; hardware replies can be slower. See raw_feedback and feedback_age.")
    try:
        for name, path in zip(profiles, paths):
            plan = make_profile(name, args.duration, args.seed)
            wait_until_open(robot, reader, args)
            rows = []
            origin = time.monotonic()
            started_at = datetime.now(timezone.utc).isoformat()
            status = "complete"
            print(f"Recording {name}: {len(plan[0])} samples -> {path}", flush=True)
            try:
                record(robot, reader, plan[1], args.arm_index, rows, origin)
            except BaseException as error:
                status = f"interrupted: {type(error).__name__}: {error}"
                raise
            finally:
                save_recording(path, name, args, plan, rows, reader.since(origin),
                               origin, started_at, status)
                plot_recording(path, args.robot_id)
                if rows:
                    values = np.asarray(rows)
                    print(f"Saved {len(rows)} samples and plot; "
                          f"{np.count_nonzero(values[:, 9] > DT)} overruns, "
                          f"max command + state {values[:, 9].max() * 1000:.2f} ms, "
                          f"{len(np.unique(values[:, 7]))} distinct hardware replies.", flush=True)
    except KeyboardInterrupt:
        print("Stopped; completed samples saved.")
    finally:
        reader.stop.set()
        # Closing the read channel cancels a pending RPC before joining the worker.
        for name, channel in vars(feedback_client).items():
            if name.endswith("_channel"):
                channel.close()
        reader.thread.join(timeout=1.)


if __name__ == "__main__":
    main()
