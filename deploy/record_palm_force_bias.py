#!/usr/bin/env python3
"""Hold the current EIR joint target and record stationary no-contact torque data."""

import argparse
import csv
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from data_collector.pink_ik import DCP_ACTIVE_JOINT_NAMES


DEFAULT_DIRECTORY = Path(__file__).resolve().parent / "middle_level_controller/box_lift_open_loop/result/bias"
JOINT_NAMES = DCP_ACTIVE_JOINT_NAMES + tuple(f"Dummy_{i}" for i in range(4))
TORQUE_FIELDS = ("tau", "tau_act", "tau_ext", "tau_jts")


def _check_response(result, operation):
    response = result.get("response", result)
    if str(response.get("code")) != "0":
        raise RuntimeError(f"{operation} failed: {response}")


def _sample(state):
    if "response" in state:
        _check_response(state, "get_control_state")
    values = []
    for field in ("q", "qdot"):
        vector = np.asarray(state[field], dtype=float)
        if vector.shape != (22,) or not np.isfinite(vector).all():
            raise ValueError(f"Expected 22 finite {field} values")
        values.append(vector)
    torques = []
    for field in TORQUE_FIELDS:
        vector = list(state.get(field, []))
        if len(vector) > 22:
            raise ValueError(f"Unexpected {field} length: {len(vector)}")
        if field == "tau_jts" and (len(vector) < 18 or not np.isfinite(vector[4:18]).all()):
            raise ValueError("Recording requires finite tau_jts for all 14 arm joints")
        torques.extend([len(vector), *vector, *([None] * (22 - len(vector)))])
    return *values, torques


def _start_teleop(client, timeout=10.):
    """Repeat the controller's stop/start sequence until teleoperation is active."""
    from neuromeka import control_msgs

    deadline = time.monotonic() + timeout
    mode = None
    while time.monotonic() < deadline:
        state = client.get_robot_data()
        if "response" in state:
            _check_response(state, "get_robot_data")
        mode = state["op_state"]
        if mode == 17:
            return
        if mode not in (5, 10):
            raise RuntimeError(f"Unexpected operation state {mode} while entering teleoperation")
        _check_response(client.stop_teleop(), "stop_teleop during startup")
        _check_response(client.start_teleop(method=control_msgs.TELE_JOINT_ABSOLUTE), "start_teleop")
        time.sleep(.2)
    raise RuntimeError(f"Robot did not enter teleoperation within {timeout:g} s (op_state={mode}); "
                       "no joint commands sent")


def record_bias(client, output, *, duration=5., settle=2., rate=50.,
                max_speed=.2, max_drift=.5):
    """Stream a fixed q22 target; save only slow samples after settling.

    Caller must arrange no external contact. This function leaves compliance,
    servo, teaching, and gripper settings as configured on the robot.
    """
    for name, value in (("duration", duration), ("rate", rate),
                        ("max_speed", max_speed), ("max_drift", max_drift)):
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"{name} must be finite and positive")
    if not np.isfinite(settle) or settle < 0:
        raise ValueError("settle must be finite and non-negative")
    state = client.get_robot_data()
    if "response" in state:
        _check_response(state, "get_robot_data")
    if state["op_state"] not in (5, 10):
        raise ValueError("Robot must already be idle or in compliance, with no other motion/teleop running")
    print(f"Existing compliance setting: {client.get_compliance_mode()}")
    q0, velocity, _ = _sample(client.get_control_state())
    if np.max(np.abs(velocity[:18])) > max_speed:
        raise ValueError("Robot is moving; wait for it to settle before starting")
    command = q0.tolist()  # Never update this target from subsequent measurements.
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    header = ["sample_index", "elapsed_time_s", "wall_time_s",
              *(f"q_{name}_deg" for name in JOINT_NAMES),
              *(f"qdot_{name}_deg_s" for name in JOINT_NAMES),
              *(f"command_{name}_deg" for name in JOINT_NAMES),
              *(column for field in TORQUE_FIELDS for column in
                (f"{field}_count", *(f"{field}_{name}" for name in JOINT_NAMES)))]
    count = skipped = 0
    with output.open("x", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        try:
            print("Starting joint teleoperation; waiting for op_state=17...")
            _start_teleop(client)
            print(f"Teleoperation ready. Holding initial q for {settle:g} s settling + {duration:g} s recording.")
            start = next_tick = time.monotonic()
            while True:
                elapsed = time.monotonic() - start
                if elapsed >= settle + duration:
                    break
                q, velocity, torques = _sample(client.get_control_state())
                if np.max(np.abs(q[:18] - q0[:18])) > max_drift:
                    raise RuntimeError(f"Joint drift exceeded {max_drift:g} deg; ending hold")
                sample_time = time.monotonic() - start
                wall_time = time.time()
                _check_response(client.movetelej_abs(jpos=command.copy(), vel_ratio=.5, acc_ratio=.5),
                                "movetelej_abs")
                if settle <= sample_time < settle + duration:
                    if np.max(np.abs(velocity[:18])) <= max_speed:
                        writer.writerow([count, sample_time - settle, wall_time,
                                         *q.tolist(), *velocity.tolist(), *command, *torques])
                        stream.flush()
                        count += 1
                    else:
                        skipped += 1
                next_tick = max(next_tick + 1. / rate, time.monotonic())
                time.sleep(max(0., next_tick - time.monotonic()))
        finally:
            # Stop even if the start RPC failed after reaching the controller.
            original_error = sys.exc_info()[0] is not None
            try:
                _check_response(client.stop_teleop(), "stop_teleop")
            except Exception as error:
                print(f"Could not stop teleoperation: {error}", file=sys.stderr)
                if not original_error:
                    raise
            finally:
                if count:
                    print(f"Saved {count} stationary samples to {output}; skipped {skipped} moving samples.")
                else:
                    stream.close()
                    output.unlink()
                    print(f"No stationary samples saved; removed empty output {output} "
                          f"(skipped {skipped} moving samples).")
    if not count:
        raise ValueError("No stationary samples recorded; this CSV cannot be used for calibration")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", default="192.168.0.180", help="EIR robot IP")
    parser.add_argument("--duration", type=float, default=5., help="Recording interval after settling, seconds")
    parser.add_argument("--settle", type=float, default=2., help="Unlogged hold interval, seconds")
    parser.add_argument("--rate", type=float, default=50., help="Target sampling/command frequency, Hz")
    parser.add_argument("--max-speed", type=float, default=.2, help="Maximum accepted active-joint speed, deg/s")
    parser.add_argument("--max-drift", type=float, default=.5, help="Abort if an active joint moves this far, deg")
    parser.add_argument("--output", type=Path, help="New CSV path; existing files are never overwritten")
    args = parser.parse_args()
    output = args.output or DEFAULT_DIRECTORY / f"no_contact_{datetime.now():%Y%m%d_%H%M%S_%f}.csv"
    print("Prepare a stationary robot with no box contact or external arm support. "
          "Keep the lifting-run hand/tool configuration and compliance setting.")
    from neuromeka import IndyDCP3

    try:
        record_bias(IndyDCP3(robot_ip=args.ip), output, duration=args.duration,
                    settle=args.settle, rate=args.rate, max_speed=args.max_speed, max_drift=args.max_drift)
    except KeyboardInterrupt:
        print("Recording interrupted; any saved stationary samples remain available.")
        raise SystemExit(130)
    except Exception as error:
        print(f"Recording failed: {error}", file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
