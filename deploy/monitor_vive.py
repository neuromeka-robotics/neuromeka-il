#!/usr/bin/env python3
"""Print VIVE poses/buttons through teleoperation's OpenVR reader; no robot needed.

From deploy: python monitor_vive.py --hz 10
Positions are in the OpenVR standing tracking frame, not the robot task frame.
"""
from __future__ import annotations

import argparse
import time

import numpy as np
from scipy.spatial.transform import Rotation


def pose_values(pose):
    """Use the same matrix entries and metre-to-mm conversion as Vive.get_input."""
    matrix = np.array([[pose.m[row][col] for col in range(4)] for row in range(3)])
    return matrix[:, 3] * 1000., Rotation.from_matrix(matrix[:, :3])


def pose_change(current, reference):
    position, rotation = current
    previous_position, previous_rotation = reference
    return (np.linalg.norm(position - previous_position),
            np.rad2deg((previous_rotation.inv() * rotation).magnitude()))


def vector_text(values):
    return "[" + ", ".join(f"{value:9.3f}" for value in values) + "]"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--controller", action="append", metavar="NAME",
                        help="e.g. controller_1; repeat to select several (default: all controllers)")
    parser.add_argument("--hz", type=float, default=20., help="poll/print frequency (default: 20 Hz)")
    parser.add_argument("--duration", type=float, help="stop after this many seconds (default: Ctrl+C)")
    parser.add_argument("--ik-type", choices=("pink", "rl_constraint"), default="rl_constraint",
                        help="button interpretation used by teleoperation (default: rl_constraint)")
    args = parser.parse_args()
    if not np.isfinite(args.hz) or args.hz <= 0:
        parser.error("--hz must be positive and finite")
    if args.duration is not None and (not np.isfinite(args.duration) or args.duration <= 0):
        parser.error("--duration must be positive and finite")

    from data_collector.device.vive import TrackpadButtons
    from third_party.vive import triad_openvr

    pool = triad_openvr.triad_openvr()
    names = args.controller or list(pool.object_names["Controller"])
    if not names:
        raise SystemExit("No VIVE controllers found. Start SteamVR, wake the controllers, and retry.")
    for name in names:
        if name not in pool.object_names["Controller"]:
            raise SystemExit(f"Unknown controller {name!r}; available: {pool.object_names['Controller']}")
        device = pool.devices[name]
        print(f"{name}: serial={device.get_serial()}, OpenVR index={device.index}")

    buttons = {name: TrackpadButtons(split=args.ik_type == "rl_constraint") for name in names}
    regions = {name: TrackpadButtons(split=True) for name in names}
    previous_pose, first_pose, previous_buttons = {}, {}, {}
    print("Pose: OpenVR standing frame, XYZ in mm, XYZ Euler RPY in degrees.")
    print("step: change since previous sample; drift: change since first valid sample.")
    print("Hold controllers still to inspect stability. Invalid tracking restarts the pose reference.")
    print("Button values: 0=released, 1=pressed. record/switch use the teleop trackpad decoder.")
    print("Upper click=record, lower click=switch (rl_constraint); whole pad=record (pink). Ctrl+C to stop.")
    start = time.monotonic()
    try:
        while args.duration is None or time.monotonic() - start < args.duration:
            tick = time.monotonic()
            poses = pool.get_pose()
            for name in names:
                device = pool.devices[name]
                tracking = poses[device.index]
                pose = device.get_pose_matrix(pose=poses)
                prefix = f"{tick - start:8.3f}s {name}"
                if tracking.bDeviceIsConnected and pose is not None:
                    current = pose_values(pose)
                    first_pose.setdefault(name, current)
                    step = pose_change(current, previous_pose.get(name, current))
                    drift = pose_change(current, first_pose[name])
                    previous_pose[name] = current
                    print(f"{prefix} pose_valid=1 xyz_mm={vector_text(current[0])} "
                          f"rpy_deg={vector_text(current[1].as_euler('xyz', degrees=True))} "
                          f"step={step[0]:.3f}mm/{step[1]:.3f}deg "
                          f"drift={drift[0]:.3f}mm/{drift[1]:.3f}deg")
                else:
                    previous_pose.pop(name, None)
                    first_pose.pop(name, None)
                    print(f"{prefix} pose_valid=0 connected={int(tracking.bDeviceIsConnected)} "
                          f"tracking_result={tracking.eTrackingResult}")

                # Same state decoder as get_controller_inputs(), with its success
                # flag retained so disconnected/stale input is not shown as valid.
                valid, state = device.vr.getControllerState(device.index)
                if not valid or not tracking.bDeviceIsConnected:
                    buttons[name].read(False, 0.)
                    regions[name].read(False, 0.)
                    previous_buttons.pop(name, None)
                    print(f"{prefix} buttons_valid=0", flush=True)
                    continue
                inputs = device.controller_state_to_dict(state)
                new_press = inputs["trackpad_pressed"] and not regions[name].pressed
                upper, lower = regions[name].read(inputs["trackpad_pressed"], inputs["trackpad_y"])
                region = ("UPPER" if upper else "LOWER" if lower else "CENTER") \
                    if inputs["trackpad_pressed"] else "RELEASED"
                record, switch = buttons[name].read(inputs["trackpad_pressed"], inputs["trackpad_y"])
                if new_press:
                    print(f"{prefix} TRACKPAD {region} PRESSED "
                          f"y={inputs['trackpad_y']:+.3f} "
                          f"record={int(record)} switch={int(switch)}", flush=True)
                discrete = {
                    "pad_pressed": inputs["trackpad_pressed"],
                    "pad_touched": inputs["trackpad_touched"],
                    "menu": inputs["menu_button"], "grip": inputs["grip_button"],
                    "record": record, "switch": switch,
                }
                before = previous_buttons.get(name, discrete)
                edges = [f"{key}={'pressed' if value else 'released'}"
                         for key, value in discrete.items() if value != before[key]]
                previous_buttons[name] = discrete
                print(f"{prefix} buttons_valid=1 packet={inputs['unPacketNum']} pad_region={region} "
                      + " ".join(f"{key}={int(value)}" for key, value in discrete.items())
                      + f" pad_xy=({inputs['trackpad_x']:+.3f},{inputs['trackpad_y']:+.3f}) "
                      f"trigger={inputs['trigger']:.3f} pressed_mask=0x{inputs['ulButtonPressed']:x}"
                      + (" CHANGED: " + ", ".join(edges) if edges else ""), flush=True)
            time.sleep(max(0., 1. / args.hz - (time.monotonic() - tick)))
    except KeyboardInterrupt:
        print("\nVIVE monitor stopped.")
    finally:
        # triad_openvr owns OpenVR initialization/shutdown. Never opens a robot connection.
        del pool


if __name__ == "__main__":
    main()
