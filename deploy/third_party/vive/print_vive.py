"""
Print VIVE pose and collection button inputs at 10 Hz, without a robot.

From deploy/: python third_party/vive/print_vive.py --device controller_1
"""

import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np

# Direct execution needs deploy/ on the import path for helper and third_party.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from helper.math_utils import MathFunc


def read_input(device):
    """Pose is [x, y, z, roll, pitch, yaw] in mm/degrees in the VIVE frame."""
    matrix = device.get_pose_matrix()
    buttons = device.get_controller_inputs()
    pose = None
    if matrix is not None:
        matrix = np.asarray(matrix.m, dtype=float)
        position = matrix[:, 3] * 1000.
        orientation = MathFunc.rad_to_degree(MathFunc.rotMat_to_euler(matrix[:, :3]))
        pose = np.round(np.concatenate((position, orientation)), 4).tolist()
    return {
        "pose": pose,
        "button": bool(buttons["trackpad_pressed"]),
        "trigger": float(buttons["trigger"]),
        "valid": matrix is not None,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="controller_1", help="VIVE controller name")
    args = parser.parse_args()

    from third_party.vive import triad_openvr

    pool = triad_openvr.triad_openvr()
    try:
        pool.print_discovered_objects()
        if args.device not in pool.devices:
            raise ValueError(f"Device {args.device!r} not found; available: {list(pool.devices)}")
        device = pool.devices[args.device]
        if device.device_class != "Controller":
            raise ValueError("Select a Controller to read both pose and buttons")
        print(f"Reading {args.device} at 10 Hz. Ctrl+C to stop.", flush=True)
        print("pose: [x, y, z, roll, pitch, yaw] (mm, degrees; VIVE frame). "
              "button: trackpad press; trigger: 0–1. Invalid tracking prints pose=null.", flush=True)
        period = 0.1
        next_tick = time.monotonic()
        while True:
            print(json.dumps(read_input(device)), flush=True)
            next_tick += period
            delay = next_tick - time.monotonic()
            if delay > 0:
                time.sleep(delay)
            else:
                next_tick = time.monotonic()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        triad_openvr.openvr.shutdown()


if __name__ == "__main__":
    main()
