#!/usr/bin/env python3
"""Calibrate VIVE teleoperation rotation from a fixed cuboid robot motion.

Robot task pose format is [x, y, z, roll, pitch, yaw] in millimeters + degrees.
VIVE pose format is OpenVR mDeviceToAbsoluteTracking 3x4 matrix: translation in
meters plus a 3x3 rotation matrix in the VIVE tracking frame.
"""
from __future__ import annotations

import argparse
import time
from typing import TYPE_CHECKING

import numpy as np

from helper.math_utils import MathFunc

if TYPE_CHECKING:
    from communication.robot import Robot

DEFAULT_CONFIG_NAME = "default"
CUBOID_SIDE_MM = 100.0
DEFAULT_CONTROL_DT = 0.05
CALIBRATION_SPEED_MM_S = 50.0


def generate_cuboid_waypoints(initial_pose, side_mm=300.):
    """
    Generate a fixed cuboid edge path from a start vertex.

    Pose format is robot task pose [x, y, z, roll, pitch, yaw] in
    millimeters + degrees. Orientation is held constant.
    """
    initial_pose = np.asarray(initial_pose, dtype=np.float64)
    side = float(side_mm)
    vertex_bits = [
        (0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 0),
        (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 0, 1),
        (1, 0, 1), (1, 0, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1),
        (0, 1, 0), (0, 0, 0),
    ]
    waypoints = []
    for x_bit, y_bit, z_bit in vertex_bits:
        pose = initial_pose.copy()
        pose[:3] += side * np.asarray([x_bit, y_bit, -z_bit], dtype=np.float64)
        waypoints.append(pose)
    return np.asarray(waypoints)


def estimate_vive_to_robot_rotation(robot_positions_mm, vive_positions_mm):
    """
    Estimate R_RV such that robot_delta ~= R_RV @ vive_delta.

    Args:
        robot_positions_mm: Array-like with shape (N, 3), paired robot TCP
            positions in millimeters.
        vive_positions_mm: Array-like with shape (N, 3), paired VIVE tracker
            positions in millimeters.

    Only relative/centered translation is used, so a fixed tracker-to-TCP
    offset is allowed when robot orientation is held constant.

    This is the orthogonal Procrustes/Kabsch fit for the rotation that best
    aligns the VIVE translation cloud to the robot translation cloud. The
    cuboid path gives us many matched displacement samples in both coordinate
    frames; the SVD below finds the least-squares rotation between them.
    """
    robot_positions = np.asarray(robot_positions_mm, dtype=np.float64)
    vive_positions = np.asarray(vive_positions_mm, dtype=np.float64)

    # Remove each cloud's centroid before fitting. This intentionally ignores
    # absolute position/origin differences and estimates only the frame rotation.
    X = vive_positions - vive_positions.mean(axis=0, keepdims=True)
    Y = robot_positions - robot_positions.mean(axis=0, keepdims=True)

    # A line-only motion cannot determine all 3D axes. We require at least a
    # planar motion; the cuboid trajectory normally provides full-rank data.
    if np.linalg.matrix_rank(X) < 2 or np.linalg.matrix_rank(Y) < 2:
        raise ValueError("Calibration motion must span at least two translation axes.")

    # Cross-covariance between VIVE-centered samples (X) and robot-centered
    # samples (Y). For row-vector samples, SVD(H) = U S Vt gives the rotation
    # R = V U^T that minimizes ||X R^T - Y||, equivalent to R @ vive_delta.
    U, singular_values, Vt = np.linalg.svd(X.T @ Y)
    R = Vt.T @ U.T

    # If noise/degeneracy makes the unconstrained solution a reflection
    # (det=-1), flip the weakest singular direction to force a proper rotation.
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Report the fit error in robot-space millimeters for sanity checking.
    residual = (R @ X.T).T - Y
    diagnostics = {
        "num_samples": len(robot_positions),
        "singular_values": singular_values,
        "rms_error_mm": float(np.sqrt(np.mean(np.sum(residual ** 2, axis=1)))),
        "determinant": float(np.linalg.det(R)),
    }
    return R, diagnostics


def interpolate_task_trajectory(waypoints: np.ndarray, control_dt: float) -> np.ndarray:
    """Linearly interpolate calibration waypoints at a fixed control period."""
    waypoints = np.asarray(waypoints, dtype=np.float64)

    trajectory = [waypoints[0].copy()]
    for start, end in zip(waypoints[:-1], waypoints[1:]):
        dist = np.linalg.norm(end[:3] - start[:3])
        steps = max(1, int(np.ceil(dist / (CALIBRATION_SPEED_MM_S * control_dt))))
        for step in range(1, steps + 1):
            alpha = step / steps
            trajectory.append((1. - alpha) * start + alpha * end)
    return np.asarray(trajectory)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", help="Collection config name; uses default or the only available config.")
    parser.add_argument("--robot-id", type=int, help="Calibrate only this robot ID; defaults to the first configured robot.")
    parser.add_argument("--execute", action="store_true", help="Actually execute the cuboid after preview.")
    return parser


def preview_path(waypoints: np.ndarray, trajectory: np.ndarray, control_dt: float) -> None:
    print("Fixed cuboid side length: %.1f mm" % CUBOID_SIDE_MM)
    print("Waypoints (robot task pose: mm + deg):")
    for i, waypoint in enumerate(waypoints):
        print("  %02d: %s" % (i, np.array2string(waypoint, precision=3, separator=", ")))
    print("Streamed targets: %d" % len(trajectory))
    print("Estimated duration: %.1f s" % ((len(trajectory) - 1) * control_dt))


def open_vive_device():
    from third_party.vive import triad_openvr

    pool = triad_openvr.triad_openvr()
    pool.print_discovered_objects()
    candidates = []
    for device_class in ("Tracker", "Controller"):
        candidates.extend(
            name for name, device in pool.devices.items()
            if device.device_class == device_class
        )
    if not candidates:
        raise RuntimeError("No VIVE controller/tracker found.")
    device_name = candidates[0]
    print("Using VIVE device: %s" % device_name)
    return pool, pool.devices[device_name]


def _openvr_pose_to_pos_rot(pose_matrix):
    mat = np.asarray(pose_matrix.m, dtype=np.float64)
    return mat[:, 3] * 1000., mat[:, :3].copy()


def load_config(config_name=None):
    from data_collector.config import CONFIGS

    if config_name is None:
        config_name = DEFAULT_CONFIG_NAME if DEFAULT_CONFIG_NAME in CONFIGS else (
            next(iter(CONFIGS)) if len(CONFIGS) == 1 else None)
    if config_name not in CONFIGS:
        raise KeyError("Select --config from: %s" % sorted(CONFIGS.keys()))
    return CONFIGS[config_name]


def make_robot(config, robot_id=None) -> tuple[Robot, dict]:
    from communication.robot import create_robot
    from copy import deepcopy

    if robot_id is None:
        robot_id = config.robot_config.robot_ids[0]
    if robot_id not in config.robot_config.robot_ids:
        raise ValueError(f"Robot ID {robot_id} is not in this configuration")
    robot_params = deepcopy(config.robot_config.robot_params[robot_id])
    # Calibration needs only arm motion, not gripper activation or force setup.
    robot_params["gripper"] = {"enable": False}
    robot_params.get("init_kwargs", {}).pop("force_mode", None)
    print(f"Calibrating robot ID {robot_id} only")
    robot = create_robot(robot_params)
    return robot, robot_params


def get_current_robot_pose(robot: Robot) -> np.ndarray:
    current_pose = np.asarray(robot.get_state()["p"], dtype=np.float64)  # mm + deg
    print("Using current robot EE pose as cuboid start: %s" % np.array2string(current_pose, precision=3, separator=", "))
    return current_pose


def collect_sample(robot: Robot, vive_device, target: np.ndarray) -> dict | None:
    robot_pose = np.asarray(robot.get_state()["p"], dtype=np.float64)  # mm + deg
    vive_pose = vive_device.get_pose_matrix()
    if vive_pose is None:
        return None
    vive_pos_mm, vive_rot = _openvr_pose_to_pos_rot(vive_pose)
    return {
        "t": time.time(),
        "target_pose": target.copy(),
        "robot_pose": robot_pose,
        "vive_pos_mm": vive_pos_mm,
        "vive_rot": vive_rot,
    }


def execute_and_collect(
    robot: Robot,
    robot_params: dict,
    vive_device,
    trajectory: np.ndarray,
    control_dt: float,
) -> list[dict]:
    control_cfg = robot_params["control"]
    vel = control_cfg["vel_scale"]
    acc = control_cfg["acc_scale"]
    samples: list[dict] = []

    print("Starting task-absolute teleop...")
    from helper.extra_utils import ROBOT_STATE
    try:
        # Select task mode even if a previous session left joint teleop active.
        robot.stop_teleop()
        deadline = time.monotonic() + 10.
        while robot.get_state()["op_state"] == ROBOT_STATE.TELE_OP:
            if time.monotonic() >= deadline:
                raise TimeoutError("Previous teleoperation mode did not stop")
            time.sleep(0.05)
        deadline = time.monotonic() + 10.
        while True:
            op_state = robot.get_state()["op_state"]
            if op_state == ROBOT_STATE.TELE_OP:
                break
            if ROBOT_STATE.in_failure_state(op_state):
                raise RuntimeError(f"Cannot start calibration: op_state={op_state}")
            if time.monotonic() >= deadline:
                raise TimeoutError("Task teleoperation did not start within 10 seconds")
            robot.start_teleop(mode="task_abs")
            time.sleep(0.2)
        next_tick = time.monotonic()
        for target in trajectory:
            print(target.tolist())
            robot.tele_move(action=target.tolist(), mode="task_abs", vel_scale=vel, acc_scale=acc)
            sample = collect_sample(robot, vive_device, target)
            if sample is not None:
                samples.append(sample)

            next_tick += control_dt
            wait_time = next_tick - time.monotonic()
            if wait_time > 0:
                time.sleep(wait_time)
    finally:
        robot.stop_teleop()
    return samples


def print_result(R_RV: np.ndarray, diagnostics: dict) -> np.ndarray:
    T_RV = np.eye(4, dtype=np.float64)
    T_RV[:3, :3] = R_RV
    calib_uvw = MathFunc.rotMat_to_euler(R_RV)

    print("\nEstimated VIVE-to-robot rotation R_RV:")
    print(np.array2string(R_RV, precision=10, suppress_small=False))
    print("\n4x4 display matrix (zero translation; teleop uses pose deltas):")
    print(np.array2string(T_RV, precision=10, suppress_small=False))
    print("\ncalib_uvw radians for DATA_CONFIG.device_params['calib_uvw']:")
    print(calib_uvw.tolist())
    print("\nDiagnostics:")
    for key, value in diagnostics.items():
        print("  %s: %s" % (key, value))
    return calib_uvw


def main() -> None:
    args = build_arg_parser().parse_args()
    config = load_config(args.config)
    control_dt = config.robot_config.control_dt if args.execute else DEFAULT_CONTROL_DT
    if not args.execute:
        print("Dry run: connecting to the robot only to read the current pose; no motion or VIVE access.")
        print("Using default preview control_dt=%.3f s." % control_dt)

    robot, robot_params = make_robot(config, args.robot_id)
    initial_pose = get_current_robot_pose(robot)
    waypoints = generate_cuboid_waypoints(initial_pose, side_mm=CUBOID_SIDE_MM)
    trajectory = interpolate_task_trajectory(waypoints, control_dt)
    preview_path(waypoints, trajectory, control_dt)

    if not args.execute:
        print("Dry run only. Re-run with --execute after checking the path.")
        return
    vive_pool, vive_device = open_vive_device()  # keep pool alive while sampling
    _ = vive_pool

    samples = execute_and_collect(
        robot=robot,
        robot_params=robot_params,
        vive_device=vive_device,
        trajectory=trajectory,
        control_dt=control_dt,
    )
    print("Collected valid paired samples: %d" % len(samples))
    if len(samples) < 3:
        raise RuntimeError("Not enough valid VIVE samples to estimate calibration.")

    robot_positions = np.asarray([s["robot_pose"][:3] for s in samples], dtype=np.float64)
    vive_positions = np.asarray([s["vive_pos_mm"] for s in samples], dtype=np.float64)
    R_RV, diagnostics = estimate_vive_to_robot_rotation(robot_positions, vive_positions)
    print_result(R_RV, diagnostics)


if __name__ == "__main__":
    main()
