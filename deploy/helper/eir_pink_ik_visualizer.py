#!/usr/bin/env python3
"""Visualize interactive dual-arm EIR inverse kinematics with Pink and Viser.

This script is visualization-only: it does not connect to or command the robot.
Drag either TCP transform control in the browser to update the IK target.
"""

from __future__ import annotations

import argparse
import threading
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

import numpy as np
import yaml


ROBOT_INTERFACE_ROOT = Path(__file__).resolve().parents[3] / "robot_interface"
DEFAULT_CONFIG_PATH = ROBOT_INTERFACE_ROOT / "robot_interface/config/eir.yaml"
FROM_REAL = True
REAL_ROBOT_IP = "192.168.0.180"
WAIST_NECK_JOINTS = ("Joint_L0", "Joint_L1", "Joint_L2_U", "Joint_L3_U")
ARM_SIDES = ("left", "right")
REAL_Q_JOINT_NAMES = (
    "Joint_L0",
    "Joint_L1",
    "Joint_L2_U",
    "Joint_L3_U",
    "Joint_L2_L",
    "Joint_L3_L",
    "Joint_L4_L",
    "Joint_L5_L",
    "Joint_L6_L",
    "Joint_L7_L",
    "Joint_L8_L",
    "Joint_L2_R",
    "Joint_L3_R",
    "Joint_L4_R",
    "Joint_L5_R",
    "Joint_L6_R",
    "Joint_L7_R",
    "Joint_L8_R",
)
REAL_ROBOT_JOINT_DOF = 22


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="EIR YAML config containing the URDF, joint pose, TCP, and Pink settings.",
    )
    parser.add_argument("--host", default="0.0.0.0", help="Viser server host.")
    parser.add_argument("--port", type=int, default=8015, help="Viser server port.")
    parser.add_argument(
        "--unlocked",
        action="store_true",
        help="Start with waist and neck joints unlocked.",
    )
    return parser.parse_args()


def _load_dependencies() -> tuple[Any, Any, Any, Any, Any, Any, Any]:
    try:
        import pinocchio as pin
        import pink
        import viser
        from pink.exceptions import NoSolutionFound
        from scipy.spatial.transform import Rotation
        from viser.extras import ViserUrdf
        from yourdfpy import URDF
    except ModuleNotFoundError as error:
        raise SystemExit(
            f"Missing dependency {error.name!r}. Install the viewer dependencies with:\n"
            "  python -m pip install 'numpy==1.26.4' 'pin==2.7.0' "
            "'pin-pink==4.3.0' 'daqp==0.9.0' 'viser==1.0.30' "
            "'yourdfpy==0.0.60'"
        ) from error
    return pin, pink, viser, NoSolutionFound, Rotation, ViserUrdf, URDF


def _load_eir_settings(config_path: Path) -> dict[str, Any]:
    if not config_path.is_file():
        raise FileNotFoundError(f"EIR config does not exist: {config_path}")

    values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    robot_values = values["robot"]
    package_root = config_path.parent.parent
    kinematics_urdf_path = package_root / robot_values["kinematics_urdf_path"]
    visual_urdf_path = package_root / robot_values["urdf_paths"]["gripper"]
    for urdf_path in (kinematics_urdf_path, visual_urdf_path):
        if not urdf_path.is_file():
            raise FileNotFoundError(f"EIR URDF does not exist: {urdf_path}")

    joint_names = tuple(values["joints"])
    if not set(WAIST_NECK_JOINTS).issubset(joint_names):
        raise ValueError("EIR config is missing one or more waist/neck joints")

    default_pose = values["default_pose"]
    if tuple(default_pose) != joint_names:
        raise ValueError("EIR default_pose order must match joints")

    return {
        "kinematics_urdf_path": kinematics_urdf_path,
        "visual_urdf_path": visual_urdf_path,
        "joint_names": joint_names,
        "default_pose": {name: float(default_pose[name]) for name in joint_names},
        "tcp": {
            side: {
                "parent_frame": str(values["tcp"][side]["parent_frame"]),
                "xyz": tuple(
                    float(value)
                    for value in values["tcp"][side]["step_tcp"]["xyz"]
                ),
                "rpy": tuple(
                    float(value)
                    for value in values["tcp"][side]["step_tcp"]["rpy"]
                ),
            }
            for side in ARM_SIDES
        },
        "ik": values["ik"],
    }


def _has_unfetched_lfs_meshes(urdf_path: Path) -> bool:
    for mesh in ET.parse(urdf_path).getroot().findall(".//mesh"):
        filename = mesh.attrib.get("filename")
        if filename is None:
            continue
        mesh_path = Path(filename.removeprefix("package://"))
        if not mesh_path.is_absolute():
            mesh_path = urdf_path.parent / mesh_path
        if not mesh_path.is_file():
            continue
        with mesh_path.open("rb") as mesh_file:
            if mesh_file.read(42).startswith(b"version https://git-lfs.github.com/spec"):
                return True
    return False


def _real_q_to_joint_values(
    real_q_degrees: Any, joint_names: tuple[str, ...]
) -> dict[str, float]:
    real_q = np.asarray(real_q_degrees, dtype=np.float64)
    expected_shape = (REAL_ROBOT_JOINT_DOF,)
    if real_q.shape != expected_shape:
        raise ValueError(
            f"IndyDCP robot q must have shape {expected_shape}, got {real_q.shape}"
        )
    if not np.all(np.isfinite(real_q)):
        raise ValueError("IndyDCP robot q contains non-finite values")
    if set(REAL_Q_JOINT_NAMES) != set(joint_names):
        raise ValueError("EIR YAML joints do not match the 18 real robot joints")

    real_q_radians = np.deg2rad(real_q[: len(REAL_Q_JOINT_NAMES)])
    return dict(zip(REAL_Q_JOINT_NAMES, real_q_radians.tolist()))


def _load_initial_joint_values(
    settings: dict[str, Any],
) -> tuple[dict[str, float], dict[str, np.ndarray] | None]:
    if not FROM_REAL:
        print("[init] Using default_pose from the EIR YAML config.")
        return dict(settings["default_pose"]), None

    from neuromeka import IndyDCP3

    print(f"[init] Reading the current joint position from {REAL_ROBOT_IP}.")
    robot = IndyDCP3(robot_ip=REAL_ROBOT_IP)
    state = robot.get_robot_data()
    if "q" not in state:
        raise RuntimeError("IndyDCP get_robot_data() returned no 'q' field")
    joint_values = _real_q_to_joint_values(state["q"], settings["joint_names"])
    task_state = np.asarray(state.get("p"), dtype=np.float64)
    if task_state.shape != (18,):
        raise ValueError(
            "IndyDCP robot p must contain 18 values "
            f"(head + left + right), got shape {task_state.shape}"
        )
    if not np.all(np.isfinite(task_state)):
        raise ValueError("IndyDCP robot p contains non-finite values")
    real_task_poses = {
        "left": task_state[6:12].copy(),
        "right": task_state[12:18].copy(),
    }
    print(
        "[init] Loaded 18 physical joints from IndyDCP, converted degrees to "
        "radians, and ignored the four dummy joints."
    )
    return joint_values, real_task_poses


class EIRPinkIKVisualizer:
    """Own the full EIR state, reduced Pink model, and Viser controls."""

    def __init__(
        self,
        settings: dict[str, Any],
        *,
        host: str,
        port: int,
        start_locked: bool,
        initial_joint_values: dict[str, float],
        real_task_poses: dict[str, np.ndarray] | None,
        dependencies: tuple[Any, Any, Any, Any, Any, Any, Any],
    ) -> None:
        (
            self._pin,
            self._pink,
            viser,
            self._no_solution_found,
            self._rotation,
            ViserUrdf,
            URDF,
        ) = dependencies
        self._joint_names = settings["joint_names"]
        self._tcp_settings = settings["tcp"]
        self._ik = settings["ik"]
        self._state_lock = threading.Lock()
        self._revision = 0
        self._last_solved_revision = -1

        self._full_model = self._pin.buildModelFromUrdf(
            str(settings["kinematics_urdf_path"])
        )
        self._validate_model()
        self._tcp_frames = self._add_tcp_frames()
        self._full_q = self._configuration_from_joint_values(
            self._full_model, initial_joint_values
        )
        self._default_full_q = self._full_q.copy()
        self._targets = self._tcp_poses(self._full_q)
        if real_task_poses is not None:
            self._report_real_pose_comparison(real_task_poses)

        urdf_path = settings["visual_urdf_path"]
        use_collision_geometry = _has_unfetched_lfs_meshes(urdf_path)
        if use_collision_geometry:
            print(
                "[viser] EIR visual meshes are Git LFS pointers; displaying "
                "the URDF collision geometry instead."
            )
        urdf = URDF.load(
            str(urdf_path),
            filename_handler=lambda fname: str(
                urdf_path.parent / Path(fname.removeprefix("package://"))
            ),
            load_meshes=not use_collision_geometry,
            build_scene_graph=not use_collision_geometry,
            load_collision_meshes=False,
            build_collision_scene_graph=use_collision_geometry,
        )
        self.server = viser.ViserServer(host=host, port=port)
        print(f"[viser] Open http://localhost:{port}")
        self.server.scene.add_frame("/world", axes_length=0.2, axes_radius=0.005)
        self.server.scene.add_grid("/grid", width=3.0, height=3.0)
        self.robot_urdf = ViserUrdf(
            self.server,
            urdf,
            root_node_name="/robot",
            load_meshes=not use_collision_geometry,
            load_collision_meshes=use_collision_geometry,
            collision_mesh_color_override=(0.65, 0.7, 0.8),
        )

        self._target_controls: dict[str, Any] = {}
        for side in ARM_SIDES:
            pose = self._targets[side]
            quaternion_xyzw = self._rotation.from_matrix(pose.rotation).as_quat()
            control = self.server.scene.add_transform_controls(
                f"/targets/{side}_tcp",
                scale=0.12,
                depth_test=False,
                position=pose.translation,
                wxyz=quaternion_xyzw[[3, 0, 1, 2]],
            )
            self._target_controls[side] = control

            @control.on_update
            def _(event: Any, side: str = side) -> None:
                wxyz = np.asarray(event.target.wxyz, dtype=np.float64)
                rotation = self._rotation.from_quat(wxyz[[1, 2, 3, 0]]).as_matrix()
                target = self._pin.SE3(
                    rotation,
                    np.asarray(event.target.position, dtype=np.float64),
                )
                with self._state_lock:
                    self._targets[side] = target
                    self._revision += 1
                self._set_solving_status()

        with self.server.gui.add_folder("EIR Pink IK"):
            self._lock_checkbox = self.server.gui.add_checkbox(
                "Lock waist + neck",
                initial_value=start_locked,
                hint=(
                    "Fix Joint_L0, Joint_L1, Joint_L2_U, and Joint_L3_U at "
                    "their current values by removing them from the IK model."
                ),
            )
            self._reset_button = self.server.gui.add_button("Reset pose and targets")
            self._status = self.server.gui.add_markdown("")

            @self._lock_checkbox.on_update
            def _(_event: Any) -> None:
                with self._state_lock:
                    self._revision += 1
                self._set_solving_status()

            @self._reset_button.on_click
            def _(_event: Any) -> None:
                self._reset()

        self._update_robot_visualization(self._full_q)
        self._set_solving_status()

    def _validate_model(self) -> None:
        for joint_name in self._joint_names:
            if not self._full_model.existJointName(joint_name):
                raise ValueError(f"EIR URDF has no joint named {joint_name!r}")
            joint_id = self._full_model.getJointId(joint_name)
            joint = self._full_model.joints[joint_id]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(f"EIR joint {joint_name!r} must have one DoF")
        for tcp_settings in self._tcp_settings.values():
            parent_frame = tcp_settings["parent_frame"]
            if self._full_model.getFrameId(parent_frame) >= self._full_model.nframes:
                raise ValueError(f"EIR URDF has no frame named {parent_frame!r}")

    def _add_tcp_frames(self) -> dict[str, str]:
        frame_names = {}
        for side, tcp_settings in self._tcp_settings.items():
            parent_frame_id = self._full_model.getFrameId(
                tcp_settings["parent_frame"]
            )
            parent_frame = self._full_model.frames[parent_frame_id]
            offset = self._pin.SE3(
                self._rotation.from_euler(
                    "xyz", tcp_settings["rpy"], degrees=False
                ).as_matrix(),
                np.asarray(tcp_settings["xyz"], dtype=np.float64),
            )
            frame_name = f"eir_{side}_tcp"
            parent_joint_id = (
                parent_frame.parentJoint
                if hasattr(parent_frame, "parentJoint")
                else parent_frame.parent
            )
            frame = self._pin.Frame(
                frame_name,
                parent_joint_id,
                parent_frame_id,
                parent_frame.placement * offset,
                self._pin.FrameType.OP_FRAME,
            )
            self._full_model.addFrame(frame)
            frame_names[side] = frame_name
        return frame_names

    def _configuration_from_joint_values(
        self, model: Any, joint_values: dict[str, float]
    ) -> np.ndarray:
        q = self._pin.neutral(model)
        for joint_name, value in joint_values.items():
            if not model.existJointName(joint_name):
                continue
            joint_id = model.getJointId(joint_name)
            q[model.joints[joint_id].idx_q] = value
        return q

    def _joint_values_from_configuration(
        self, model: Any, q: np.ndarray
    ) -> dict[str, float]:
        values: dict[str, float] = {}
        for joint_name in self._joint_names:
            if not model.existJointName(joint_name):
                continue
            joint_id = model.getJointId(joint_name)
            values[joint_name] = float(q[model.joints[joint_id].idx_q])
        return values

    def _tcp_poses(self, full_q: np.ndarray) -> dict[str, Any]:
        data = self._full_model.createData()
        self._pin.forwardKinematics(self._full_model, data, full_q)
        self._pin.updateFramePlacements(self._full_model, data)
        return {
            side: data.oMf[self._full_model.getFrameId(frame_name)].copy()
            for side, frame_name in self._tcp_frames.items()
        }

    def _report_real_pose_comparison(
        self, real_task_poses: dict[str, np.ndarray]
    ) -> None:
        print("[init] IndyDCP versus URDF TCP pose comparison:")
        for side, urdf_transform in self._targets.items():
            real_pose = real_task_poses[side]
            urdf_rotation = self._rotation.from_matrix(urdf_transform.rotation)
            urdf_pose = np.concatenate(
                (
                    urdf_transform.translation * 1000.0,
                    urdf_rotation.as_euler("xyz", degrees=True),
                )
            )
            real_rotation = self._rotation.from_euler(
                "xyz", real_pose[3:], degrees=True
            )
            position_delta_mm = urdf_pose[:3] - real_pose[:3]
            local_position_delta_mm = urdf_rotation.inv().apply(position_delta_mm)
            position_error_mm = float(np.linalg.norm(position_delta_mm))
            orientation_error_deg = float(
                np.rad2deg((real_rotation.inv() * urdf_rotation).magnitude())
            )
            matches = position_error_mm < 1.0 and orientation_error_deg < 0.5
            print(
                f"  {side}: {'MATCH' if matches else 'MISMATCH'}\n"
                f"    IndyDCP [mm, deg]: "
                f"{np.array2string(real_pose, precision=5)}\n"
                f"    URDF    [mm, deg]: "
                f"{np.array2string(urdf_pose, precision=5)}\n"
                f"    errors: position={position_error_mm:.6f} mm, "
                f"orientation={orientation_error_deg:.6f} deg\n"
                f"    URDF - IndyDCP xyz [world mm]: "
                f"{np.array2string(position_delta_mm, precision=5)}\n"
                f"    URDF - IndyDCP xyz [TCP mm]: "
                f"{np.array2string(local_position_delta_mm, precision=5)}"
            )

    def _build_ik_model(
        self, full_q: np.ndarray, lock_waist_neck: bool
    ) -> tuple[Any, np.ndarray]:
        if not lock_waist_neck:
            return self._full_model, full_q.copy()

        joint_ids = [
            self._full_model.getJointId(joint_name)
            for joint_name in WAIST_NECK_JOINTS
        ]
        model = self._pin.buildReducedModel(self._full_model, joint_ids, full_q)
        joint_values = self._joint_values_from_configuration(self._full_model, full_q)
        return model, self._configuration_from_joint_values(model, joint_values)

    def _restore_full_configuration(
        self, model: Any, q: np.ndarray, previous_full_q: np.ndarray
    ) -> np.ndarray:
        full_q = previous_full_q.copy()
        reduced_values = self._joint_values_from_configuration(model, q)
        for joint_name, value in reduced_values.items():
            full_joint = self._full_model.joints[
                self._full_model.getJointId(joint_name)
            ]
            full_q[full_joint.idx_q] = value
        return full_q

    def _solve(
        self,
        full_q: np.ndarray,
        targets: dict[str, Any],
        lock_waist_neck: bool,
    ) -> tuple[np.ndarray, bool, int, dict[str, tuple[float, float]]]:
        from pink.limits import ConfigurationLimit
        from pink.tasks import FrameTask, PostureTask

        model, initial_q = self._build_ik_model(full_q, lock_waist_neck)
        configuration = self._pink.Configuration(
            model, model.createData(), initial_q, copy_data=False
        )

        frame_tasks = []
        for side in ARM_SIDES:
            task = FrameTask(
                self._tcp_frames[side],
                position_cost=float(self._ik["position_cost"]),
                orientation_cost=float(self._ik["orientation_cost"]),
                lm_damping=float(self._ik["lm_damping"]),
            )
            task.set_target(targets[side])
            frame_tasks.append(task)

        tasks: list[Any] = list(frame_tasks)
        posture_cost = float(self._ik["posture_cost"])
        if posture_cost > 0.0:
            posture_task = PostureTask(cost=posture_cost)
            posture_task.set_target(initial_q)
            tasks.append(posture_task)

        position_tolerance = float(self._ik["position_tolerance_m"])
        orientation_tolerance = float(self._ik["orientation_tolerance_rad"])
        max_iterations = int(self._ik["max_iterations"])
        integration_dt = float(self._ik["integration_dt"])
        errors: dict[str, tuple[float, float]] = {}
        success = False
        iterations = 0

        for iterations in range(max_iterations + 1):
            errors = {}
            for side, task in zip(ARM_SIDES, frame_tasks):
                error = task.compute_error(configuration)
                errors[side] = (
                    float(np.linalg.norm(error[:3])),
                    float(np.linalg.norm(error[3:])),
                )
            success = all(
                position_error < position_tolerance
                and orientation_error < orientation_tolerance
                for position_error, orientation_error in errors.values()
            )
            if success or iterations == max_iterations:
                break

            try:
                velocity = self._pink.solve_ik(
                    configuration,
                    tasks=tasks,
                    dt=integration_dt,
                    solver=str(self._ik["solver"]),
                    damping=float(self._ik["damping"]),
                    limits=(ConfigurationLimit(model),),
                )
            except self._no_solution_found:
                break
            if not np.all(np.isfinite(velocity)):
                break
            configuration.integrate_inplace(velocity, integration_dt)

        result_q = self._restore_full_configuration(
            model, configuration.q, previous_full_q=full_q
        )
        return result_q, success, iterations, errors

    def _update_robot_visualization(self, full_q: np.ndarray) -> None:
        joint_values = self._joint_values_from_configuration(self._full_model, full_q)
        self.robot_urdf.update_cfg(joint_values)

    def _set_solving_status(self) -> None:
        lock_state = "locked" if self._lock_checkbox.value else "unlocked"
        self._status.content = (
            f"**Status:** Solving...  \nWaist + neck: **{lock_state}**"
        )

    def _set_result_status(
        self,
        success: bool,
        iterations: int,
        errors: dict[str, tuple[float, float]],
        full_q: np.ndarray,
        lock_waist_neck: bool,
    ) -> None:
        result = "Success" if success else "Not converged"
        lock_state = (
            "locked (14 IK DoF)" if lock_waist_neck else "unlocked (18 IK DoF)"
        )
        error_lines = "  \n".join(
            f"{side}: position `{position_error:.6f} m`, orientation "
            f"`{orientation_error:.6f} rad`"
            for side, (position_error, orientation_error) in errors.items()
        )
        joint_values = self._joint_values_from_configuration(self._full_model, full_q)
        waist_neck = [joint_values[name] for name in WAIST_NECK_JOINTS]
        self._status.content = (
            f"**Status:** {result} after {iterations} iterations  \n"
            f"Waist + neck: **{lock_state}**  \n"
            f"{error_lines}  \n"
            f"Waist/neck q [rad]: "
            f"`{np.array2string(np.asarray(waist_neck), precision=4)}`"
        )

    def _reset(self) -> None:
        full_q = self._default_full_q.copy()
        targets = self._tcp_poses(full_q)
        with self._state_lock:
            self._full_q = full_q
            self._targets = targets
            self._revision += 1

        self._update_robot_visualization(full_q)
        for side, pose in targets.items():
            quaternion_xyzw = self._rotation.from_matrix(pose.rotation).as_quat()
            self._target_controls[side].position = pose.translation
            self._target_controls[side].wxyz = quaternion_xyzw[[3, 0, 1, 2]]
        self._set_solving_status()

    def step(self) -> None:
        with self._state_lock:
            revision = self._revision
            if revision == self._last_solved_revision:
                return
            full_q = self._full_q.copy()
            targets = {side: target.copy() for side, target in self._targets.items()}
            lock_waist_neck = bool(self._lock_checkbox.value)

        result_q, success, iterations, errors = self._solve(
            full_q, targets, lock_waist_neck
        )
        with self._state_lock:
            self._full_q = result_q
            self._last_solved_revision = revision

        self._update_robot_visualization(result_q)
        self._set_result_status(
            success, iterations, errors, result_q, lock_waist_neck
        )

    def close(self) -> None:
        if hasattr(self.server, "stop"):
            self.server.stop()


def main() -> None:
    args = _parse_args()
    dependencies = _load_dependencies()
    settings = _load_eir_settings(args.config.resolve())
    initial_joint_values, real_task_poses = _load_initial_joint_values(settings)
    visualizer = EIRPinkIKVisualizer(
        settings,
        host=args.host,
        port=args.port,
        start_locked=not args.unlocked,
        initial_joint_values=initial_joint_values,
        real_task_poses=real_task_poses,
        dependencies=dependencies,
    )
    try:
        while True:
            visualizer.step()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\n[eir-pink] Interrupted by user.")
    finally:
        visualizer.close()


if __name__ == "__main__":
    main()
