"""Pink inverse kinematics for STEP-compatible humanoid teleoperation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import yaml


DCP_ACTIVE_JOINT_NAMES = (
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

CHAIN_JOINT_NAMES = {
    0: ("Joint_L2_U", "Joint_L3_U"),
    1: (
        "Joint_L2_L",
        "Joint_L3_L",
        "Joint_L4_L",
        "Joint_L5_L",
        "Joint_L6_L",
        "Joint_L7_L",
        "Joint_L8_L",
    ),
    2: (
        "Joint_L2_R",
        "Joint_L3_R",
        "Joint_L4_R",
        "Joint_L5_R",
        "Joint_L6_R",
        "Joint_L7_R",
        "Joint_L8_R",
    ),
}


class PinkTeleopIK:
    """Solve one STEP task-frame target with optional true joint locking.

    Robot inputs and outputs use IndyDCP units: q in degrees and task poses in
    millimetres/degrees. Pink and Pinocchio operate in radians/metres.
    """

    def __init__(self, config_path: str | Path) -> None:
        try:
            import pinocchio as pin
            import pink
            from pink.exceptions import NoSolutionFound
            from pink.limits import ConfigurationLimit
            from pink.tasks import FrameTask, PostureTask
            from scipy.spatial.transform import Rotation
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Pink IK requires pin, pin-pink, scipy, and a QP solver"
            ) from exc

        self._pin = pin
        self._pink = pink
        self._no_solution_found = NoSolutionFound
        self._configuration_limit = ConfigurationLimit
        self._frame_task_type = FrameTask
        self._posture_task_type = PostureTask
        self._rotation = Rotation

        self._config_path = Path(config_path).expanduser().resolve()
        if not self._config_path.is_file():
            raise FileNotFoundError(
                f"Pink IK config does not exist: {self._config_path}")
        values = yaml.safe_load(self._config_path.read_text(encoding="utf-8"))
        package_root = self._config_path.parent.parent
        urdf_path = (
            package_root / values["robot"]["kinematics_urdf_path"]
        ).resolve()
        if not urdf_path.is_file():
            raise FileNotFoundError(f"Pink IK URDF does not exist: {urdf_path}")

        configured_joints = set(values["joints"])
        if configured_joints != set(DCP_ACTIVE_JOINT_NAMES):
            raise ValueError(
                "Pink IK config joints do not match the humanoid's 18 active "
                "IndyDCP joints")

        self._settings = values["ik"]
        self._full_model = pin.buildModelFromUrdf(str(urdf_path))
        self._validate_model()
        self._task_frames = {
            0: "head_yaw",
            1: self._add_step_tcp_frame("left", values["tcp"]["left"]),
            2: self._add_step_tcp_frame("right", values["tcp"]["right"]),
        }

    def _validate_model(self) -> None:
        for joint_name in DCP_ACTIVE_JOINT_NAMES:
            if not self._full_model.existJointName(joint_name):
                raise ValueError(
                    f"Pink IK URDF has no joint named {joint_name!r}")
            joint = self._full_model.joints[
                self._full_model.getJointId(joint_name)
            ]
            if joint.nq != 1 or joint.nv != 1:
                raise ValueError(
                    f"Pink IK joint {joint_name!r} must have one DoF")
        if self._full_model.getFrameId("head_yaw") >= self._full_model.nframes:
            raise ValueError("Pink IK URDF has no 'head_yaw' frame")

    def _add_step_tcp_frame(self, side: str, settings: dict) -> str:
        parent_frame_name = str(settings["parent_frame"])
        parent_frame_id = self._full_model.getFrameId(parent_frame_name)
        if parent_frame_id >= self._full_model.nframes:
            raise ValueError(
                f"Pink IK URDF has no frame named {parent_frame_name!r}")
        try:
            offset_settings = settings["step_tcp"]
        except KeyError as exc:
            raise ValueError(
                f"Pink IK config is missing tcp.{side}.step_tcp") from exc

        xyz = np.asarray(offset_settings["xyz"], dtype=np.float64)
        rpy = np.asarray(offset_settings["rpy"], dtype=np.float64)
        if xyz.shape != (3,) or rpy.shape != (3,):
            raise ValueError(
                f"Pink IK tcp.{side}.step_tcp must define 3D xyz/rpy")
        if not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(rpy)):
            raise ValueError(
                f"Pink IK tcp.{side}.step_tcp contains non-finite values")

        parent_frame = self._full_model.frames[parent_frame_id]
        parent_joint_id = (
            parent_frame.parentJoint
            if hasattr(parent_frame, "parentJoint")
            else parent_frame.parent
        )
        offset = self._pin.SE3(
            self._rotation.from_euler("xyz", rpy).as_matrix(), xyz)
        frame_name = f"step_{side}_tcp"
        self._full_model.addFrame(self._pin.Frame(
            frame_name,
            parent_joint_id,
            parent_frame_id,
            parent_frame.placement * offset,
            self._pin.FrameType.OP_FRAME,
        ))
        return frame_name

    @staticmethod
    def _validate_vector(value: Any, size: int, name: str) -> np.ndarray:
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (size,):
            raise ValueError(
                f"{name} must have shape ({size},), got {vector.shape}")
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} contains NaN or infinite values")
        return vector

    def _model_configuration(
            self, model: Any, joint_values_rad: dict[str, float]) -> np.ndarray:
        q = self._pin.neutral(model)
        for joint_name, value in joint_values_rad.items():
            if not model.existJointName(joint_name):
                continue
            joint = model.joints[model.getJointId(joint_name)]
            q[joint.idx_q] = value
        return q

    @staticmethod
    def _joint_values(model: Any, q: np.ndarray) -> dict[str, float]:
        values = {}
        for joint_name in DCP_ACTIVE_JOINT_NAMES:
            if not model.existJointName(joint_name):
                continue
            joint = model.joints[model.getJointId(joint_name)]
            values[joint_name] = float(q[joint.idx_q])
        return values

    def _ik_model(
            self,
            full_q: np.ndarray,
            joint_values_rad: dict[str, float],
            arm_index: int,
            lock_non_selected_joints: bool) -> tuple[Any, np.ndarray]:
        if not lock_non_selected_joints:
            return self._full_model, full_q.copy()

        movable_joint_names = set(CHAIN_JOINT_NAMES[arm_index])
        locked_joint_ids = [
            self._full_model.getJointId(joint_name)
            for joint_name in DCP_ACTIVE_JOINT_NAMES
            if joint_name not in movable_joint_names
        ]
        # A reduced model removes locked joints from the QP decision vector;
        # their placements are fixed at the current real-robot configuration.
        model = self._pin.buildReducedModel(
            self._full_model, locked_joint_ids, full_q)
        return model, self._model_configuration(model, joint_values_rad)

    def solve(
            self,
            tpos,
            init_jpos,
            arm_index: int,
            lock_non_selected_joints: bool = False) -> dict:
        if arm_index not in CHAIN_JOINT_NAMES:
            raise ValueError(
                f"Pink IK arm_index must be one of "
                f"{tuple(CHAIN_JOINT_NAMES)}, got {arm_index}")

        target = self._validate_vector(tpos, 6, "Pink IK task target")
        q22_deg = self._validate_vector(
            init_jpos, 22, "Pink IK initial joint position")
        joint_values_rad = dict(zip(
            DCP_ACTIVE_JOINT_NAMES,
            np.deg2rad(q22_deg[:len(DCP_ACTIVE_JOINT_NAMES)]),
        ))
        full_q = self._model_configuration(
            self._full_model, joint_values_rad)
        model, initial_q = self._ik_model(
            full_q,
            joint_values_rad,
            arm_index,
            lock_non_selected_joints,
        )

        configuration = self._pink.Configuration(
            model, model.createData(), initial_q, copy_data=False)
        frame_name = self._task_frames[arm_index]
        frame_task = self._frame_task_type(
            frame_name,
            position_cost=float(self._settings["position_cost"]),
            orientation_cost=float(self._settings["orientation_cost"]),
            lm_damping=float(self._settings["lm_damping"]),
        )
        target_transform = self._pin.SE3(
            self._rotation.from_euler(
                "xyz", target[3:], degrees=True).as_matrix(),
            target[:3] / 1000.0,
        )
        frame_task.set_target(target_transform)

        tasks = [frame_task]
        posture_cost = float(self._settings["posture_cost"])
        if posture_cost > 0.0:
            posture_task = self._posture_task_type(cost=posture_cost)
            posture_task.set_target(initial_q)
            tasks.append(posture_task)

        position_error = float("inf")
        orientation_error = float("inf")
        max_iterations = int(self._settings["max_iterations"])
        integration_dt = float(self._settings["integration_dt"])
        iterations = 0
        for iterations in range(max_iterations + 1):
            error = frame_task.compute_error(configuration)
            position_error = float(np.linalg.norm(error[:3]))
            orientation_error = float(np.linalg.norm(error[3:]))
            if (
                position_error
                < float(self._settings["position_tolerance_m"])
                and orientation_error
                < float(self._settings["orientation_tolerance_rad"])
            ):
                solved_values = dict(joint_values_rad)
                solved_values.update(self._joint_values(
                    model, configuration.q))
                jpos = np.rad2deg([
                    solved_values[name] for name in DCP_ACTIVE_JOINT_NAMES
                ]).tolist()
                return {
                    "success": True,
                    "jpos": jpos,
                    "iterations": iterations,
                    "position_error_m": position_error,
                    "orientation_error_rad": orientation_error,
                }
            if iterations == max_iterations:
                break

            try:
                velocity = self._pink.solve_ik(
                    configuration,
                    tasks=tasks,
                    dt=integration_dt,
                    solver=str(self._settings["solver"]),
                    damping=float(self._settings["damping"]),
                    limits=(self._configuration_limit(model),),
                )
            except self._no_solution_found:
                break
            if not np.all(np.isfinite(velocity)):
                break
            configuration.integrate_inplace(velocity, integration_dt)

        return {
            "success": False,
            "iterations": iterations,
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "error": (
                f"did not converge after {iterations} iterations "
                f"(position={position_error:.6g} m, "
                f"orientation={orientation_error:.6g} rad)"
            ),
        }
