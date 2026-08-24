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
            1: "left_tcp",
            2: "right_tcp",
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
        for frame_name in ("left_tcp", "right_tcp"):
            if self._full_model.getFrameId(frame_name) >= self._full_model.nframes:
                raise ValueError(f"Pink IK URDF has no TCP frame {frame_name!r}")

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

    def _result_jpos(
            self,
            model: Any,
            configuration: Any,
            initial_joint_values_rad: dict[str, float]) -> list[float]:
        """Return the latest valid Pink iterate in public q18 order."""
        solved_values = dict(initial_joint_values_rad)
        solved_values.update(self._joint_values(model, configuration.q))
        return np.rad2deg([
            solved_values[name] for name in DCP_ACTIVE_JOINT_NAMES
        ]).tolist()

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
                return {
                    "success": True,
                    "jpos": self._result_jpos(
                        model, configuration, joint_values_rad),
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
            # Preserve the final valid iterate so real-time callers can use
            # the best result found before the iteration budget was exhausted.
            "jpos": self._result_jpos(
                model, configuration, joint_values_rad),
            "iterations": iterations,
            "position_error_m": position_error,
            "orientation_error_rad": orientation_error,
            "error": (
                f"did not converge after {iterations} iterations "
                f"(position={position_error:.6g} m, "
                f"orientation={orientation_error:.6g} rad)"
            ),
        }

    def _ik_model_multi(
            self,
            full_q: np.ndarray,
            joint_values_rad: dict[str, float],
            arm_indices: list[int],
            lock_non_selected_joints: bool) -> tuple[Any, np.ndarray]:
        if not lock_non_selected_joints:
            return self._full_model, full_q.copy()

        movable_joint_names = set()
        for arm_idx in arm_indices:
            movable_joint_names.update(CHAIN_JOINT_NAMES[arm_idx])
        locked_joint_ids = [
            self._full_model.getJointId(joint_name)
            for joint_name in DCP_ACTIVE_JOINT_NAMES
            if joint_name not in movable_joint_names
        ]
        model = self._pin.buildReducedModel(
            self._full_model, locked_joint_ids, full_q)
        return model, self._model_configuration(model, joint_values_rad)

    def solve_multi(
            self,
            targets: dict[int, Any],
            init_jpos,
            lock_non_selected_joints: bool = False) -> dict:
        """Solve IK for multiple arm chains simultaneously.

        Args:
            targets: Mapping from arm_index to 6-D task pose target.
            init_jpos: Initial joint position (q22 in degrees).
            lock_non_selected_joints: If True, freezes every joint not
                belonging to any of the requested arm chains.

        Returns:
            dict with keys ``success``, ``jpos`` (q18 in degrees), etc.
        """
        arm_indices = sorted(targets.keys())
        for arm_idx in arm_indices:
            if arm_idx not in CHAIN_JOINT_NAMES:
                raise ValueError(
                    f"Pink IK arm_index must be one of "
                    f"{tuple(CHAIN_JOINT_NAMES)}, got {arm_idx}")

        q22_deg = self._validate_vector(
            init_jpos, 22, "Pink IK initial joint position")
        joint_values_rad = dict(zip(
            DCP_ACTIVE_JOINT_NAMES,
            np.deg2rad(q22_deg[:len(DCP_ACTIVE_JOINT_NAMES)]),
        ))
        full_q = self._model_configuration(
            self._full_model, joint_values_rad)
        model, initial_q = self._ik_model_multi(
            full_q,
            joint_values_rad,
            arm_indices,
            lock_non_selected_joints,
        )

        configuration = self._pink.Configuration(
            model, model.createData(), initial_q, copy_data=False)

        tasks = []
        for arm_idx in arm_indices:
            frame_name = self._task_frames[arm_idx]
            frame_task = self._frame_task_type(
                frame_name,
                position_cost=float(self._settings["position_cost"]),
                orientation_cost=float(self._settings["orientation_cost"]),
                lm_damping=float(self._settings["lm_damping"]),
            )
            target = self._validate_vector(
                targets[arm_idx], 6, f"Pink IK target for arm {arm_idx}")
            target_transform = self._pin.SE3(
                self._rotation.from_euler(
                    "xyz", target[3:], degrees=True).as_matrix(),
                target[:3] / 1000.0,
            )
            frame_task.set_target(target_transform)
            tasks.append(frame_task)

        posture_cost = float(self._settings["posture_cost"])
        if posture_cost > 0.0:
            posture_task = self._posture_task_type(cost=posture_cost)
            posture_task.set_target(initial_q)
            tasks.append(posture_task)

        per_arm_errors = {arm_idx: {
            "position_error": float("inf"),
            "orientation_error": float("inf"),
        } for arm_idx in arm_indices}

        max_iterations = int(self._settings["max_iterations"])
        integration_dt = float(self._settings["integration_dt"])
        iterations = 0
        for iterations in range(max_iterations + 1):
            all_converged = True
            for arm_idx in arm_indices:
                frame_task = tasks[arm_indices.index(arm_idx)]
                error = frame_task.compute_error(configuration)
                pos_err = float(np.linalg.norm(error[:3]))
                ori_err = float(np.linalg.norm(error[3:]))
                per_arm_errors[arm_idx]["position_error"] = pos_err
                per_arm_errors[arm_idx]["orientation_error"] = ori_err
                if (pos_err >= float(self._settings["position_tolerance_m"])
                        or ori_err >= float(self._settings["orientation_tolerance_rad"])):
                    all_converged = False

            if all_converged:
                return {
                    "success": True,
                    "jpos": self._result_jpos(
                        model, configuration, joint_values_rad),
                    "iterations": iterations,
                    "per_arm_errors": {
                        k: {
                            "position_error_m": v["position_error"],
                            "orientation_error_rad": v["orientation_error"],
                        }
                        for k, v in per_arm_errors.items()
                    },
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
            # Preserve the final valid iterate so real-time callers can use
            # the best result found before the iteration budget was exhausted.
            "jpos": self._result_jpos(
                model, configuration, joint_values_rad),
            "iterations": iterations,
            "failed_arms": [
                arm_idx
                for arm_idx, error in per_arm_errors.items()
                if (
                    error["position_error"]
                    >= float(self._settings["position_tolerance_m"])
                    or error["orientation_error"]
                    >= float(self._settings["orientation_tolerance_rad"])
                )
            ],
            "per_arm_errors": {
                k: {
                    "position_error_m": v["position_error"],
                    "orientation_error_rad": v["orientation_error"],
                }
                for k, v in per_arm_errors.items()
            },
            "error": (
                f"did not converge after {iterations} iterations; "
                f"per-arm errors: "
                + ", ".join(
                    f"arm {k}: pos={v['position_error']:.6g}m "
                    f"ori={v['orientation_error']:.6g}rad"
                    for k, v in per_arm_errors.items()
                )
            ),
        }
