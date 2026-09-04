"""Deployment recording for move-box policy shadow replay."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Sequence

import numpy as np

from .config import (
    DCP_ACTIVE_JOINT_NAMES,
    POLICY_ACTION_JOINT_NAMES,
    POLICY_ACTION_SCALES_RAD,
)


ROBOT_JOINT_NAMES = DCP_ACTIVE_JOINT_NAMES + tuple(
    f"Dummy_{index}" for index in range(4)
)


class DeploymentRecorder:
    """Collect one record per real control cycle and save it as NPZ."""

    def __init__(
        self,
        output_dir: Path,
        model_path: Path,
        control_dt: float,
    ) -> None:
        self.output_dir = Path(output_dir)
        self.model_path = Path(model_path).resolve()
        self.control_dt = float(control_dt)
        self.elapsed_s: list[float] = []
        self.wall_time_s: list[float] = []
        self.robot_q_deg: list[np.ndarray] = []
        self.robot_qdot_deg_s: list[np.ndarray] = []
        self.robot_p: list[np.ndarray] = []
        self.robot_pdot: list[np.ndarray] = []
        self.robot_op_state: list[int] = []
        self.policy_action: list[np.ndarray] = []
        self.command_q_deg: list[np.ndarray] = []
        self.action_valid: list[bool] = []
        self.box_transform_base: list[np.ndarray] = []

    def append(
        self,
        *,
        elapsed_s: float,
        wall_time_s: float,
        robot_state: dict,
        policy_action: Sequence[float] | None,
        command_q_deg: Sequence[float] | None,
        box_transform_base: np.ndarray | None,
    ) -> None:
        q_deg = np.asarray(robot_state["q"], dtype=np.float64)
        qdot_deg_s = np.asarray(robot_state["qdot"], dtype=np.float64)
        task_pose = np.asarray(robot_state["p"], dtype=np.float64)
        task_velocity = np.asarray(robot_state["pdot"], dtype=np.float64)
        if q_deg.shape != (22,) or qdot_deg_s.shape != (22,):
            raise ValueError("Recorded robot q and qdot must be 22-vectors")
        if task_pose.shape != (18,) or task_velocity.shape != (18,):
            raise ValueError("Recorded robot p and pdot must be 18-vectors")

        has_action = policy_action is not None and command_q_deg is not None
        if has_action:
            action = np.asarray(policy_action, dtype=np.float64)
            command = np.asarray(command_q_deg, dtype=np.float64)
            transform = np.asarray(box_transform_base, dtype=np.float64)
            if action.shape != (len(POLICY_ACTION_JOINT_NAMES),):
                raise ValueError("Recorded policy action has an invalid shape")
            if command.shape != (22,):
                raise ValueError("Recorded joint command has an invalid shape")
            if transform.shape != (4, 4):
                raise ValueError("Recorded box transform has an invalid shape")
        else:
            action = np.full(len(POLICY_ACTION_JOINT_NAMES), np.nan)
            command = np.full(22, np.nan)
            transform = np.full((4, 4), np.nan)

        self.elapsed_s.append(float(elapsed_s))
        self.wall_time_s.append(float(wall_time_s))
        self.robot_q_deg.append(q_deg.copy())
        self.robot_qdot_deg_s.append(qdot_deg_s.copy())
        self.robot_p.append(task_pose.copy())
        self.robot_pdot.append(task_velocity.copy())
        self.robot_op_state.append(int(robot_state["op_state"]))
        self.policy_action.append(action.copy())
        self.command_q_deg.append(command.copy())
        self.action_valid.append(has_action)
        self.box_transform_base.append(transform.copy())

    def save(self) -> Path | None:
        if not self.elapsed_s:
            return None

        self.output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = self.output_dir / f"box_lift_rl_{timestamp}.npz"
        np.savez_compressed(
            output_path,
            format_version=np.asarray(1, dtype=np.int64),
            model_path=np.asarray(str(self.model_path)),
            nominal_control_dt=np.asarray(self.control_dt, dtype=np.float64),
            robot_joint_names=np.asarray(ROBOT_JOINT_NAMES),
            policy_joint_names=np.asarray(POLICY_ACTION_JOINT_NAMES),
            policy_action_scales_rad=np.asarray(
                POLICY_ACTION_SCALES_RAD, dtype=np.float64
            ),
            elapsed_s=np.asarray(self.elapsed_s, dtype=np.float64),
            wall_time_s=np.asarray(self.wall_time_s, dtype=np.float64),
            robot_q_deg=np.stack(self.robot_q_deg),
            robot_qdot_deg_s=np.stack(self.robot_qdot_deg_s),
            robot_p=np.stack(self.robot_p),
            robot_pdot=np.stack(self.robot_pdot),
            robot_op_state=np.asarray(self.robot_op_state, dtype=np.int64),
            policy_action=np.stack(self.policy_action),
            command_q_deg=np.stack(self.command_q_deg),
            action_valid=np.asarray(self.action_valid, dtype=np.bool_),
            box_transform_base=np.stack(self.box_transform_base),
        )
        return output_path
