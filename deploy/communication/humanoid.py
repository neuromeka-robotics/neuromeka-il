"""Neuromeka upper-body humanoid robot adapter."""

from typing import Dict, List

import numpy as np

from communication.robot import Robot


class HumanoidRobot(Robot):
    """Adapter for the 22-joint, three-task-chain humanoid controller.

    Direct joint commands are always passed through as full 22-value vectors.
    An 18-value IK result is converted to the required q22 command by appending
    four dummy zeros. Pink locks joints inside IK, while the outgoing command
    holds non-selected joints at the episode's initial reference positions.
    """

    JOINT_DOF = 22
    IK_JOINT_DOF = 18
    DUMMY_JOINT_DOF = 4
    TASK_POSE_DOF = 6
    TASK_STATE_DOF = 18
    VALID_ARM_INDICES = (0, 1, 2)  # head, left arm, right arm
    ARM_JOINT_SLICES = {
        0: slice(2, 4),    # head
        1: slice(4, 11),   # left arm
        2: slice(11, 18),  # right arm
    }

    def __init__(self, robot_ip: str, gripper_config: Dict | None = None,
        **kwargs):
        gripper_config = gripper_config or {"enable": False}
        backend = gripper_config.get("backend", "external")
        if backend not in ["external", "integrated_dh"]:
            raise ValueError(f"Unsupported gripper backend: {backend}")

        # The integrated gripper uses IndyDCP3 directly. Preserve the original
        # external gripper path for configurations that explicitly request it.
        external_config = gripper_config if backend == "external" else None
        super().__init__(
            robot_ip=robot_ip, gripper_config=external_config, **kwargs)

        self._integrated_gripper_enabled = (
            gripper_config.get("enable", False)
            and backend == "integrated_dh"
        )
        self._gripper_tool_index = int(gripper_config.get("tool_index", 0))
        self._gripper_type = int(gripper_config.get("gripper_type", 2))
        self._gripper_activate_command = int(
            gripper_config.get("activate_command", 0))
        self._gripper_position_command = int(
            gripper_config.get("position_command", 2))
        self._gripper_closed_position = int(
            gripper_config.get("closed_position", 0))
        self._gripper_open_position = int(
            gripper_config.get("open_position", 1000))
        self._last_gripper_position = None

        if self._integrated_gripper_enabled:
            self.activate_gripper()

    def get_task_pose(self, state: Dict, arm_index: int = 0) -> List[float]:
        if arm_index not in self.VALID_ARM_INDICES:
            raise ValueError(
                f"arm_index {arm_index} is not supported; expected one of "
                f"{self.VALID_ARM_INDICES}")
        task_state = self._validate_vector(
            state["p"], self.TASK_STATE_DOF, "task-space robot state")
        start = arm_index * self.TASK_POSE_DOF
        return task_state[start:start + self.TASK_POSE_DOF]

    def make_joint_command_from_ik(
            self, value, joint_reference=None,
            arm_index: int = 0,
            lock_non_selected_joints: bool = False) -> List[float]:
        if arm_index not in self.ARM_JOINT_SLICES:
            raise ValueError(
                f"arm_index {arm_index} is not supported; expected one of "
                f"{tuple(self.ARM_JOINT_SLICES)}")
        ik_joints = np.asarray(value, dtype=np.float64)
        if ik_joints.ndim == 1 and ik_joints.size == self.JOINT_DOF:
            ik_q22 = self.validate_joint_command(ik_joints)
        else:
            active_joints = self._validate_vector(
                value, self.IK_JOINT_DOF, "IK joint result")
            ik_q22 = active_joints + [0.] * self.DUMMY_JOINT_DOF

        if lock_non_selected_joints:
            reference_q22 = self.validate_joint_command(joint_reference)
            joint_slice = self.ARM_JOINT_SLICES[arm_index]
            command = reference_q22.copy()
            command[joint_slice] = ik_q22[joint_slice]
            command[self.IK_JOINT_DOF:] = [0.] * self.DUMMY_JOINT_DOF
            return self.validate_joint_command(command)

        return self.validate_joint_command(ik_q22)

    def activate_gripper(self):
        if not self._integrated_gripper_enabled:
            return None
        return self.robot_client.set_gripper_command(
            command=self._gripper_activate_command,
            gripper_type=self._gripper_type,
            pvt_data=[0, 0, 0, 0],
            tool_index=self._gripper_tool_index,
        )

    def move_gripper(self, mode: str, value: float):
        if not self._integrated_gripper_enabled:
            return super().move_gripper(mode=mode, value=value)
        if mode not in ["thread", "no_thread"]:
            raise ValueError(f"Unsupported gripper mode: {mode}")

        value = float(np.clip(value, 0., 1.))
        position = int(round(
            self._gripper_closed_position
            + value * (
                self._gripper_open_position - self._gripper_closed_position)))
        if position == self._last_gripper_position:
            return None

        result = self.robot_client.set_gripper_command(
            command=self._gripper_position_command,
            gripper_type=self._gripper_type,
            pvt_data=[position, 0, 0, 0],
            tool_index=self._gripper_tool_index,
        )
        self._last_gripper_position = position
        return result

    def get_gripper_state(self):
        if not self._integrated_gripper_enabled:
            return super().get_gripper_state()

        data = self.robot_client.get_gripper_data_for(
            self._gripper_tool_index)
        raw_position = float(data.get(
            "gripper_position", self._gripper_open_position))
        travel = self._gripper_open_position - self._gripper_closed_position
        if travel == 0:
            gripper_pos = 0.
        else:
            gripper_pos = float(np.clip(
                (raw_position - self._gripper_closed_position) / travel,
                0., 1.))
        return {
            "gripper_pos": gripper_pos,
            "grasp_state": bool(data.get("gripper_state", 0)),
        }
