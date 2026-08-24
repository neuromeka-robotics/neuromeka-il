"""Convert teleoperation-device output into the configured robot command."""

from dataclasses import dataclass
from typing import List

from communication.robot import Robot


VALID_CONTROL_MODES = ("joint_abs", "task_abs")


class ControlConversionError(RuntimeError):
    """Raised when a device command cannot safely be converted."""


@dataclass(frozen=True)
class ConvertedControl:
    mode: str
    command: List[float]
    ik_success: bool = True
    ik_error: str | None = None


def convert_device_control(
        robot: Robot,
        device_command,
        device_mode: str,
        robot_mode: str,
        current_state: dict,
        arm_index: int,
        ik_type: str = "step",
        lock_non_selected_joints: bool = False,
        locked_joint_reference=None,
        pink_solver=None) -> ConvertedControl:
    """Return the command that will be sent to and saved for the robot.

    Joint vectors are only checked for exact shape and finite values. They are
    never padded, embedded, clipped, or normalized.
    """
    if device_mode not in VALID_CONTROL_MODES:
        raise ValueError(f"Unsupported device control mode: {device_mode}")
    if robot_mode not in VALID_CONTROL_MODES:
        raise ValueError(f"Unsupported robot control mode: {robot_mode}")
    if ik_type not in ("step", "pink"):
        raise ValueError(f"Unsupported IK type: {ik_type}")
    if ik_type == "step" and lock_non_selected_joints:
        raise ValueError(
            "Joint locking is unavailable for STEP IK; use Pink IK or "
            "disable lock_non_selected_joints")

    ik_success = True
    ik_error = None

    if device_mode == robot_mode == "joint_abs":
        command = robot.validate_joint_command(device_command)
    elif device_mode == robot_mode == "task_abs":
        command = robot.validate_task_command(
            device_command, arm_index=arm_index)
    elif device_mode == "task_abs" and robot_mode == "joint_abs":
        task_command = robot.validate_task_command(
            device_command, arm_index=arm_index)
        initial_joints = robot.validate_joint_command(current_state["q"])
        if ik_type == "step":
            result = robot.compute_inverse_kinematics(
                tpos=task_command,
                init_jpos=initial_joints,
                arm_index=arm_index,
            )
        else:
            if pink_solver is None:
                raise ValueError("Pink IK was selected without a Pink solver")
            result = pink_solver.solve(
                tpos=task_command,
                init_jpos=initial_joints,
                arm_index=arm_index,
                lock_non_selected_joints=lock_non_selected_joints,
            )
        ik_success = bool(result.get("success", False))
        if not ik_success:
            ik_error = result.get("error", "unknown controller error")
            # Pink exposes its final valid iterate on non-convergence. STEP
            # does not, so its existing fail-fast behavior remains unchanged.
            if ik_type != "pink" or "jpos" not in result:
                raise ControlConversionError(
                    f"{ik_type.upper()} inverse kinematics failed for arm "
                    f"{arm_index}: {ik_error}")
        try:
            command_reference = None
            if ik_type == "pink" and lock_non_selected_joints:
                if locked_joint_reference is None:
                    raise ValueError(
                        "Pink joint locking requires the initial joint "
                        "command reference")
                command_reference = robot.validate_joint_command(
                    locked_joint_reference)
            command = robot.make_joint_command_from_ik(
                result["jpos"],
                joint_reference=command_reference,
                arm_index=arm_index,
                # Pink already excludes these joints from its optimization.
                # This additionally holds their outgoing commands at the
                # episode's initial positions instead of measured positions.
                lock_non_selected_joints=lock_non_selected_joints,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ControlConversionError(
                f"Inverse kinematics returned an invalid joint command: {exc}") \
                from exc
    else:  # device joint_abs -> robot task_abs
        joint_command = robot.validate_joint_command(device_command)
        result = robot.compute_forward_kinematics(
            jpos=joint_command,
            arm_index=arm_index,
        )
        if not result.get("success", False):
            raise ControlConversionError(
                f"Forward kinematics failed for arm {arm_index}: "
                f"{result.get('error', 'unknown controller error')}")
        try:
            command = robot.validate_task_command(
                result["tpos"], arm_index=arm_index)
        except (KeyError, TypeError, ValueError) as exc:
            raise ControlConversionError(
                f"Forward kinematics returned an invalid task command: {exc}") \
                from exc

    return ConvertedControl(
        mode=robot_mode,
        command=command,
        ik_success=ik_success,
        ik_error=ik_error,
    )
