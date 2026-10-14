"""Convert one or two Vive task targets into one Nami controller command."""

from dataclasses import dataclass
from pathlib import Path
import warnings
import json

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from data_collector.teleop_adapter import TeleopAdapter
from communication.robot import NamiRobot
from helper.extra_utils import default_start_movement

from data_collector.etc.pink_ik import PinkIK


def start_nami_collection(controller, *, enable_compliance=False, stiffness=None):
    """Enable compliance before teleop; the controller clears it when teleop ends."""
    if enable_compliance:
        for robot_id in controller.robot_ids:
            controller.robot[robot_id].enable_compliance_mode(stiffness=stiffness)
    default_start_movement(controller)
    # Do not issue set_compliance_mode(False) on finish: that command has a firmware bug.


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


TASK_FRAMES = {0: "head_yaw", 1: "left_tcp", 2: "right_tcp"}


class NamiPinkIK:
    """Nami interface: q22 degrees in, q18 degrees out, TCPs in mm/degrees."""

    def __init__(self, config_path: str | Path) -> None:
        config_path = Path(config_path).expanduser().resolve()
        values = yaml.safe_load(config_path.read_text(encoding="utf-8"))
        if set(values["joints"]) != set(DCP_ACTIVE_JOINT_NAMES):
            raise ValueError("Pink IK config must contain Nami's 18 active joints")
        urdf_path = config_path.parent.parent / values["robot"]["kinematics_urdf_path"]
        self.solver = PinkIK(urdf_path, DCP_ACTIVE_JOINT_NAMES, values["ik"])
        if self.solver.model.nq != 18 or self.solver.model.nv != 18:
            raise ValueError("Use the fixed-base, 18-joint Nami kinematics URDF")
        self.solver.validate_frames(TASK_FRAMES.values())

    @staticmethod
    def _validate_vector(value, size, name):
        vector = np.asarray(value, dtype=float)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must be a finite vector of shape ({size},)")
        return vector

    def _joint_values(self, jpos):
        q = self._validate_vector(jpos, 22, "Nami joint position")
        return dict(zip(DCP_ACTIVE_JOINT_NAMES, np.deg2rad(q[:18])))

    @staticmethod
    def _validate_arms(arm_indices):
        if set(arm_indices) - set(TASK_FRAMES):
            raise ValueError("Nami arm indices must be 0 (head), 1 (left), or 2 (right)")

    def forward_multi(self, jpos, arm_indices=(1, 2)):
        self._validate_arms(arm_indices)
        poses = self.solver.forward(
            self._joint_values(jpos), [TASK_FRAMES[arm] for arm in arm_indices])
        return {arm: np.concatenate((
            poses[TASK_FRAMES[arm]].translation * 1000.,
            Rotation.from_matrix(poses[TASK_FRAMES[arm]].rotation).as_euler(
                "xyz", degrees=True),
        )).tolist() for arm in arm_indices}

    def solve_multi(self, targets, init_jpos, locked_joints=(), joint_reference=None):
        self._validate_arms(targets)
        frame_targets = {}
        for arm in sorted(targets):
            pose = self._validate_vector(targets[arm], 6, f"Target for arm {arm}")
            frame_targets[TASK_FRAMES[arm]] = self.solver.pin.SE3(
                Rotation.from_euler("xyz", pose[3:], degrees=True).as_matrix(),
                pose[:3] / 1000.,
            )
        result = self.solver.solve(
            frame_targets, self._joint_values(init_jpos), locked_joints,
            self._joint_values(joint_reference) if locked_joints else None)
        # Preserve the original last-valid-iterate fallback on non-convergence.
        result["jpos"] = np.rad2deg([
            result["joint_positions"][name] for name in DCP_ACTIVE_JOINT_NAMES
        ]).tolist()
        del result["joint_positions"]
        result["per_arm_errors"] = {
            arm: result["per_frame_errors"][TASK_FRAMES[arm]] for arm in sorted(targets)}
        del result["per_frame_errors"]
        if not result["success"]:
            result["failed_arms"] = [
                arm for arm in sorted(targets) if TASK_FRAMES[arm] in result["failed_frames"]]
            del result["failed_frames"]
        return result

    def solve(self, tpos, init_jpos, arm_index, **kwargs):
        return self.solve_multi({arm_index: tpos}, init_jpos, **kwargs)


@dataclass(frozen=True)
class NamiTeleopConfig:
    robot_control_mode: str = "task_abs"
    arm_indices: tuple[int, ...] = (1, 2)
    pink_config_path: str | None = None
    locked_joints: tuple[str, ...] = ()
    lock_non_selected_joints: bool = False

    def __post_init__(self):
        if self.robot_control_mode not in ("task_abs", "joint_abs"):
            raise ValueError("robot_control_mode must be task_abs or joint_abs")
        if (not self.arm_indices or len(set(self.arm_indices)) != len(self.arm_indices)
                or set(self.arm_indices) - set(CHAIN_JOINT_NAMES)):
            raise ValueError("arm_indices must contain distinct Nami chain indices (0, 1, 2)")
        unknown = set(self.locked_joints) - set(DCP_ACTIVE_JOINT_NAMES)
        if unknown:
            raise ValueError(f"Unknown locked joints: {sorted(unknown)}")
        if self.robot_control_mode == "task_abs":
            if self.locked_joints or self.lock_non_selected_joints:
                raise ValueError("Joint locking requires joint_abs with Pink IK")
        elif not self.pink_config_path:
            raise ValueError("joint_abs requires pink_config_path")

    @property
    def effective_locked_joints(self):
        locked = set(self.locked_joints)
        if self.lock_non_selected_joints:
            movable = {name for arm in self.arm_indices for name in CHAIN_JOINT_NAMES[arm]}
            locked.update(set(DCP_ACTIVE_JOINT_NAMES) - movable)
        return tuple(name for name in DCP_ACTIVE_JOINT_NAMES if name in locked)


class NamiTeleop:
    """Stateful command conversion, shared by collection and the notebook.

    reset() captures the lock reference at the start of each recording. convert()
    computes commands without moving hardware. In task mode, send each returned
    arm pose with tele_move(pose, mode="task_abs", arm_index=arm). In joint mode,
    send the returned q22 once with tele_move(q22, mode="joint_abs").
    """

    def __init__(self, robot, config: NamiTeleopConfig, pink_solver=None):
        self.robot = robot
        self.config = config
        self.joint_reference = None
        self.last_ik_result = None
        self.pink_solver = pink_solver
        if config.robot_control_mode == "joint_abs" and pink_solver is None:
            self.pink_solver = NamiPinkIK(config.pink_config_path)

    def reset(self, state):
        self.joint_reference = self.robot.validate_joint_command(state["q"])
        self.last_ik_result = None

    def device_poses(self, state):
        return {arm: self.robot.get_task_pose(state, arm) for arm in self.config.arm_indices}

    def convert(self, targets, state):
        if set(targets) != set(self.config.arm_indices):
            raise ValueError("Targets must match the configured arm_indices")
        targets = {arm: self.robot.validate_task_command(targets[arm], arm)
                   for arm in self.config.arm_indices}
        if self.config.robot_control_mode == "task_abs":
            return targets
        elif self.config.robot_control_mode == "joint_abs":
            if self.joint_reference is None:
                raise RuntimeError("Call reset(state) before joint teleoperation")
            result = self.pink_solver.solve_multi(
                targets=targets, init_jpos=self.robot.validate_joint_command(state["q"]),
                locked_joints=self.config.effective_locked_joints,
                joint_reference=self.joint_reference)
            self.last_ik_result = result
            if not result.get("success", False):
                if "jpos" not in result:
                    raise RuntimeError(f"Pink IK failed: {result.get('error', 'no solution')}")
                warnings.warn(
                    "Pink IK did not converge; sending the last valid iterate: "
                    + result.get("error", "no solution"), RuntimeWarning, stacklevel=2)
            command = self.robot.make_joint_command_from_ik(result["jpos"])
            # Preserve the exact captured degree values in the command as well.
            for name in self.config.effective_locked_joints:
                index = DCP_ACTIVE_JOINT_NAMES.index(name)
                command[index] = self.joint_reference[index]
            return command
        else:
            raise ValueError(f"Unsupported control mode: {self.config.robot_control_mode}")

    def recorded_command(self, command):
        if self.config.robot_control_mode == "task_abs":
            return [value for arm in self.config.arm_indices for value in command[arm]]
        return list(command)

    def reset_failed_devices(self, devices):
        """Re-anchor failed Vive targets on their next input, matching Nami."""
        result = self.last_ik_result
        if result is not None and not result.get("success", False):
            for arm in result.get("failed_arms", self.config.arm_indices):
                devices[arm].reset()


class NamiTeleopAdapter(TeleopAdapter):
    """Route selected Vive controllers to one whole-body Nami connection."""

    @staticmethod
    def validate_config(collector_config):
        params = collector_config.robot_config.robot_params
        if len(params) != 1:
            raise ValueError("Nami teleoperation requires exactly one robot instance in robot_params")
        if collector_config.task_config.data_config.device_type != "vive":
            raise ValueError("Nami teleoperation requires Vive devices")
        if not {"q", "p"}.issubset(collector_config.data_to_collect.get("proprio", [])):
            raise ValueError("Nami collection requires q and p proprioception")
        robot_params = next(iter(params.values()))
        grippers = NamiRobot.get_gripper_configs(robot_params.get("gripper"), robot_params["ip"])
        factory_kwargs = getattr(collector_config.teleop_adapter_factory, "keywords", {})
        teleop_config = factory_kwargs.get("teleop_config")
        if teleop_config is not None and set(grippers) - set(teleop_config.arm_indices):
            raise ValueError("Each enabled Nami gripper requires its arm's Vive device")

    def __init__(self, collector_config, *, teleop_config: NamiTeleopConfig):
        super().__init__(collector_config)
        self.teleop_config = teleop_config
        self.robot_id = self.robot_ids[0]
        self.device_ids = list(teleop_config.arm_indices)
        params = collector_config.robot_config.robot_params[self.robot_id]
        self.gripper_arm_indices = tuple(NamiRobot.get_gripper_configs(params.get("gripper"), params["ip"]))
        if set(self.gripper_arm_indices) - set(self.device_ids):
            raise ValueError("Each enabled Nami gripper requires its arm's Vive device")

    def bind(self, robot_cluster, devices, device_mode):
        if device_mode != "task_abs":
            raise ValueError("Nami teleoperation requires task-space device input")
        super().bind(robot_cluster, devices, device_mode)
        if tuple(robot_cluster.robots[self.robot_id].gripper_arm_indices) != self.gripper_arm_indices:
            raise ValueError("Nami robot grippers do not match the collection configuration")
        self.session = NamiTeleop(robot_cluster.robots[self.robot_id], self.teleop_config)
        # Pink converts task-space input when joint-space commands are selected.
        self.control_mode = self.teleop_config.robot_control_mode

    def device_poses(self, buffer_data):
        poses = self.session.device_poses({"p": buffer_data[f"p_{self.robot_id}"]})
        return {f"p_{arm}": pose for arm, pose in poses.items()}

    def reset(self, buffer_data):
        super().reset(buffer_data)
        self.session.reset({"q": buffer_data[f"q_{self.robot_id}"]})

    def convert(self, device_data, buffer_data):
        targets = {arm: self.post_process(device_data[arm]["control"]) for arm in self.device_ids}
        command = self.session.convert(targets, {"q": buffer_data[f"q_{self.robot_id}"]})
        self.session.reset_failed_devices(self.devices)
        grippers = [float(1. - np.round(device_data[arm]["trigger"]))
                    for arm in self.gripper_arm_indices]
        return {self.robot_id: command}, {self.robot_id: grippers}

    def recorded_actions(self, actions):
        return {self.robot_id: self.session.recorded_command(actions[self.robot_id])}

    def send(self, action, mode, vel_scale, acc_scale):
        if mode == "task_abs":
            for arm, pose in action[self.robot_id].items():
                self.session.robot.tele_move(
                    action=pose, mode=mode, arm_index=arm,
                    vel_scale=vel_scale[self.robot_id], acc_scale=acc_scale[self.robot_id])
        else:
            super().send(action, mode, vel_scale, acc_scale)

    @property
    def recording_metadata(self):
        return {
            "robot_control_mode": self.teleop_config.robot_control_mode,
            "arm_indices": self.teleop_config.arm_indices,
            "locked_joints": json.dumps(self.teleop_config.effective_locked_joints),
            "gripper_arm_indices": self.gripper_arm_indices,
        }
