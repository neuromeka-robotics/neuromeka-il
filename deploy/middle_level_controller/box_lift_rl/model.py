"""ONNX policy wrapper matching the Genesis move-box observation contract."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from helper.config_utils import ROBOT_CONFIG, TASK_CONFIG
from helper.controller_utils import Empty_NN_policy
from helper.extra_utils import NN_CONTROL_STATE

from .config import (
    JOINT_POSITION_HISTORY_LENGTH,
    MAX_JOINT_TARGET_STEP_DEG,
    ONNX_INPUT_NAME,
    ONNX_OUTPUT_NAME,
    POLICY_ACTION_JOINT_NAMES,
    POLICY_ACTION_SCALES_RAD,
    POLICY_ROBOT_JOINT_INDICES,
    TARGET_BOX_POSITION_BASE_M,
    TARGET_BOX_RPY_RAD,
)


def rotation_matrix_to_rpy(rotation: np.ndarray) -> np.ndarray:
    """Return XYZ roll/pitch/yaw for ``R = Rz(yaw) Ry(pitch) Rx(roll)``."""

    rotation = np.asarray(rotation, dtype=np.float64)
    if rotation.shape != (3, 3):
        raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
    horizontal = np.hypot(rotation[0, 0], rotation[1, 0])
    if horizontal > 1.0e-6:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], horizontal)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], horizontal)
        yaw = 0.0
    return np.asarray([roll, pitch, yaw], dtype=np.float64)


def pose_observation(transform: np.ndarray) -> np.ndarray:
    """Encode position and Euler sin/cos exactly as ``MoveBoxEnv`` does."""

    transform = np.asarray(transform, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("Box transform must be a finite 4x4 matrix")
    rpy = rotation_matrix_to_rpy(transform[:3, :3])
    orientation = np.asarray(
        [
            np.sin(rpy[0]), np.cos(rpy[0]),
            np.sin(rpy[1]), np.cos(rpy[1]),
            np.sin(rpy[2]), np.cos(rpy[2]),
        ],
        dtype=np.float64,
    )
    return np.concatenate((transform[:3, 3], orientation)).astype(np.float32)


def transform_from_position_rpy(
    position_m: Sequence[float], rpy_rad: Sequence[float]
) -> np.ndarray:
    position = np.asarray(position_m, dtype=np.float64)
    roll, pitch, yaw = np.asarray(rpy_rad, dtype=np.float64)
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cy, sy = np.cos(yaw), np.sin(yaw)
    rotation_x = np.asarray([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    rotation_y = np.asarray([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    rotation_z = np.asarray([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation_z @ rotation_y @ rotation_x
    transform[:3, 3] = position
    return transform


class MoveBoxObservationBuilder:
    """Build the 51-D observation expected by the exported move-box policy."""

    def __init__(self) -> None:
        self.num_actions = len(POLICY_ACTION_JOINT_NAMES)
        self.target_box_pose_observation = pose_observation(
            transform_from_position_rpy(
                TARGET_BOX_POSITION_BASE_M, TARGET_BOX_RPY_RAD
            )
        )
        self.reset()

    def reset(self) -> None:
        self.joint_position_history = np.zeros(
            (JOINT_POSITION_HISTORY_LENGTH, self.num_actions),
            dtype=np.float32,
        )
        self.last_action = np.zeros(self.num_actions, dtype=np.float32)
        self.last_last_action = np.zeros(self.num_actions, dtype=np.float32)

    @staticmethod
    def policy_joint_positions_rad(qpos_deg: Sequence[float]) -> np.ndarray:
        qpos = np.asarray(qpos_deg, dtype=np.float64)
        if qpos.shape != (22,) or not np.all(np.isfinite(qpos)):
            raise ValueError(f"Robot q must be a finite 22-vector, got {qpos.shape}")
        return np.deg2rad(qpos[list(POLICY_ROBOT_JOINT_INDICES)]).astype(
            np.float32
        )

    def build(
        self, transform_base_box: np.ndarray, qpos_deg: Sequence[float]
    ) -> np.ndarray:
        joint_position = self.policy_joint_positions_rad(qpos_deg)
        self.joint_position_history = np.roll(
            self.joint_position_history, shift=-1, axis=0
        )
        self.joint_position_history[-1] = joint_position

        observation_parts = [
            pose_observation(transform_base_box),
            # self.target_box_pose_observation,
            self.joint_position_history.reshape(-1),
            self.last_action,
            self.last_last_action,
        ]
        observation = np.concatenate(observation_parts).astype(np.float32)
        if not np.all(np.isfinite(observation)):
            raise ValueError("Policy observation contains NaN or infinite values")
        return observation

    def advance_action(self, action: Sequence[float]) -> None:
        action_array = np.asarray(action, dtype=np.float32)
        if action_array.shape != (self.num_actions,):
            raise ValueError(
                f"Policy action must have shape ({self.num_actions},), "
                f"got {action_array.shape}"
            )
        self.last_last_action[:] = self.last_action
        self.last_action[:] = action_array


def action_to_robot_command(
    policy_action: Sequence[float],
    current_qpos_deg: Sequence[float],
    home_qpos_deg: Sequence[float],
) -> np.ndarray:
    """Apply the simulator's relative radian action and return DCP q22 degrees."""

    action = np.asarray(policy_action, dtype=np.float64)
    current_qpos = np.asarray(current_qpos_deg, dtype=np.float64)
    command = np.asarray(home_qpos_deg, dtype=np.float64).copy()
    if action.shape != (len(POLICY_ACTION_JOINT_NAMES),):
        raise ValueError(f"Policy returned invalid action shape {action.shape}")
    if current_qpos.shape != (22,) or command.shape != (22,):
        raise ValueError("Current and home robot positions must be 22-vectors")
    if not (
        np.all(np.isfinite(action))
        and np.all(np.isfinite(current_qpos))
        and np.all(np.isfinite(command))
    ):
        raise ValueError("Policy action and robot positions must be finite")

    indices = np.asarray(POLICY_ROBOT_JOINT_INDICES, dtype=np.int64)
    current_policy_q_rad = np.deg2rad(current_qpos[indices])
    target_step_rad = action * np.asarray(
        POLICY_ACTION_SCALES_RAD, dtype=np.float64
    )
    target_step_deg = np.rad2deg(target_step_rad)
    largest_step_deg = float(np.max(np.abs(target_step_deg)))
    if largest_step_deg > MAX_JOINT_TARGET_STEP_DEG:
        print(
            f"WARNING: Clamping unsafe one-cycle joint target step: "
            f"{largest_step_deg:.2f} deg exceeds "
            f"{MAX_JOINT_TARGET_STEP_DEG:.2f} deg"
        )
        target_step_rad = np.clip(
            target_step_rad,
            -np.deg2rad(MAX_JOINT_TARGET_STEP_DEG),
            np.deg2rad(MAX_JOINT_TARGET_STEP_DEG),
        )
        # raise RuntimeError(
        #     "Policy requested an unsafe one-cycle joint target step: "
        #     f"{largest_step_deg:.2f} deg exceeds "
        #     f"{MAX_JOINT_TARGET_STEP_DEG:.2f} deg"
        # )

    target_policy_q_rad = current_policy_q_rad + target_step_rad
    command[indices] = np.rad2deg(target_policy_q_rad)
    return command


class NN_policy(Empty_NN_policy):
    """Run the exported RSL-RL actor with the eval script's ONNX backend."""

    use_gripper = False

    def __init__(self, robot_config: ROBOT_CONFIG, task_config: TASK_CONFIG):
        super().__init__(robot_config=robot_config, task_config=task_config)
        if len(robot_config.robot_ids) != 1:
            raise ValueError("The move-box policy requires exactly one EIR robot")
        self.robot_id = robot_config.robot_ids[0]
        model_config = task_config.model_config
        self.model_path = Path(model_config.model_dir) / model_config.model_file
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model does not exist: {self.model_path}")
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "box_lift_rl requires onnxruntime, matching the ONNX path in "
                "nrmk-genesis/experiments/move_box/eval.py"
            ) from exc

        self.device = torch.device(model_config.device)
        self.device_id = (
            self.device.index if self.device.index is not None else 0
        )
        self.element_type = np.float32
        if self.device.type == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError(
                    "CUDA was requested but PyTorch CUDA is unavailable"
                )
            provider = "CUDAExecutionProvider"
            provider_options = {
                "device_id": self.device_id,
                "user_compute_stream": str(
                    torch.cuda.current_stream(self.device).cuda_stream
                ),
            }
            providers = [(provider, provider_options)]
        else:
            provider = "CPUExecutionProvider"
            providers = [provider]

        if provider not in ort.get_available_providers():
            raise RuntimeError(
                f"ONNX Runtime provider '{provider}' is unavailable; "
                f"available providers: {ort.get_available_providers()}"
            )

        self.session = ort.InferenceSession(
            str(self.model_path), providers=providers
        )
        self.io_binding = self.session.io_binding()
        self.actions = torch.empty(
            (1, len(POLICY_ACTION_JOINT_NAMES)),
            device=self.device,
            dtype=torch.float32,
        ).contiguous()
        self.io_binding.bind_output(
            name=ONNX_OUTPUT_NAME,
            device_type=self.device.type,
            device_id=self.device_id,
            element_type=self.element_type,
            shape=tuple(self.actions.shape),
            buffer_ptr=self.actions.data_ptr(),
        )

        self.observation_builder = MoveBoxObservationBuilder()
        self.home_qpos_deg = np.asarray(
            robot_config.robot_params[self.robot_id]["home_pos"],
            dtype=np.float64,
        )
        self._validate_model_contract()

    def _infer(self, observation: np.ndarray) -> np.ndarray:
        policy_obs = torch.as_tensor(
            observation,
            device=self.device,
            dtype=torch.float32,
        ).reshape(1, -1).contiguous()
        self.io_binding.bind_input(
            name=ONNX_INPUT_NAME,
            device_type=self.device.type,
            device_id=self.device_id,
            element_type=self.element_type,
            shape=tuple(policy_obs.shape),
            buffer_ptr=policy_obs.data_ptr(),
        )
        self.session.run_with_iobinding(self.io_binding)
        action = self.actions[0].detach().cpu().numpy().copy()
        if action.shape != (len(POLICY_ACTION_JOINT_NAMES),):
            raise ValueError(
                f"ONNX action must have shape ({len(POLICY_ACTION_JOINT_NAMES)},), "
                f"got {action.shape}"
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("ONNX action contains NaN or infinite values")
        return action

    def _validate_model_contract(self) -> None:
        dummy_observation = self.observation_builder.build(
            np.eye(4), self.home_qpos_deg
        )
        try:
            self._infer(dummy_observation)
        except Exception as exc:
            raise ValueError(
                f"ONNX model input does not match the {dummy_observation.size}-D "
                "deployment observation"
            ) from exc
        finally:
            self.observation_builder.reset()

    def reset(self) -> None:
        self.observation_builder.reset()

    def __call__(
        self,
        *,
        transform_base_box: np.ndarray,
        qpos_deg: Sequence[float],
    ) -> dict:
        observation = self.observation_builder.build(
            transform_base_box, qpos_deg
        )
        policy_action = self._infer(observation)
        robot_command = action_to_robot_command(
            policy_action,
            current_qpos_deg=qpos_deg,
            home_qpos_deg=self.home_qpos_deg,
        )
        # self.observation_builder.advance_action(policy_action)
        return {
            "action": policy_action,
            "robot_action_0": robot_command,
            "control_state": NN_CONTROL_STATE.TASK_IN_PROGRESS,
        }
