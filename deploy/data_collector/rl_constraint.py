"""Dual-arm plane reflex inference using only NumPy, SciPy and ONNX Runtime."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

from data_collector.pink_ik import DCP_ACTIVE_JOINT_NAMES


class PlaneObservation:
    """Training v5: desired poses, oldest-first measured q history, initial q, previous poses."""

    def __init__(self, contract: dict, control_dt: float):
        self.contract = contract
        if (contract["interface"] != "dual_arm_plane"
                or contract["interface_version"] != 5
                or contract["history_order"] != "oldest_to_newest"
                or contract["pose_order"] != "left_right_xyz_wxyz"
                or contract["action_semantics"] != "measured_joint_delta_rad"):
            raise ValueError("Unsupported dual-arm-plane ONNX observation/action contract")
        if not np.isclose(control_dt, contract["control_dt"]):
            raise ValueError(
                f"RL constraint requires control_dt={contract['control_dt']}, got {control_dt}")
        self.indices = np.array([DCP_ACTIVE_JOINT_NAMES.index(n) for n in contract["joint_names"]])
        if set(self.indices) != set(range(4, 18)) or len(self.indices) != 14:
            raise ValueError("RL constraint requires exactly the 14 EIR arm joints")
        self.history_len = contract["history_len"]
        self.num_obs = 28 + (self.history_len + 1) * len(self.indices)
        if self.num_obs != contract["num_obs"]:
            raise ValueError("Invalid ONNX history dimensions")
        self.translation = np.asarray(contract["world_from_robot_translation_m"])
        self.nominal_quat = np.array([p["nominal_quaternion_wxyz"] for p in contract["palms"]])
        self.reset()

    def reset(self):
        self.joint_history = None
        self.initial_joints = None
        self.current_targets = None
        self.previous_targets = None
        self.palm_reference = None

    def joint_positions(self, q):
        q = np.asarray(q, dtype=np.float64)
        if q.shape != (22,) or not np.isfinite(q).all():
            raise ValueError("RL constraint requires finite q22 in degrees")
        return np.deg2rad(q[self.indices]).astype(np.float32)

    @staticmethod
    def task_poses(targets):
        poses = np.asarray([targets[1], targets[2]], dtype=np.float64)
        if poses.shape != (2, 6) or not np.isfinite(poses).all():
            raise ValueError("RL constraint requires finite left/right XYZ/RPY targets")
        return poses

    def update(self, targets, q):
        """Call once per recording control tick, including all ticks using Pink."""
        joints = self.joint_positions(q)
        if self.joint_history is None:
            self.joint_history = np.tile(joints, (self.history_len, 1))
        else:
            self.joint_history[:-1] = self.joint_history[1:]
            self.joint_history[-1] = joints
        targets = self.task_poses(targets)
        self.previous_targets = self.current_targets if self.current_targets is not None else targets.copy()
        self.current_targets = targets

    def activate(self, q, measured_targets):
        """Latch the RL-entry posture and palm material points without clearing history."""
        self.initial_joints = self.joint_positions(q)
        rotations = Rotation.from_euler("xyz", self.task_poses(measured_targets)[:, 3:], degrees=True)
        normals = rotations.inv().apply(self.contract["plane_normals_world"])
        points = []
        for palm, normal in zip(self.contract["palms"], normals):
            rotation = Rotation.from_quat(palm["rotation_xyzw"])
            normal = rotation.inv().apply(normal)
            sign = np.where(np.abs(normal) < 1e-6, 0., np.sign(normal))
            offset = rotation.apply(-0.5 * np.asarray(palm["size_m"]) * sign)
            points.append(np.asarray(palm["center_m"]) + offset)
        self.palm_reference = np.asarray(points)

    def pose_observation(self, targets):
        rotations = Rotation.from_euler("xyz", targets[:, 3:], degrees=True)
        positions = targets[:, :3] / 1000. + self.translation + rotations.apply(self.palm_reference)
        quat = rotations.as_quat()[:, [3, 0, 1, 2]]
        # q and -q describe the same orientation, but the actor observes their components.
        quat *= np.where((quat * self.nominal_quat).sum(axis=1) < 0., -1., 1.)[:, None]
        return np.concatenate((positions, quat), axis=1).reshape(-1)

    def build(self):
        if self.initial_joints is None or self.current_targets is None:
            raise RuntimeError("Update observations and activate RL before inference")
        return np.concatenate((
            self.pose_observation(self.current_targets), self.joint_history.reshape(-1),
            self.initial_joints, self.pose_observation(self.previous_targets),
        )).astype(np.float32)

    def joint_command(self, action, q, locked_reference):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("RL constraint returned invalid joint offsets")
        target = np.clip(
            self.joint_positions(q) + self.contract["action_scale"] * action,
            self.contract["joint_lower_rad"], self.contract["joint_upper_rad"],
        )
        self.joint_positions(locked_reference)  # validate the complete reference
        command = np.asarray(locked_reference, dtype=np.float64).copy()
        command[self.indices] = np.rad2deg(target)
        command[18:] = 0.
        return command.tolist()


class RLConstraintTeleop:
    """Keep observations warm in Pink and run the actor only when selected."""

    def __init__(self, model_path: str, control_dt: float):
        import onnxruntime as ort

        path = Path(model_path).expanduser()
        options = ort.SessionOptions()
        options.intra_op_num_threads = 1
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"])
        metadata = self.session.get_modelmeta().custom_metadata_map
        if "dual_arm_plane" not in metadata:
            raise ValueError("Export this model with dual_arm_plane/eval.py to include its contract")
        self.observation = PlaneObservation(json.loads(metadata["dual_arm_plane"]), control_dt)
        inputs, outputs = self.session.get_inputs(), self.session.get_outputs()
        if (len(inputs) != 1 or inputs[0].name != "obs"
                or inputs[0].shape[-1] != self.observation.num_obs
                or inputs[0].type != "tensor(float)"
                or len(outputs) != 1 or outputs[0].name != "actions"
                or outputs[0].shape[-1] != 14):
            raise ValueError("ONNX graph does not match the dual-arm deployment contract")
        self.reset()

    def reset(self):
        self.active_mode = "pink"
        self.observation.reset()

    def toggle(self, q, measured_targets):
        if self.active_mode == "pink":
            self.observation.activate(q, measured_targets)
            self.active_mode = "rl_constraint"
        else:
            self.active_mode = "pink"
        return self.active_mode

    def update(self, targets, q):
        self.observation.update(targets, q)

    def command(self, q, locked_reference):
        obs = self.observation.build()[None, :]
        action, = self.session.run(["actions"], {"obs": obs})
        # Training simulates transport/actuator delay. Send the new command now;
        # do not add another artificial delay to the real robot's control path.
        return self.observation.joint_command(action[0], q, locked_reference)
