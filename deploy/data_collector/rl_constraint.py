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


class CompliantPlaneObservation(PlaneObservation):
    """v2/v3: synchronized, term-major histories in the robot base frame."""

    def __init__(self, contract: dict, control_dt: float):
        self.contract = contract
        self.mode_dim = 2 if contract["policy_interface"] == "compliant_plane_history_v3" else 1
        if (contract["pose_frame"] != "robot_base"
                or contract["orientation_representation"] != "rotation_6d_columns"
                or contract["history_order"] != "term_major_offset_order"
                or contract["action_semantics"] != "encoder_joint_delta_rad"):
            raise ValueError("Unsupported compliant-plane observation/action contract")
        if contract.get("observe_constraint_normal") or contract.get("observe_palm_plane_distance"):
            raise ValueError("This deployment requires a policy without privileged plane observations")
        if not np.isclose(control_dt, contract["control_dt"]):
            raise ValueError(f"RL constraint requires control_dt={contract['control_dt']}, got {control_dt}")
        self.indices = np.array([DCP_ACTIVE_JOINT_NAMES.index(n) for n in contract["joint_names"]])
        if len(self.indices) != 14 or set(self.indices) != set(range(4, 18)):
            raise ValueError("RL constraint requires exactly the 14 EIR arm joints")
        self.offsets = np.asarray(contract["joint_pos_history_offsets_steps"], dtype=int)
        if self.offsets.ndim != 1 or not len(self.offsets) or np.any(self.offsets < 0):
            raise ValueError("Invalid joint history offsets")
        self.observe_error = contract["observe_joint_target_error_history"]
        self.include_actions = contract["include_previous_actions"]
        self.include_last_last = contract["include_last_last_action"]
        self.term_dims = ([14, 14] if self.observe_error else [14]) + [18]
        action_dims = 14 * (1 + self.include_last_last) if self.include_actions else 0
        command_only_dims = len(self.offsets) * (sum(self.term_dims) + self.mode_dim) + action_dims
        # Genesis exports predating this metadata field distinguish the two
        # layouts unambiguously through num_obs. Explicit metadata takes priority.
        self.observe_measured = contract.get(
            "observe_measured_palm_pose_history", contract["num_obs"] != command_only_dims)
        self.term_dims += ([18] if self.observe_measured else []) + [self.mode_dim]
        self.num_obs = len(self.offsets) * sum(self.term_dims) + action_dims
        if self.num_obs != contract["num_obs"]:
            raise ValueError("Invalid compliant-plane observation dimensions")
        self.palm_reference = np.asarray(contract["palm_reference_local_m"], dtype=np.float64)
        if self.palm_reference.shape != (2, 3):
            raise ValueError("Expected left/right palm inner-face reference points")
        self.reset()

    def reset(self):
        self.frames = None
        self.sent_command = None
        self.initial_joints = None  # retained as an RL-entry diagnostic; absent from the new observation
        self.last_action = np.zeros(14, dtype=np.float32)
        self.last_last_action = np.zeros(14, dtype=np.float32)

    def activate(self, q, measured_targets):
        self.initial_joints = self.joint_positions(q)

    def pose_observation(self, targets):
        rotations = Rotation.from_euler("xyz", targets[:, 3:], degrees=True)
        positions = targets[:, :3] / 1000. + rotations.apply(self.palm_reference)
        # First two columns, COLUMN-major: R00,R10,R20,R01,R11,R21.
        rot6 = rotations.as_matrix()[:, :, :2].transpose(0, 2, 1).reshape(2, 6)
        return np.concatenate((positions, rot6), axis=1).reshape(-1)

    def update(self, targets, q, *, measured_targets=None, compliance_mode):
        joints = self.joint_positions(q)
        # Legacy measured-pose policies start with a stationary reset frame.
        # Command-only policies consume the supplied desired pose from tick zero.
        if self.frames is None:
            if self.observe_measured:
                targets = measured_targets
            self.sent_command = np.asarray(q, dtype=np.float64).copy()
        terms = [joints]
        if self.observe_error:
            # Last target actually sent; no DCP qdes or synthetic bias/delay.
            terms.append(self.joint_positions(self.sent_command) - joints)
        terms.append(self.pose_observation(self.task_poses(targets)))
        if self.observe_measured:
            terms.append(self.pose_observation(self.task_poses(measured_targets)))
        modes = np.asarray(compliance_mode, dtype=np.float32)
        if modes.shape != (self.mode_dim,) or not np.isin(modes, (0., 1.)).all():
            raise ValueError(f"Expected {self.mode_dim} binary policy compliance-mode values")
        terms.append(modes)
        frame = np.concatenate(terms).astype(np.float32)
        if self.frames is None:
            self.frames = np.tile(frame, (int(self.offsets.max()) + 1, 1))
        else:
            self.frames[1:] = self.frames[:-1]
            self.frames[0] = frame

    def build(self):
        if self.frames is None:
            raise RuntimeError("Update observations before inference")
        sampled = self.frames[self.offsets]
        terms = np.split(sampled, np.cumsum(self.term_dims)[:-1], axis=1)
        parts = [term.reshape(-1) for term in terms]
        if self.include_actions:
            parts.append(self.last_action)
            if self.include_last_last:
                parts.append(self.last_last_action)
        return np.concatenate(parts).astype(np.float32)

    def record_command(self, command, q):
        self.sent_command = np.asarray(command, dtype=np.float64).copy()
        self.last_last_action[:] = self.last_action
        scale = self.contract["action_scale"]
        effective_action = (
            (self.joint_positions(command) - self.joint_positions(q)) / scale
            if scale else 0.)
        self.last_action[:] = effective_action

    def joint_command(self, action, q, locked_reference):
        action = np.asarray(action, dtype=np.float32)
        if action.shape != (14,) or not np.isfinite(action).all():
            raise ValueError("RL constraint returned invalid joint offsets")
        target = self.joint_positions(q) + self.contract["action_scale"] * action
        self.joint_positions(locked_reference)
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
            raise ValueError("Export this model with deploy/export_rl_constraint.py to include its contract")
        contract = json.loads(metadata["dual_arm_plane"])
        interface = contract.get("policy_interface")
        if interface is None:
            builder = PlaneObservation
        elif interface in ("compliant_plane_history_v2", "compliant_plane_history_v3"):
            builder = CompliantPlaneObservation
        else:
            raise ValueError(f"Unsupported policy interface: {interface}")
        self.observation = builder(contract, control_dt)
        self.uses_compliant_history = builder is CompliantPlaneObservation
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

    def update(self, targets, q, *, measured_targets=None, compliance_mode=None):
        if self.uses_compliant_history:
            if self.observation.observe_measured and measured_targets is None:
                raise ValueError("The policy requires local FK gripper-base poses")
            if compliance_mode is None:
                compliance_mode = [float(self.active_mode == "rl_constraint")] * self.observation.mode_dim
            self.observation.update(targets, q, measured_targets=measured_targets,
                                    compliance_mode=compliance_mode)
        else:
            self.observation.update(targets, q)

    def record_command(self, command, q):
        if self.uses_compliant_history:
            self.observation.record_command(command, q)

    def command(self, q, locked_reference):
        obs = self.observation.build()[None, :]
        action, = self.session.run(["actions"], {"obs": obs})
        # PACE produces the training-time encoder state. Use the hardware encoder
        # directly and send this target immediately, without an added bias/delay.
        return self.observation.joint_command(action[0], q, locked_reference)
