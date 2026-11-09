"""Masked-image ACT with goal-to-goal refinement, as in arXiv:2604.22235."""
import time

import numpy as np
import torch

from helper.extra_utils import NN_CONTROL_STATE
from middle_level_controller.act_il.model import NN_policy as ACTPolicyWrapper
from nrmk_il.policies.selection import ControlMode
from perception.zero_shot_detection import DinoV2MaskLabeler
from .config import MASK_CONFIG, SERVO_CONFIG


def angle_delta(target, current):
    return (np.asarray(target) - current + 180.) % 360. - 180.


def constrain_target(current, predicted, fixed_roll_pitch, settings):
    """Bound each refinement and preserve the initial roll/pitch (mm/degrees)."""
    target = np.asarray(predicted, dtype=float).copy()
    if target.shape != (6,) or not np.all(np.isfinite(target)):
        raise ValueError("Policy returned an invalid task pose")
    delta = target[:3] - current[:3]
    norm = np.linalg.norm(delta)
    target[:3] = current[:3] + delta * min(1., settings["max_correction_mm"] / max(norm, 1e-9))
    target[3:5] = fixed_roll_pitch
    yaw_delta = angle_delta(target[5], current[5])
    target[5] = current[5] + np.clip(yaw_delta, -settings["max_correction_yaw_deg"], settings["max_correction_yaw_deg"])
    return target


class NN_policy(ACTPolicyWrapper):
    def __init__(self, *args, **kwargs):
        if not MASK_CONFIG["target"]["image_path"]:
            raise ValueError("Set ZERO_SHOT_REFERENCE to the target object's reference image path")
        super().__init__(*args, **kwargs)
        if self.control_mode != ControlMode.VISUAL_SERVO or self.n_robots != 1 or self.use_gripper:
            raise ValueError("Use a single_robot, visual_servo ACT checkpoint without gripper actions")
        if self.image_crop:
            raise ValueError("This preset uses full-frame masks; train without extra.image_crop")
        expected = {"observation.images.rgb.wrist"}
        if set(self.policy.config.input_shapes) != expected:
            raise ValueError(f"Visual servoing requires image-only inputs: {expected}")
        self.tracker = DinoV2MaskLabeler(**{key: MASK_CONFIG[key] for key in (
            "host", "port", "timeout", "min_mask_area", "max_mask_fraction")})
        self.settings = SERVO_CONFIG.copy()
        self.reset()

    def reset(self):
        super().reset()
        self.tracker.reset()
        self.final_target = None
        self.fixed_roll_pitch = None
        self.target_started = None
        self.lost_frames = 0

    @torch.inference_mode()
    def __call__(self, **kwargs):
        current = np.asarray(kwargs["end_pose"], dtype=float)
        if current.shape != (6,) or not np.all(np.isfinite(current)):
            raise ValueError("Expected selected Nami arm pose (6 finite values)")
        if self.fixed_roll_pitch is None:
            self.fixed_roll_pitch = current[3:5].copy()
        result = self.tracker(kwargs["images.rgb.wrist"], MASK_CONFIG["target"],
                              obj_center_limit=MASK_CONFIG["obj_center_limit"])
        self.last_mask = result["mask"]
        if not result["valid"]:
            self.lost_frames += 1
            self.final_target = None
            return {"robot_action_0": current.copy(), "control_state":
                    NN_CONTROL_STATE.TASK_FAIL if self.lost_frames >= self.settings["max_lost_frames"]
                    else NN_CONTROL_STATE.TASK_IN_PROGRESS}
        self.lost_frames = 0
        # Continue toward the fixed estimate until reached; track masks every frame.
        if self.final_target is not None:
            reached = (np.linalg.norm(self.final_target[:3] - current[:3]) < self.settings["target_reached_mm"]
                       and np.max(np.abs(angle_delta(self.final_target[3:], current[3:]))) < self.settings["target_reached_deg"])
            if not reached:
                state = (NN_CONTROL_STATE.TASK_FAIL if time.monotonic() - self.target_started > self.settings["target_timeout_s"]
                         else NN_CONTROL_STATE.TASK_IN_PROGRESS)
                return {"robot_action_0": self.final_target.copy(), "control_state": state}
        masked = kwargs["images.rgb.wrist"] * self.last_mask
        output = super().__call__(**{**kwargs, "images.rgb.wrist": masked})
        predicted = output["robot_action_0"]
        if not np.all(np.isfinite(predicted)):
            raise ValueError("Non-finite visual-servo prediction")
        distance = np.linalg.norm(predicted[:3] - current[:3])
        yaw = abs(float(angle_delta(predicted[5], current[5])))
        print(f"Visual servo correction: {distance:.2f} mm, yaw {yaw:.2f} deg", flush=True)
        if distance < self.settings["position_threshold_mm"] and yaw < self.settings["yaw_threshold_deg"]:
            output["robot_action_0"] = current.copy()
            output["control_state"] = NN_CONTROL_STATE.TASK_FINISH
            return output
        self.final_target = constrain_target(current, predicted, self.fixed_roll_pitch, self.settings)
        self.target_started = time.monotonic()
        output["robot_action_0"] = self.final_target.copy()
        return output

    def close(self):
        self.tracker.close()
