"""Shared collection/deployment settings for Nami left-arm visual servoing."""
import os
from pathlib import Path

from helper.config_utils import ROBOT_CONFIG, TASK_CONFIG, CAMERA_CONFIG, MODEL_CONFIG, EXTRA_CONFIG
from helper.extra_utils import start_movement_with_joint_locks

ROOT = Path(__file__).resolve().parents[3]
TASK_NAME = "syringe_visual_servo"
ROBOT_ID = 0
ARM_INDEX = 1  # Nami task chain: 0=head, 1=left, 2=right

CUSTOM_ROBOT_CONFIG = ROBOT_CONFIG(
    robot_params={ROBOT_ID: {
        "ip": "192.168.0.120",  # TODO: Set the Nami controller IP here.
        "class_name": "NamiRobot",
        "home_pos": [  # TODO: Update this q22 home pose (degrees) when changing the setup.
            # Captured from 192.168.0.120 on 2026-11-06 at 18:27:08 KST.
            -24.672165, -0.25110382, 0.0084437, -0.0026853958,  # torso/head
            -132.97899, -38.144188, 64.956215, -79.3939,
            -0.19412842, -52.06403, -52.027473,  # left arm
            136.93849, 24.994835, -72.9547, 102.60062,
            -73.98576, 35.99345, -0.003034973,  # right arm
            0.0, 0.0, 0.0, 0.0,  # dummy slots
        ],
        # Zero-based torso/head joints. Always send SDK indices 1, 2, 3, 4 at startup.
        # Set [] to disable all locking with set_locked_joint(0).
        "locked_joint_indices": [0, 1, 2, 3],
        "gripper": {"enable": False},
        "control": {"vel_scale": 1., "acc_scale": 10.,
                    "move_vel_scale": 60., "move_acc_scale": 60.},
    }}, control_dt=0.05,
)
CAMERA_CONFIG_VS = CAMERA_CONFIG(cam_params={"wrist": {
    "serial": "233522079515",  # TODO: Set the wrist RealSense camera serial here.
    "enable_depth": False,
}})
CUSTOM_TASK_CONFIG = TASK_CONFIG(
    name=TASK_NAME, camera_config=CAMERA_CONFIG_VS,
    model_config=MODEL_CONFIG(model_dir=os.environ.get(
        "VISUAL_SERVO_MODEL_DIR", str(ROOT / "train/weights" / TASK_NAME / "SET_MODEL_DIRECTORY"))),
    data_config=None,
    extra_config=EXTRA_CONFIG(start_movement_fn=start_movement_with_joint_locks),
)
MASK_CONFIG = {
    # Required: set ZERO_SHOT_REFERENCE to the target object's reference image path.
    "target": {"name": "syringe", "image_path": os.environ.get("ZERO_SHOT_REFERENCE")},
    "host": os.environ.get("ZERO_SHOT_HOST", "127.0.0.1"),
    "port": int(os.environ.get("ZERO_SHOT_PORT", "5555")),
    "timeout": 2000,
    "min_mask_area": 20,
    "max_mask_fraction": 0.9,
    # Optional pixel-space ROI: {"min": [xmin, ymin], "max": [xmax, ymax]}.
    "obj_center_limit": None,
}
SERVO_CONFIG = {
    "position_threshold_mm": 5.,
    "yaw_threshold_deg": 0.5,
    "target_reached_mm": 1.,
    "target_reached_deg": 0.5,
    "max_correction_mm": 50.,
    "max_correction_yaw_deg": 15.,
    "max_lost_frames": 5,
    "target_timeout_s": 10.,
    "max_runtime_s": 120.,
}
