"""Configuration for the humanoid box-lift open-loop trajectory."""

import os
from copy import deepcopy
from pathlib import Path

from data_collector.config import CONFIGS
from helper.config_utils import CAMERA_CONFIG, EXTRA_CONFIG
from helper.extra_utils import default_home_movement


# Keep the robot and joint-absolute/compliance settings aligned with the
# configuration used to collect the lift-box demonstration.
_DATA_COLLECTOR_CONFIG = CONFIGS["lift_box"]

CUSTOM_ROBOT_CONFIG = deepcopy(_DATA_COLLECTOR_CONFIG.robot_config)
CUSTOM_ROBOT_CONFIG.control_dt = 0.02  # New compliant-plane policies are trained at 50 Hz.

CUSTOM_TASK_CONFIG = deepcopy(_DATA_COLLECTOR_CONFIG.task_config)
CUSTOM_TASK_CONFIG.name = os.path.basename(os.path.dirname(__file__))
CUSTOM_TASK_CONFIG.camera_config = CAMERA_CONFIG(cam_params={})
CUSTOM_TASK_CONFIG.model_config = None
CUSTOM_TASK_CONFIG.data_config = None
CUSTOM_TASK_CONFIG.extra_config = EXTRA_CONFIG(
    home_movement_fn=default_home_movement,
)
CUSTOM_TASK_CONFIG.control_config.compliance.enable = True
CUSTOM_TASK_CONFIG.control_config.robot_control_mode = "joint_abs"

# Select one controller for the complete trajectory.
IK_TYPE = "rl_constraint"  # "pink" replays recorded commands; "rl_constraint" projects them.
RL_CONSTRAINT_MODEL_PATH = "/home/user/yunho/nrmk-genesis/logs/eir-dual-arm-plane/20260914-150210/model_2400.onnx"  # Set the NEW exported ONNX path before enabling.
# RL_CONSTRAINT_MODEL_PATH = "/home/user/yunho/nrmk-genesis/logs/eir-dual-arm-plane/20260914-123544/model_1999.onnx"  # Set the NEW exported ONNX path before enabling.
RL_CONSTRAINT_DRY_RUN = False
# Match current training's free_motion_probability=1.0: policy modes are zero.
# Physical robot compliance remains enabled independently above.
RL_COMPLIANCE_INTERACTIVE = False
RL_COMPLIANCE_COMMAND = True

TRAJECTORY_PATH = Path(__file__).with_name("trajectories.csv")
TRAJECTORY_DT = 0.05  # Source CSV commands are held, without interpolation, at 50 Hz.
# Temporary oscillation diagnostic: keep the desired command at sample zero.
HOLD_FIRST_TARGET = False
JOINT_STATE_LOG_DIR = Path(__file__).with_name("joint_state_logs")

# Do not start streaming unless the robot is already at the first sample.
START_POSITION_TOLERANCE_DEG = 0.5
