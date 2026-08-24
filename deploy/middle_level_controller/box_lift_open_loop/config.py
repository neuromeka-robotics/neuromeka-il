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

CUSTOM_TASK_CONFIG = deepcopy(_DATA_COLLECTOR_CONFIG.task_config)
CUSTOM_TASK_CONFIG.name = os.path.basename(os.path.dirname(__file__))
CUSTOM_TASK_CONFIG.camera_config = CAMERA_CONFIG(cam_params={})
CUSTOM_TASK_CONFIG.model_config = None
CUSTOM_TASK_CONFIG.data_config = None
CUSTOM_TASK_CONFIG.extra_config = EXTRA_CONFIG(
    home_movement_fn=default_home_movement,
)

TRAJECTORY_PATH = Path(__file__).with_name("trajectories.csv")
JOINT_STATE_LOG_DIR = Path(__file__).with_name("joint_state_logs")

# Do not start streaming unless the robot is already at the first sample.
START_POSITION_TOLERANCE_DEG = 0.5
