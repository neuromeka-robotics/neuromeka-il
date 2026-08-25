"""Real-robot configuration for the Genesis move-box RL policy."""

import os
from pathlib import Path

import numpy as np

from communication.humanoid import HumanoidRobot
from helper.config_utils import (
    CAMERA_CONFIG,
    COMPLIANCE_CONFIG,
    CONTROL_CONFIG,
    EXTRA_CONFIG,
    MODEL_CONFIG,
    ROBOT_CONFIG,
    TASK_CONFIG,
)
from helper.extra_utils import default_home_movement


_WORKSPACE_ROOT = Path(__file__).resolve().parents[4]

CUSTOM_ROBOT_CONFIG = ROBOT_CONFIG(
    robot_class=HumanoidRobot,
    robot_params={
        0: {
            "ip": "192.168.0.180",
            "home_pos": [
                0.0, 0.0, 18.5, 0.0,
                0.0, -79.985, -96.142, -100.096, -90.0, 0.0, 0.0,
                0.0, 79.985, 96.142, 100.096, 90.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            ],
            "gripper": {
                "enable": False,
                "backend": "integrated_dh",
                "tool_index": 0,
                "gripper_type": 2,
                "activate_command": 0,
                "position_command": 2,
                "closed_position": 0,
                "open_position": 1000,
            },
            "control": {
                "vel_scale": 1.0,
                "acc_scale": 10.0,
                "move_vel_scale": 10.0,
                "move_acc_scale": 10.0,
            },
            "init_kwargs": {},
        }
    },
    control_dt=0.05,
)

CUSTOM_TASK_CONFIG = TASK_CONFIG(
    name=os.path.basename(os.path.dirname(__file__)),
    camera_config=CAMERA_CONFIG(
        cam_params={
            "head": {
                "serial": "254622075364",
                "enable_depth": False,
            }
        }
    ),
    model_config=MODEL_CONFIG(
        model_type="onnx",
        # Robot-base-frame policy exported from the move-box training run.
        model_dir=str(
            _WORKSPACE_ROOT
            / "nrmk-genesis/logs/eir-move-box/20260825-182658"
        ),
        model_file="model_400.onnx",
        device="cuda",
    ),
    data_config=None,
    extra_config=EXTRA_CONFIG(home_movement_fn=default_home_movement),
    control_config=CONTROL_CONFIG(
        robot_control_mode="joint_abs",
        compliance=COMPLIANCE_CONFIG(
            enable=True,
            stiffness=[100] * 22,
        ),
    ),
)


VISUALIZE = False

# The real-world ArUco box estimator defines x=long (0.307 m) and y=short
# (0.153 m), but the Genesis sim box defines x=short (~0.15 m) and y=long
# (~0.31 m).  This is a 90-degree mismatch around Z.  Multiplying the real
# box pose (base frame) by this matrix on the right reorients the box frame
# to match the simulation convention.
BOX_FRAME_CORRECTION = np.array(
    [
        [0.0, -1.0, 0.0, 0.0],
        [1.0,  0.0, 0.0, 0.0],
        [0.0,  0.0, 1.0, 0.0],
        [0.0,  0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

CAMERA_NAME = "head"
PSF_CAMERA_CALIBRATION_PATH = Path(
    "/opt/neuromeka/psf/calib_results/Thc_EIR8_260810.json"
)

# Exact actor action order from experiments/move_box/conf/train.yaml.
# Note: Joint_L0 is locked in the current model (14 actuated DOFs).
POLICY_ACTION_JOINT_NAMES = (
    "Joint_L2_R", "Joint_L2_L",
    "Joint_L3_R", "Joint_L3_L",
    "Joint_L4_R", "Joint_L4_L",
    "Joint_L5_R", "Joint_L5_L",
    "Joint_L6_R", "Joint_L6_L",
    "Joint_L7_R", "Joint_L7_L",
    "Joint_L8_R", "Joint_L8_L",
)
POLICY_ACTION_SCALES_RAD = (
    0.2, 0.2,
    0.2, 0.2,
    0.2, 0.2,
    0.2, 0.2,
    0.2, 0.2,
    0.1, 0.1,
    0.1, 0.1,
)

# Fixed 18-joint order returned by the EIR DCP API. Keep this local so the
# deployment mapping cannot change when data-collection code is edited.
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
ROBOT_JOINT_INDEX = {
    name: index for index, name in enumerate(DCP_ACTIVE_JOINT_NAMES)
}
POLICY_ROBOT_JOINT_INDICES = tuple(
    ROBOT_JOINT_INDEX[name] for name in POLICY_ACTION_JOINT_NAMES
)

JOINT_POSITION_HISTORY_LENGTH = 10

# Current 177-D policy input: current box pose (9), 10 x joint position (140),
# previous action (14), and action t-2 (14). These legacy target values are
# retained so a target-conditioned checkpoint can be restored if needed.
TARGET_BOX_POSITION_BASE_M = (0.6, 0.0, 1.2245)
TARGET_BOX_RPY_RAD = (0.0, 0.0, 0.0)

START_POSITION_TOLERANCE_DEG = 0.5
MAX_CONSECUTIVE_BOX_POSE_MISSES = 10

# Reject, rather than clip, unexpectedly large one-cycle policy targets. This
# leaves the training action contract unchanged while preventing an unsafe
# command from reaching the real robot. A unit policy action on a 0.2 scale is
# about 11.46 degrees, so 15 degrees admits the nominal training range.
MAX_JOINT_TARGET_STEP_DEG = 30.0

ONNX_INPUT_NAME = "obs"
ONNX_OUTPUT_NAME = "actions"
