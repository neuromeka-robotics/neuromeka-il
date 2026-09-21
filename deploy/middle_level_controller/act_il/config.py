import os
from pathlib import Path
from helper.config_utils import *
from helper.extra_utils import home_movement_w_open_gripper


DUAL_ARM_IDX = {
    "left": 0,
    "right": 1,
}

# Dual-arm deployment without gripper.
CUSTOM_ROBOT_CONFIG = ROBOT_CONFIG(
    robot_params={
        DUAL_ARM_IDX["left"]: {
            "ip": "192.168.0.95",
            "class_name": "DualArmRobot",
            "init_kwargs": {"arm_index": DUAL_ARM_IDX["left"], "dof": 6},
            "home_pos": [-214.36339, 68.52022, -67.54501, 34.251938, 89.2215, 44.465378],
            "gripper": {"enable": False},
            "control": {
                "vel_scale": 1.,
                "acc_scale": 10.,
                "move_vel_scale": 50.,
                "move_acc_scale": 50.,
            },
        },
        DUAL_ARM_IDX["right"]: {
            "ip": "192.168.0.95",
            "class_name": "DualArmRobot",
            "init_kwargs": {"arm_index": DUAL_ARM_IDX["right"], "dof": 6},
            "home_pos": [-139.61623, -80.11214, 92.796585, -34.95785, -100.46367, -52.19238],
            "gripper": {"enable": False},
            "control": {
                "vel_scale": 1.,
                "acc_scale": 10.,
                "move_vel_scale": 50.,
                "move_acc_scale": 50.,
            },
        },
    },
    control_dt=0.05,
)

CUSTOM_TASK_CONFIG = TASK_CONFIG(
    name="act_il",
    camera_config=CAMERA_CONFIG(
        cam_params={
            "wrist": {"serial": "233622076119", "enable_depth": False},
        },
    ),
    model_config=MODEL_CONFIG(
        model_type="act",
        model_dir=str(Path(__file__).resolve().parents[3]
                      / "train/weights/dual_arm/2026-09-21-18-54-53"),
        model_file="policy_last.ckpt",
        success_threshold=0.8,
        device="cuda",
    ),
    data_config=None,  # Deployment only; DAGGER disabled.
    extra_config=EXTRA_CONFIG(),
)


# # Dual-arm deployment with the left endport gripper.
# CUSTOM_ROBOT_CONFIG = ROBOT_CONFIG(
#     robot_params={
#         DUAL_ARM_IDX["left"]: {
#             "ip": "192.168.0.95",
#             "class_name": "DualArmRobot",
#             "init_kwargs": {"arm_index": DUAL_ARM_IDX["left"], "dof": 6},
#             "home_pos": [-214.36339, 68.52022, -67.54501, 34.251938, 89.2215, 44.465378],
#             "gripper": {
#                 "enable": True,
#                 "type": "EndportDHGripperClient",
#                 "params": {
#                     "robot_ip": "192.168.0.95",
#                     "tool_index": DUAL_ARM_IDX["left"],
#                     "speed": 100,
#                     "force": 100,
#                 },
#             },
#             "control": {
#                 "vel_scale": 1.,
#                 "acc_scale": 10.,
#                 "move_vel_scale": 50.,
#                 "move_acc_scale": 50.,
#             },
#         },
#         DUAL_ARM_IDX["right"]: {
#             "ip": "192.168.0.95",
#             "class_name": "DualArmRobot",
#             "init_kwargs": {"arm_index": DUAL_ARM_IDX["right"], "dof": 6},
#             "home_pos": [-139.61623, -80.11214, 92.796585, -34.95785, -100.46367, -52.19238],
#             "gripper": {"enable": False},
#             "control": {
#                 "vel_scale": 1.,
#                 "acc_scale": 10.,
#                 "move_vel_scale": 50.,
#                 "move_acc_scale": 50.,
#             },
#         },
#     },
#     control_dt=0.05,
# )

# CUSTOM_TASK_CONFIG = TASK_CONFIG(
#     name="act_il",
#     camera_config=CAMERA_CONFIG(
#         cam_params={
#             "wrist": {"serial": "233622076119", "enable_depth": False},
#         },
#     ),
#     model_config=MODEL_CONFIG(
#         model_type="act",
#         model_dir=str(Path(__file__).resolve().parents[3]
#                       / "train/weights/dual_arm_gripper/2026-09-21-19-05-10"),
#         model_file="policy_last.ckpt",
#         success_threshold=0.8,
#         device="cuda",
#     ),
#     data_config=None,  # Deployment only; DAGGER disabled.
#     extra_config=EXTRA_CONFIG(home_movement_fn=home_movement_w_open_gripper),
# )
