import os
from pathlib import Path
from functools import partial
from typing import Callable
from helper.config_utils import *
from helper.math_utils import clip_task_space_control
from helper.extra_utils import home_movement_w_open_gripper
from data_collector.teleop_adapter import TeleopAdapter
from data_collector.etc.nami_teleop import NamiTeleopAdapter, NamiTeleopConfig
from communication.robot import NamiRobot


@dataclass
class DataCollectorConfig:
    robot_config: ROBOT_CONFIG
    task_config: TASK_CONFIG
    data_to_collect: dict[str]
    data_to_collect_once: List[str] # Data that needs to be collected only once per episode
    # Existing configurations use the one-device-per-robot adapter by default.
    teleop_adapter_factory: Callable = TeleopAdapter

    def __post_init__(self):
        adapter_factory = self.teleop_adapter_factory
        if isinstance(adapter_factory, partial):
            adapter_factory = adapter_factory.func
        validate_config = getattr(adapter_factory, "validate_config", None)
        if validate_config is not None:
            validate_config(self)


TASK_NAME = "dual_arm"
DUAL_ARM_IP = "192.168.0.95"
DUAL_ARM_IDX = {
    "left": 0,
    "right": 1
}

CONFIGS = {
    TASK_NAME: DataCollectorConfig(
        robot_config=ROBOT_CONFIG(
            robot_params={
                DUAL_ARM_IDX["left"]: {
                    "ip": DUAL_ARM_IP,
                    "class_name": "DualArmRobot",
                    "init_kwargs": {"arm_index": DUAL_ARM_IDX["left"], "dof": 6},
                    "home_pos": [-214.36339, 68.52022, -67.54501, 34.251938, 89.2215, 44.465378],
                    "gripper": {
                        "enable": False,
                        "type": "EndportDHGripperClient",
                        "params": {
                            "robot_ip": DUAL_ARM_IP,
                            "tool_index": DUAL_ARM_IDX["left"],
                            "speed": 100,
                            "force": 100,
                        },
                    },
                    "control": {
                        "vel_scale": 1.,
                        "acc_scale": 10.,
                        "move_vel_scale": 50.,
                        "move_acc_scale": 50.,
                    },
                },
                DUAL_ARM_IDX["right"]: {
                    "ip": DUAL_ARM_IP,
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
        ),
        task_config=TASK_CONFIG(
            name=TASK_NAME,
            camera_config=CAMERA_CONFIG(
                cam_params={
                    "wrist": {
                        "serial": "233622076119",
                        "enable_depth": False,
                    },
                },
            ),
            model_config=None,
            # Both VIVE controllers use the same calibration.
            data_config=DATA_CONFIG(
                device_type="vive",
                device_params={
                    "calib_uvw": [-2.9706547207224423, 0.8474163419310942, -3.0559698162599465],
                },
            ),
            extra_config=EXTRA_CONFIG(home_movement_fn=home_movement_w_open_gripper),
        ),
        data_to_collect={
            "proprio": ["q", "qdot", "p", "pdot"],
            # Arm 1 returns default gripper state as gripper is not enabled.
            "gripper": ["gripper_position"],
            "camera": {"wrist": ["rgb", "intrinsics"]},
            "control": ["tele_abs_control", "gripper_command"],
        },
        data_to_collect_once=["intrinsics"],
    ),
}

# Nami is one controller, even when both arms have Vive devices.
# Override NAMI_IK_CONFIG to use an EIR config outside the sibling checkout.
NAMI_IK_CONFIG = os.environ.get(
    "NAMI_IK_CONFIG",
    str(Path(__file__).resolve().parents[3] / "robot_interface/robot_interface/config/eir.yaml"),
)


# Enable neither, either, or both arms. Set tool_index to each physical endport.
NAMI_GRIPPERS = {
    "per_arm": {
        1: {"enable": False, "type": "EndportDHGripperClient",
            "params": {"tool_index": 0, "speed": 50, "force": 50}},
        2: {"enable": False, "type": "EndportDHGripperClient",
            "params": {"tool_index": 1, "speed": 50, "force": 50}},
    },
}

NAMI_IP = "192.168.0.67"  # TODO: Set your Nami controller IP.
NAMI_HOME_POS = [0., 0., 25., 0., -137., -25., 73., -102., 74., -38., 0.2599228, 137, 25., -73., 102., -74., 38, 0., 0., 0., 0., 0.]  # TODO: Set a validated q22 home pose for your workspace.
# NAMI_HOME_POS = [
#                 0.0, 0.0, 18.5, 0.0,
#                 0.0, -79.985, -96.142, -100.096, -90.0, 0.0, 0.0,
#                 0.0, 79.985, 96.142, 100.096, 90.0, 0.0, 0.0,
#                 0.0, 0.0, 0.0, 0.0,
#             ]

def make_nami_config(mode, gripper_config=None):
    gripper_config = NAMI_GRIPPERS if gripper_config is None else gripper_config
    has_grippers = bool(NamiRobot.get_gripper_configs(gripper_config))
    return DataCollectorConfig(
        robot_config=ROBOT_CONFIG(
            robot_params={0: {
                "ip": NAMI_IP,
                "class_name": "NamiRobot",
                "home_pos": NAMI_HOME_POS,
                "gripper": gripper_config,
                "control": {"vel_scale": 1., "acc_scale": 10.,
                            "move_vel_scale": 50., "move_acc_scale": 50.},
            }},
            control_dt=0.05,
        ),
        task_config=TASK_CONFIG(
            name=f"nami_{mode}",
            camera_config=CAMERA_CONFIG(cam_params={}),
            model_config=None,
            data_config=DATA_CONFIG(
                device_type="vive",
                device_params={
                    # Original Nami calibration; recalibrate for your base stations.
                    "calib_uvw": [1.6082502887310102, 0.003284217687448091, -0.8360883464227169],
                    "per_device": {
                        1: {"device_name": "controller_1"},  # left arm
                        2: {"device_name": "controller_2"},  # right arm
                    },
                },
            ),
        ),
        data_to_collect={
            "proprio": ["q", "qdot", "p", "pdot"],
            **({"gripper": ["gripper_position", "grasp_state"]} if has_grippers else {}),
            "camera": {},
            "control": ["tele_abs_control"] + (["gripper_command"] if has_grippers else []),
        },
        data_to_collect_once=[],
        teleop_adapter_factory=partial(
            NamiTeleopAdapter,
            teleop_config=NamiTeleopConfig(
                robot_control_mode=mode,
                arm_indices=(1, 2),
                pink_config_path=NAMI_IK_CONFIG if mode == "joint_abs" else None,
                # Same default as the original Nami: hold torso and head.
                # Alternatively, use locked_joints=("Joint_L0", "Joint_L1")
                # with lock_non_selected_joints=False to lock only the torso.
                # lock_non_selected_joints=(mode == "joint_abs"),
                lock_non_selected_joints=False,
            ),
        ),
    )


CONFIGS["nami_task"] = make_nami_config("task_abs")
CONFIGS["nami_joint"] = make_nami_config("joint_abs")


def get_config(name):
    """Check runtime requirements only for the selected collection preset."""
    config = CONFIGS[name]
    factory = config.teleop_adapter_factory
    teleop_config = factory.keywords.get("teleop_config") if isinstance(factory, partial) else None
    if isinstance(teleop_config, NamiTeleopConfig) and teleop_config.robot_control_mode == "joint_abs":
        ik_path = Path(teleop_config.pink_config_path).expanduser()
        if not ik_path.is_file():
            raise FileNotFoundError(
                f"Nami joint_abs teleoperation requires an IK config file: {ik_path}. "
                "Set NAMI_IK_CONFIG to an existing EIR YAML file, or use nami_task for task_abs."
            )
    return config


# CONFIGS = {
#     "default": DataCollectorConfig(
#         robot_config = ROBOT_CONFIG(
#             robot_params = {
#                 0: {
#                     "ip": "192.168.0.151",
#                     "home_pos": [0., 0, -90., 0., -90., 0.],
#                     "gripper": {
#                         "enable": False,
#                         "type": "RobotiqUSBClient",
#                         "params": {
#                             "port": "/dev/robotiq_2f85"
#                         }
#                     },
#                     "control": {
#                         "vel_scale": 1.,  # 0 ~ 1
#                         "acc_scale": 10.,  # 0 ~ 10
#                         "move_vel_scale": 50.,  # 0 ~ 100
#                         "move_acc_scale": 50.  # 0 ~ 1000
#                     },
#                     "init_kwargs": {
#                         # Uncomment to use force control (Check if your robot supports force control)
#                         #"force_mode": {
#                         #    "enable": False,
#                         #    "des_force": [0, 0, 0, 0, 0, 0], # Tracking force/torque in X, Y, Z, Roll, Pitch, Yaw. Put 0 for compliant control
#                         #    "enabled_force": [False, False, False, False, False, False], # Enable force control in X, Y, Z, Roll, Pitch, Yaw
#                         #}
#                     }
#                 }
#             },
#             control_dt = 0.05
#         ),
#         task_config = TASK_CONFIG(
#             name = "default",  # Task name to collect data
#             camera_config = CAMERA_CONFIG(
#                 cam_params = {
#                     "wrist": {
#                         "serial": "233522079515",
#                         "enable_depth": False
#                     },
#                 }
#             ),
#             model_config = None,

#             # To use either vive or spacemouse for teleoperation, uncomment the corresponding data_config
#             data_config = DATA_CONFIG(
#                 device_type = "vive",
#                 device_params = {
#                     "calib_uvw": [-1.5901484677973787, -3.130982168874227, 1.1842360521284816],
#                 }
#             ),
#             # data_config = DATA_CONFIG(
#             #     device_type = "spacemouse",
#             #     device_params = {
#             #         "max_pos_speed": 0.1, # m/s
#             #         "max_rot_speed": 0.5, # rad/s
#             #         "fix": {
#             #             "roll": False,
#             #             "pitch": False,
#             #             "yaw": False
#             #         },
#             #         "init_params": {
#             #             "max_value": 500, # {300, 500} 300 for wired version and 500 for wireless
#             #             "deadzone": (0,0,0,0,0,0),
#             #             "frequency": 200, # default: 200
#             #             "get_max_k": 30,
#             #         }

#             #     }
#             # ),

#             # Check controller_utils.py for more details
#             extra_config = EXTRA_CONFIG(
#                 home_movement_fn = home_movement_w_open_gripper,
#                 #control_post_process_fn = lambda control: clip_task_space_control(control=control, range={"z": {"min": 47.355045}, "y": {"min": -624.91986, "max": -544.72253}, "x": {"min": 243.68747, "max": 459.47855}})
#             )
#         ),
#         data_to_collect = {
#             "proprio": ["q", "qdot", "p", "pdot"],
#             "gripper": ["gripper_position", "grasp_state"],
#             "camera": {
#                 "wrist": ["rgb", "intrinsics", "depth"],
#             },
#             "control": ["tele_abs_control", "gripper_command"]
#             #"ft": ["ft_Fx", "ft_Fy", "ft_Fz", "ft_Tx", "ft_Ty", "ft_Tz"],
#             #"force_gain": ["fg_kp", "fg_kv", "fg_kl2", "fg_mass", "fg_damping", "fg_stiffness", "fg_kpf", "fg_kif"],
#             #"force_mode": ["fm_enable", "fm_des_force", "fm_enabled_force"],
#         },
#         data_to_collect_once = ["intrinsics"],
#         #data_to_collect_once = ["fg_kp", "fg_kv", "fg_kl2", "fg_mass", "fg_damping", "fg_stiffness", "fg_kpf", "fg_kif", "fm_enable", "fm_des_force", "fm_enabled_force"]
#     ),
# }
