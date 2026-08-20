import os
from pathlib import Path
from helper.config_utils import *
from helper.math_utils import clip_task_space_control
from helper.extra_utils import home_movement_w_open_gripper
from communication.humanoid import HumanoidRobot


ROBOT_INTERFACE_EIR_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "robot_interface/robot_interface/config/eir.yaml"
)


@dataclass
class DataCollectorConfig:
    robot_config: ROBOT_CONFIG
    task_config: TASK_CONFIG
    data_to_collect: dict[str]
    data_to_collect_once: List[str] # Data that needs to be collected only once per episode

    def __post_init__(self):
        expected_control = (
            f"{self.task_config.teleop_config.robot_control_mode}_control")
        configured_controls = self.data_to_collect.get("control", [])
        if expected_control not in configured_controls:
            raise ValueError(
                "data_to_collect['control'] must explicitly contain "
                f"'{expected_control}' for robot_control_mode="
                f"'{self.task_config.teleop_config.robot_control_mode}'")
        wrong_control = (
            "task_abs_control" if expected_control == "joint_abs_control"
            else "joint_abs_control")
        assert wrong_control not in configured_controls, (
            f"data_to_collect['control'] contains '{wrong_control}' but "
            f"robot_control_mode is '{self.task_config.teleop_config.robot_control_mode}'. "
            f"Remove '{wrong_control}' from the control list.")

CONFIGS = {
    "default": DataCollectorConfig(
        robot_config = ROBOT_CONFIG(
            robot_class = HumanoidRobot,
            robot_params = {
                0: {
                    "ip": "192.168.0.180",
                    "home_pos": [-0.022166412, 0.01979957, 0.0222174, 0.025983755, -138.70251, -25.199673, 76.68596, -99.856125, 77.27083, -37.662327, 4.3381577, 137.05807, 25.11176, -73.127144, 102.0329, -73.94338, 37.87939, 0.020310974, 0.0, 0.0, 0.0, 0.0],
                    "gripper": {
                        "enable": False,
                        "backend": "integrated_dh",
                        "tool_index": 0,
                        # Values verified against the IndyDCP example/PSF
                        # backend. All commands send exactly four PVT values.
                        "gripper_type": 2,
                        "activate_command": 0,
                        "position_command": 2,
                        "closed_position": 0,
                        "open_position": 1000,
                    },
                    "control": {
                        "vel_scale": 1.,  # 0 ~ 1
                        "acc_scale": 10.,  # 0 ~ 10
                        "move_vel_scale": 10.,  # 0 ~ 100
                        "move_acc_scale": 10.  # 0 ~ 1000
                    },
                    "init_kwargs": {
                        # Uncomment to use force control (Check if your robot supports force control)
                        #"force_mode": {
                        #    "enable": False,
                        #    "des_force": [0, 0, 0, 0, 0, 0], # Tracking force/torque in X, Y, Z, Roll, Pitch, Yaw. Put 0 for compliant control
                        #    "enabled_force": [False, False, False, False, False, False], # Enable force control in X, Y, Z, Roll, Pitch, Yaw
                        #}
                    }
                }
            },
            control_dt = 0.05
        ),
        task_config = TASK_CONFIG(
            name = "default",  # Task name to collect data
            camera_config = CAMERA_CONFIG(
                cam_params = {
                    "wrist": {
                        "serial": "254622075364",
                        "enable_depth": False
                    },
                }
            ),
            model_config = None,

            # To use either vive or spacemouse for teleoperation, uncomment the corresponding data_config
            data_config = DATA_CONFIG(
                device_type = "vive",
                device_params = {
                    "calib_uvw": [1.5864221543618953, 0.02660983748319314, -2.028333652578424],
                }
            ),
            # data_config = DATA_CONFIG(
            #     device_type = "spacemouse",
            #     device_params = {
            #         "max_pos_speed": 0.1, # m/s
            #         "max_rot_speed": 0.5, # rad/s
            #         "fix": {
            #             "roll": False,
            #             "pitch": False,
            #             "yaw": False
            #         },
            #         "init_params": {
            #             "max_value": 500, # {300, 500} 300 for wired version and 500 for wireless
            #             "deadzone": (0,0,0,0,0,0),
            #             "frequency": 200, # default: 200
            #             "get_max_k": 30, 
            #         }

            #     }
            # ),
            
            # Check controller_utils.py for more details
            extra_config = EXTRA_CONFIG(
                home_movement_fn = default_home_movement,
                #control_post_process_fn = lambda control: clip_task_space_control(control=control, range={"z": {"min": 47.355045}, "y": {"min": -624.91986, "max": -544.72253}, "x": {"min": 243.68747, "max": 459.47855}})
            ),
            teleop_config = TELEOP_CONFIG(
                # arm_index controls which arm chain each teleop device drives.
                # It supports three setups:
                #
                # 1) Single robot, single arm, single device  (int)
                #    arm_index = 1
                #    One VIVE / SpaceMouse controls robot 0, left arm.
                #
                # 2) Single robot (e.g. humanoid), dual arm, two devices  (List[int])
                #    arm_index = [1, 2]
                #    Device 0 -> left arm (1), device 1 -> right arm (2).
                #    joint_abs : Pink solves BOTH arms in ONE call using two
                #                FrameTasks.  The resulting q18 is converted to
                #                q22 and sent as a single move_telej_abs.
                #                lock_non_selected_joints=True freezes the head
                #                and torso; False lets Pink move the torso to
                #                help both arms reach their targets.
                #                STEP IK is NOT supported for this case.
                #    task_abs  : two separate task commands (move_telel_abs),
                #                one per arm, are sent to the same robot.
                #                No IK involved.
                #
                # 3) Two robots, one arm each, two devices  (Dict[int, int])
                #    arm_index = {0: 0, 1: 0}
                #    Robot 0 -> 0 index arm, robot 1 -> 0 index arm.
                #    Each robot receives its own independent command.
                #
                arm_index = 1,  # 0: head, 1: left arm, 2: right arm
                robot_control_mode = "task_abs",
                ik_type = "pink",  # "step" or "pink"
                pink_config_path = str(ROBOT_INTERFACE_EIR_CONFIG),
                # Used only for task_abs device -> Pink IK -> joint_abs robot.
                # True locks every chain except the selected arm/head.
                # STEP IK does not support locking, so this must then be False.
                lock_non_selected_joints = True,
                compliance = COMPLIANCE_CONFIG(
                    enable = True,
                    stiffness = [50] * 22,
                ),
            ),
        ),
        # Saved data keys follow these naming rules:
        #   - Per-robot data : "{type}_{robot_id}"
        #     e.g. q_0, qdot_0, p_0, pdot_0, joint_abs_control_0, gripper_command_0
        #   - Camera data    : "images.{type}.{cam_name}"
        #     e.g. images.rgb.wrist, images.intrinsics.wrist
        # For single-robot dual-arm teleop (arm_index = [1, 2]), the merged
        # joint command is still saved under joint_abs_control_0 because there
        # is only one robot (robot_id = 0).
        data_to_collect = {
            "proprio": ["q", "qdot", "p", "pdot"],
            # "gripper": ["gripper_position", "grasp_state"],
            "camera": {
                "wrist": ["rgb", "intrinsics"],
            },
            # This must match teleop_config.robot_control_mode above.
            "control": ["task_abs_control"],
            # "control": ["joint_abs_control", "gripper_command"]
            #"ft": ["ft_Fx", "ft_Fy", "ft_Fz", "ft_Tx", "ft_Ty", "ft_Tz"],
            #"force_gain": ["fg_kp", "fg_kv", "fg_kl2", "fg_mass", "fg_damping", "fg_stiffness", "fg_kpf", "fg_kif"],
            #"force_mode": ["fm_enable", "fm_des_force", "fm_enabled_force"],
        },
        data_to_collect_once = ["intrinsics"],
        #data_to_collect_once = ["fg_kp", "fg_kv", "fg_kl2", "fg_mass", "fg_damping", "fg_stiffness", "fg_kpf", "fg_kif", "fm_enable", "fm_des_force", "fm_enabled_force"]
    ),
}
