import os
from helper.config_utils import *
from helper.math_utils import clip_task_space_control
from helper.extra_utils import home_movement_w_open_gripper


@dataclass
class DataCollectorConfig:
    robot_config: ROBOT_CONFIG
    task_config: TASK_CONFIG

CONFIGS = {
    "default": DataCollectorConfig(
        robot_config = ROBOT_CONFIG(
            robot_params = {
                0: {
                    "ip": "192.168.0.151",
                    "home_pos": [0., 0., 0., 0., 0., 0.],
                    "gripper": {
                        "enable": False,
                        "type": "RobotiqUSBClient",
                        "params": {
                            "port": "/dev/robotiq_2f85"
                        }
                    },
                    "control": {
                        "vel_scale": 1.,  # 0 ~ 1
                        "acc_scale": 10.,  # 0 ~ 10
                        "move_vel_scale": 50.,  # 0 ~ 100
                        "move_acc_scale": 50.  # 0 ~ 1000
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
                        "serial": "233522079515",
                        "enable_depth": False
                    },
                }
            ),
            model_config = None,
            #data_config = DATA_CONFIG(
            #    device_type = "vive",
            #    device_params = {
            #        "calib_uvw": [-1.5901484677973787, -3.130982168874227, 1.1842360521284816],
            #    }
            #),
            #data_config = DATA_CONFIG(
            #    device_type = "spacemouse",
            #    device_params = {
            #        "max_pos_speed": 0.1, # m/s
            #        "max_rot_speed": 0.5, # rad/s
            #        "fix": {
            #            "roll": False,
            #            "pitch": False,
            #            "yaw": False
            #        },
            #        "init_params": {
            #            "max_value": 500, # {300, 500} 300 for wired version and 500 for wireless
            #            "deadzone": (0,0,0,0,0,0),
            #            "frequency": 200, # default: 200
            #            "get_max_k": 30, 
            #        }

            #    }
            #),
            
            extra_config = EXTRA_CONFIG(
                home_movement_fn = home_movement_w_open_gripper,
                #control_post_process_fn = lambda control: clip_task_space_control(control=control, range={"z": {"min": 47.355045}, "y": {"min": -624.91986, "max": -544.72253}, "x": {"min": 243.68747, "max": 459.47855}})
            )
        )
    ),
}
