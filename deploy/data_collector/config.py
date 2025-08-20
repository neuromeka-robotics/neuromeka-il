import os
from helper.config_utils import *


DATA_COLLECTOR_ROBOT_CONFIG = ROBOT_CONFIG(
    robot_params = {
        0: {
            "ip": "192.168.0.135",
            "home_pos": [0., -90, 0., -90., 0., -90.],
            "gripper": {
                "enable": True,
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
)

DATA_COLLECTOR_TASK_CONFIG = TASK_CONFIG(
    name = "pick_and_place",  # Task name to collect data
    
    camera_config = CAMERA_CONFIG(
        cam_params = {
            "left": {
                "serial": "427622274252",
                "exposure": 30000,
                "enable_depth": False
            },
            "right": {
                "serial": "427622270318",
                "exposure": 30000,
                "enable_depth": False
            }
        }
    ),
    
    data_config = DATA_CONFIG(
        device_type = "vive",
        device_params = {
            "calib_uvw": [-1.5413670975757867, 3.1056404500490942, 1.1692256106471774]
        }
    )
)