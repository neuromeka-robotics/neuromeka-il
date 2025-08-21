import os
from helper.config_utils import *


CUSTOM_ROBOT_CONFIG = ROBOT_CONFIG(
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
                "move_vel_scale": 20.,  # 0 ~ 100
                "move_acc_scale": 20.  # 0 ~ 1000
            }
        }
    },
    
    control_dt = 0.05
)

CUSTOM_TASK_CONFIG = TASK_CONFIG(
    name = os.path.abspath(__file__).split("/")[-2],  # Name of the folder. In current example, "act_il"
    
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
    
    model_config = MODEL_CONFIG(
        model_type = "act",
        model_dir = "/GLOBAL/PATH/TO/MODEL/DIRECTORY",
        model_file = "policy_last.ckpt",
        success_threshold = 0.5,
        device = "cuda"
    ),
    
    data_config = None
)