import os
from helper.config_utils import *
from helper.math_utils import clip_task_space_control


DATA_COLLECTOR_ROBOT_CONFIG = ROBOT_CONFIG(
    robot_params = {
        0: {
            "ip": "192.168.0.111",
            "home_pos": [0., 0, -90., 0., -90., 0.],
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
)

DATA_COLLECTOR_TASK_CONFIG = TASK_CONFIG(
    name = "broom",  # Task name to collect data
    
    camera_config = CAMERA_CONFIG(
        cam_params = {
            "left": {
                "serial": "207222072747",
                "enable_depth": False
            },
            "right": {
                "serial": "317622073859",
                "enable_depth": False
            }
        }
    ),
    
    model_config = None,
    
    data_config = DATA_CONFIG(
        device_type = "vive",
        device_params = {
            "calib_uvw": [-1.5901484677973787, -3.130982168874227, 1.1842360521284816],
        }
    ),
    
    extra_config = EXTRA_CONFIG(
        control_post_process_fn = lambda control: clip_task_space_control(control=control, range={"z": {"min": 306.}})
    )
)
        