import os
from helper.config_utils import *
from helper.math_utils import clip_task_space_control


CUSTOM_ROBOT_CONFIG = ROBOT_CONFIG(
    robot_params = {
        0: {
            "ip": "192.168.0.111",
            "home_pos": [0., 0, -90., 0., -90., 0.],
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

CUSTOM_TASK_CONFIG = TASK_CONFIG(
    name = os.path.abspath(__file__).split("/")[-2],  # Name of the folder. In current example, "act_il"
    
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
    
    model_config = MODEL_CONFIG(
        model_type = "act",
        model_dir = "/home/nrmk/neuromeka-il/train/weights/pick_and_place/2025-08-21-20-24-29",
        model_file = "policy_last.ckpt",
        success_threshold = 0.8,
        device = "cuda"
    ),
    
    data_config = None,
    
    extra_config = EXTRA_CONFIG(
        control_post_process_fn = lambda control: clip_task_space_control(control=control, range={"z": {"min": 265.}})
    )
)