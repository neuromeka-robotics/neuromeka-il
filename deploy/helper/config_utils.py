from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Callable
import os

from nrmk_il.helper.utils import get_base_dir

@dataclass
class ROBOT_CONFIG:
    robot_ids: List[int] | None = None
    
    robot_params: Dict[int, Dict] = field(
        default_factory=lambda: {
            0: {
                "ip": "192.168.0.135",
                "home_pos": [-5.0032735, -20.997824, -84.92132, -0.020232247, -73.95311, -3.4574845],
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
        }
    )
    
    control_dt: float = 0.05
    
    def __post_init__(self):
        self.robot_ids = list(self.robot_params.keys())
        
        for robot_id in self.robot_ids:
            assert "ip" in self.robot_params[robot_id].keys()
            
            assert "home_pos" in self.robot_params[robot_id].keys()
            assert isinstance(self.robot_params[robot_id]["home_pos"], list) or isinstance(self.robot_params[robot_id]["home_pos"], None)
            
            assert "gripper" in self.robot_params[robot_id].keys()
            assert "enable" in self.robot_params[robot_id]["gripper"].keys() and isinstance(self.robot_params[robot_id]["gripper"]["enable"], bool)
            
            assert "control" in self.robot_params[robot_id].keys()
            for control_param_type in ["vel_scale", "acc_scale", "move_vel_scale", "move_acc_scale"]:
                assert control_param_type in self.robot_params[robot_id]["control"].keys()


@dataclass
class CAMERA_CONFIG:
    cam_names: List[str] | None = None
    cam_params: Dict[str, Dict] = field(
        default_factory=lambda: {
            "wrist": {
                "serial": "233522079515",
                "exposure": 50,
                "enable_depth": False 
            }
        }
    )
    
    def __post_init__(self):
        self.cam_names = list(self.cam_params.keys())
        
        for cam_name in self.cam_names:
            assert "serial" in self.cam_params[cam_name].keys(), f"Camera serial number is not provided for {cam_name} camera"
            

@dataclass
class MODEL_CONFIG:
    model_type: str = "act"
    model_dir: str = "/GLOBAL/PATH/TO/MODEL/DIRECTORY"
    model_file: str = "policy_last.ckpt"
    success_threshold: float = 1.
    device: str = "cuda"
    port: int = 5555  # Required for external nn model usage via server-client communication
    
    def __post_init__(self):
        assert self.device in ["cpu", "cuda"], f"Unavailable device {self.device}"
    

@dataclass
class DATA_CONFIG:
    data_dir: str = os.path.join(get_base_dir(), "data")
    data_viz_dir: str =  os.path.join(get_base_dir(), "data_viz")
    
    device_type: str = "vive"
    device_params: Dict = field(
        default_factory=lambda: {
            "calib_uvw": [-1.5413670975757867, 3.1056404500490942, 1.1692256106471774]
        }
    )
    
    def __post_init__(self):
        assert self.device_type in ["vive"], f"Unavailable device {self.device_type}"
        
@dataclass
class EXTRA_CONFIG:
    control_post_process_fn : Callable | None = lambda x: x


@dataclass
class TASK_CONFIG:
    name: str = "base"
    camera_config: CAMERA_CONFIG | None = CAMERA_CONFIG()
    model_config: MODEL_CONFIG | None = MODEL_CONFIG()
    data_config: DATA_CONFIG | None = DATA_CONFIG()
    extra_config: EXTRA_CONFIG = EXTRA_CONFIG()
    
    def __post_init__(self):
        assert self.camera_config is not None, \
            "Camera configuration is not provided"
        assert self.model_config is not None or self.data_config is not None, \
            "Either model or data configuration must be provided"
    
    
BASE_ROBOT_CONFIG = ROBOT_CONFIG()
BASE_TASK_CONFIG = TASK_CONFIG()