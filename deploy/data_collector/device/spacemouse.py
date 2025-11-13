from typing import Dict

import numpy as np
from scipy.spatial.transform import Rotation
from multiprocessing.managers import SharedMemoryManager
from third_party.spacemouse.spacemouse import Spacemouse

from data_collector.device.base import BaseDevice
from helper.math_utils import MathFunc, TaskControlTransformation
from helper.extra_utils import ROBOT_CONTROL_MODE

class SpaceMouse(BaseDevice):
    # static variable shard among all instances
    DEVICE_IDX = 1
    CONTROL_MODE = ROBOT_CONTROL_MODE.TELE_TASK_ABSOLUTE
    
    def __init__(self, **kwargs):
        self.control_transform = TaskControlTransformation([0., 0., 0.])
        
        # Set device
        self.shm_manager = SharedMemoryManager()
        self.shm_manager.__enter__()
        self.device_params = kwargs["device_params"]
        self.control_dt = kwargs["control_dt"]
                
        self.device = Spacemouse(shm_manager=self.shm_manager, **self.device_params["init_params"])
        self.device.__enter__()
        
    def exit(self):
        try:
            self.device.__exit__()
        except:
            pass
        
        try:
            self.shm_manager.__exit__()
        except:
            pass
    
    def reset(self):
        self.is_initialized = False
        self.control_transform.reset()
        self.target_pose = np.zeros(6, dtype=np.float32)
    
    def get_input(self, **kwargs):
        output = dict()
        output["button"] = self.device.is_button_pressed(0)  # Bool
        output["trigger"] = float(self.device.is_button_pressed(1))  # Continuous (0 ~ 1)
        output["valid"] = True  # Bool
        
        MAX_POS_SPEED = self.device_params["max_pos_speed"]
        MAX_ROT_SPEED = self.device_params["max_rot_speed"]
        control_dt = self.control_dt
        mousePose = self.device.get_motion_state_transformed()
        dpos = mousePose[:3] * (1 + np.abs(mousePose[:3])) * (MAX_POS_SPEED * control_dt)
        drot_xyz = mousePose[3:] * (1 + np.abs(mousePose[3:])) * (MAX_ROT_SPEED * control_dt)
        
        if self.device_params.get("fix"):
            ori_fix_config: Dict = self.device_params.get("fix")
            if ori_fix_config.get("roll"):
                drot_xyz[0] = 0
            if ori_fix_config.get("pitch"):
                drot_xyz[1] = 0
            if ori_fix_config.get("yaw"):
                drot_xyz[2] = 0
            
        self.target_pose += np.concatenate((dpos, drot_xyz))

        pos = self.target_pose[:3] * 1000.  # m -> mm
        rot = self.target_pose[3:]
        rotMat = Rotation.from_euler('xyz', rot).as_matrix()
        self.control_transform.current_device.pos = pos
        self.control_transform.current_device.rot = rotMat
        
        if not self.is_initialized:
            assert "robot_pose" in kwargs.keys(), "Required data for initialization are not given"

            self.control_transform.init_device.pos = self.control_transform.current_device.pos.copy()
            self.control_transform.init_device.rot = self.control_transform.current_device.rot.copy()
            end_pose = np.array(kwargs["robot_pose"])
            end_pose[3:] *= np.pi / 180  # degree -> rad
            self.control_transform.init_robot_end.pos = end_pose[:3]
            self.control_transform.init_robot_end.rot = MathFunc.euler_to_rotMat(end_pose[3:][0], end_pose[3:][1], end_pose[3:][2])
            self.is_initialized = True
            
        output["control"] = self.control_transform.apply()  # 0~2: position / 3~5: orientation (euler)
        return output
