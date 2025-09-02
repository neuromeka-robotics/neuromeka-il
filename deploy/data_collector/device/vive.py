# base
import time
import numpy as np

# helper functions
from data_collector.device.base import BaseDevice
from helper.math_utils import MathFunc, TaskControlTransformation
from helper.extra_utils import ROBOT_CONTROL_MODE

# config
from data_collector.config import DATA_COLLECTOR_TASK_CONFIG


class Vive(BaseDevice):
    CONTROL_MODE = ROBOT_CONTROL_MODE.TELE_TASK_ABSOLUTE
    
    # static variable shard among all instances
    DEVICE_POOLS = None
    DEVICE_IDX = 1
    
    def __init__(self):
        self.device_name = f"controller_{Vive.DEVICE_IDX}"
        self.control_transform = TaskControlTransformation(
            fixed_robot_to_fixed_device_euler=DATA_COLLECTOR_TASK_CONFIG.data_config.device_params["calib_uvw"])
        
        # Check devices
        if Vive.DEVICE_POOLS is None:
            from third_party.vive import triad_openvr
            Vive.DEVICE_POOLS = triad_openvr.triad_openvr()
            Vive.DEVICE_POOLS.print_discovered_objects()
        
        # Wake-up device
        for _ in range(200):
            Vive.DEVICE_POOLS.devices[self.device_name].trigger_haptic_pulse()
            time.sleep(0.01)
        
        # Set device
        self.device = Vive.DEVICE_POOLS.devices[self.device_name]
        
        # Increase device id for next call
        if f"controller_{Vive.DEVICE_IDX + 1}" in Vive.DEVICE_POOLS.devices.keys():
            Vive.DEVICE_IDX += 1

        self.reset()
    
    def reset(self):
        self.is_initialized = False
        self.control_transform.reset()

    def get_input(self, **kwargs):
        pose = self.device.get_pose_matrix()
        controller_inputs = self.device.get_controller_inputs()
        
        # Communication lost
        if pose is None:
            return {"control": None, "button": None, "trigger": None, "valid": False}
        
        pos = np.multiply(1000, [pose.m[0][3], pose.m[1][3], pose.m[2][3]])
        rotMat = np.array([[pose.m[0][0], pose.m[0][1], pose.m[0][2]],
                           [pose.m[1][0], pose.m[1][1], pose.m[1][2]],
                           [pose.m[2][0], pose.m[2][1], pose.m[2][2]]])
        
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
            
        output = dict()
        output["control"] = self.control_transform.apply()  # 0~2: position / 3~5: orientation (euler)
        output["button"] = controller_inputs["trackpad_pressed"]  # Bool
        output["trigger"] = controller_inputs["trigger"]  # Continuous (0 ~ 1)
        output["valid"] = True  # Bool
        return output
        
