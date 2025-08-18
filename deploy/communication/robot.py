from typing import List, Dict
import time
import numpy as np
from threading import Thread
from collections import deque

from neuromeka import IndyDCP3 as RobotClient
from neuromeka import BlendingType, StopCategory

# helper functions
from helper.extra_utils import ROBOT_CONTROL_MODE, ROBOT_STATE

class Robot:
    def __init__(self, robot_ip: str):
        self.robot_client = RobotClient(robot_ip=robot_ip)

    def get_state(self) -> Dict:
        state = self.robot_client.get_robot_data()
        return state
    
    def get_io_state(self) -> Dict:
        state = self.robot_client.get_io_data()
        return state
    
    def start_teleop(self, mode: str = "joint_abs"):
        assert mode in ["joint_abs", "task_abs"], f"Unsupported mode: {mode}"
        
        if mode == "joint_abs":
            self.robot_client.start_teleop(method=ROBOT_CONTROL_MODE.TELE_JOINT_ABSOLUTE)
        elif mode == "task_abs":
            self.robot_client.start_teleop(method=ROBOT_CONTROL_MODE.TELE_TASK_ABSOLUTE)
        else:
            raise NotImplementedError
            
    def stop_teleop(self):
        self.robot_client.stop_teleop()

    def stop_motion(self, mode="emergency"):
        assert mode in ["emergency", "smooth"], f"Unsupported mode: {mode}"

        if mode == "emergency":
            self.robot_client.stop_motion(stop_category=StopCategory.CAT0)
        elif mode == "smooth":
            self.robot_client.stop_motion(stop_category=StopCategory.CAT2)
        else:
            raise NotImplementedError

    def recover(self):
        self.robot_client.recover()

    def set_direct_teaching(self, enable: bool):
        self.robot_client.set_direct_teaching(enable)

    def move(self, 
             target_pos: List[float], 
             mode: str = "joint_abs", 
             wait=False, 
             vel_ratio=50, acc_ratio=50,
             **kwargs):
        assert mode in ["joint_abs", "task_abs"], f"Unsupported mode: {mode}"

        if mode == "joint_abs":
            self.robot_client.movej(
                jtarget=target_pos, 
                vel_ratio=vel_ratio, 
                acc_ratio=acc_ratio, 
                blending_type=kwargs.get("blending_type", BlendingType.NONE))
        elif mode == "task_abs":
            self.robot_client.movel(
                ttarget=target_pos,
                vel_ratio=vel_ratio, 
                acc_ratio=acc_ratio, 
                blending_type=kwargs.get("blending_type", BlendingType.NONE))
        else:
            raise NotImplementedError

        # Operation state based checking
        if wait:
            time.sleep(0.2)
            while self.get_state()["op_state"] == ROBOT_STATE.MOVE:
                time.sleep(0.1)

    def tele_move(self, 
                  action: List[float], 
                  mode: str = "joint_abs", 
                  vel_scale=0.5, acc_scale=0.5,
                  **kwargs):
        assert mode in ["joint_abs", "task_abs"], f"Unsupported mode: {mode}"

        if mode == "joint_abs":
            self.robot_client.movetelej_abs(
                jpos=action,
                vel_ratio=vel_scale,
                acc_ratio=acc_scale
            )
        elif mode == "task_abs":
            self.robot_client.movetelel_abs(
                tpos=action,
                vel_ratio=vel_scale,
                acc_ratio=acc_scale
            )
        else:
            raise NotImplementedError
         
    def compute_forward_kinematics(self, jpos: List[float]):
        fk_return = self.robot_client.forward_kin(jpos=jpos)
        if fk_return["response"]["code"] == "0":
            fk_return["success"] = True
        else:
            fk_return["tpos"] = self.get_state()["p"]  # Just return current p if FK fails
            fk_return["success"] = False
        del fk_return["response"]
        return fk_return
    
    def compute_inverse_kinematics(self, tpos: List[float], init_jpos: List[float]):
        ik_return = self.robot_client.inverse_kin(tpos=tpos, init_jpos=init_jpos)
        if ik_return["response"]["code"] == "0":
            ik_return["success"] = True
        else:
            ik_return["jpos"] = self.get_state()["q"]  # Just return current q if IK fails
            ik_return["success"] = False
        del ik_return["response"]
        return ik_return


class RobotCluster:
    def __init__(self, robots: Dict[int, Robot]):
        self.robot_ids = list(robots.keys())
        self.robots = robots
    
    def get_state(self, robot_ids: List[int]):
        state = dict()
        for robot_id in robot_ids:
            state[robot_id] = self.robots[robot_id].get_state()
        return state

    def get_io_state(self, robot_ids: List[int]):
        state = dict()
        for robot_id in robot_ids:
            state[robot_id] = self.robots[robot_id].get_io_state()
        return state

    def start_teleop(self, robot_ids: List[int], mode: str = "joint_abs"):
        for robot_id in robot_ids:
            self.robots[robot_id].start_teleop(mode=mode)
    
    def stop_teleop(self, robot_ids: List[int]):
        for robot_id in robot_ids:
            self.robots[robot_id].stop_teleop()
    
    def stop_motion(self, robot_ids: List[int], mode="emergency"):
        for robot_id in robot_ids:
            self.robots[robot_id].stop_motion(mode=mode)

    def recover(self, robot_ids: List[int]):
        for robot_id in robot_ids:
            self.robots[robot_id].recover()

    def set_direct_teaching(self, robot_ids: List[int], enable: bool):
        for robot_id in robot_ids:
            self.robots[robot_id].set_direct_teaching(enable=enable)

    def move(self, 
             target_pos: Dict[int, List[float]], 
             vel_ratio=Dict[int, float], acc_ratio=Dict[int, float],
             mode: str = "joint_abs", 
             wait=False, 
             **kwargs):
        for robot_id, pos in target_pos.items():
            self.robots[robot_id].move(
                target_pos=pos,
                mode=mode,
                wait=wait,
                vel_ratio=vel_ratio[robot_id], acc_ratio=acc_ratio[robot_id],
                **kwargs
            )

    def tele_move(self, 
                  action: Dict[int, List[float]], 
                  vel_scale=Dict[int, float], acc_scale=Dict[int, float],
                  mode: str = "joint_abs", 
                  **kwargs):
        for robot_id, pos in action.items():
            self.robots[robot_id].tele_move(
                action=pos,
                mode=mode,
                vel_scale=vel_scale[robot_id], acc_scale=acc_scale[robot_id],
                **kwargs
            )
