from typing import List, Dict
import time
import numpy as np
from threading import Thread
from collections import deque
import importlib

from neuromeka import IndyDCP3 as RobotClient
from neuromeka import BlendingType, StopCategory

# helper functions
from helper.extra_utils import ROBOT_CONTROL_MODE, ROBOT_STATE


class Robot:
    def __init__(self, robot_ip: str, gripper_config: Dict | None = None, **kwargs):
        self.robot_client = RobotClient(robot_ip=robot_ip)
        
        if gripper_config is not None and gripper_config["enable"]:
            from communication.gripper import BaseGripperClient
            
            try:
                module = importlib.import_module("communication.gripper")
                GripperClient = getattr(module, gripper_config["type"])
                self.gripper_client: BaseGripperClient = GripperClient(**gripper_config.get("params", {}))
            except ModuleNotFoundError:
                raise ValueError(f"No module found for communication.gripper") from None
            except AttributeError:
                raise ValueError(f"{gripper_config['type']} not found in communication.gripper") from None
        else:
            self.gripper_client = None
        if "force_mode" in kwargs:
            print("Setting force mode...")
            self.robot_client.set_force_mode(kwargs["force_mode"])

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
    
    def get_gripper_state(self):
        if self.gripper_client is None:
            gripper_pos = 1.
            grasp_state = False
        else:
            gripper_pos = self.gripper_client.gripper_pos
            grasp_state = self.gripper_client.is_grasping
            
        return {
            "gripper_pos": gripper_pos,
            "grasp_state": grasp_state
        }

    def get_transformed_ft_sensor_data(self):
        return self.robot_client.get_transformed_ft_sensor_data()

    def get_ft_sensor_data(self):
        return self.robot_client.get_ft_sensor_data()

    def get_force_control_gain(self):
        return self.robot_client.get_force_control_gain()

    def get_force_mode(self):
        return self.robot_client.get_force_mode()

    def set_force_mode(self, force_mode_dict):
        return self.robot_client.set_force_mode(force_mode_dict)

    def set_ft_zero(self):
        return self.robot_client.get_ft_zero()
        
    def move_gripper(self, mode: str, value: float):
        if self.gripper_client is None:
            return
        
        value = max(0, min(value, 1))
        if mode == "thread":
            self.gripper_client.MoveGripperWThread(gripper_value=value)
        elif mode == "no_thread":
            self.gripper_client.MoveGripperWOThread(gripper_value=value)
        else:
            raise ValueError


class DualArmRobot(Robot):
    def __init__(self, robot_ip: str, arm_index: int, gripper_config=None,
                 dof: int = 6, **kwargs):
        if arm_index not in (0, 1):
            raise ValueError("arm_index must be 0 or 1")
        if not isinstance(dof, int) or dof <= 0:
            raise ValueError("dof must be a positive integer")
        self.arm_index = arm_index
        self.dof = dof
        super().__init__(robot_ip, gripper_config=gripper_config, **kwargs)

    def get_state(self) -> Dict:
        state = dict(self.robot_client.get_robot_data())
        for key, width in (("q", self.dof), ("qdot", self.dof),
                           ("p", 6), ("pdot", 6), ("ref_frame", 6), ("tool_frame", 6)):
            if key in state:
                start = self.arm_index * width
                state[key] = state[key][start:start + width]
        return state

    @staticmethod
    def _vector(value, size):
        vector = np.asarray(value, dtype=float)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"Expected {size} finite values")
        return vector.tolist()

    def move(self, target_pos, mode="joint_abs", wait=False,
             vel_ratio=50, acc_ratio=50, **kwargs):
        params = dict(arm_index=self.arm_index, vel_ratio=vel_ratio,
                      acc_ratio=acc_ratio,
                      blending_type=kwargs.get("blending_type", BlendingType.NONE))
        if mode == "joint_abs":
            self.robot_client.movej(jtarget=self._vector(target_pos, self.dof), **params)
        elif mode == "task_abs":
            self.robot_client.movel(ttarget=self._vector(target_pos, 6), **params)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        if wait:
            time.sleep(0.2)
            while self.get_state()["op_state"] == ROBOT_STATE.MOVE:
                time.sleep(0.1)

    def set_servo_all(self, enable=True):
        """Enable or disable servos for the shared controller."""
        return self.robot_client.set_servo_all(enable=enable)

    def tele_move(self, action, mode="joint_abs", vel_scale=0.5, acc_scale=0.5, **kwargs):
        """Joint mode takes the full 22-joint target; task mode selects this arm."""
        if mode not in ("joint_abs", "task_abs"):
            raise ValueError(f"Unsupported mode: {mode}")

        if mode == "joint_abs":
            return self.robot_client.movetelej_abs(
                jpos=self._vector(action, 22),
                vel_ratio=vel_scale, acc_ratio=acc_scale)
        else:
            return self.robot_client.movetelel_abs(
                tpos=self._vector(action, 6), arm_index=self.arm_index,
                vel_ratio=vel_scale, acc_ratio=acc_scale)

    def compute_forward_kinematics(self, jpos):
        result = self.robot_client.forward_kin(
            jpos=self._vector(jpos, self.dof), arm_index=self.arm_index)
        result["success"] = str(result.pop("response")["code"]) == "0"
        if not result["success"]:
            result["tpos"] = self.get_state()["p"]
        return result

    def compute_inverse_kinematics(self, tpos, init_jpos):
        # IK expects the whole controller state, including any auxiliary joints.
        joints = list(self.robot_client.get_robot_data()["q"])
        if len(joints) < 2 * self.dof:
            raise ValueError("Controller state does not contain both arms")
        start = self.arm_index * self.dof
        joints[start:start + self.dof] = self._vector(init_jpos, self.dof)
        result = self.robot_client.inverse_kin(
            tpos=self._vector(tpos, 6), init_jpos=joints, arm_index=self.arm_index)
        result["success"] = str(result.pop("response")["code"]) == "0"
        if result["success"]:
            if len(result["jpos"]) != self.dof:
                result["jpos"] = result["jpos"][start:start + self.dof]
        else:
            result["jpos"] = self.get_state()["q"]
        return result


def create_robot(robot_params: Dict) -> Robot:
    """Build the configured class; legacy configurations default to Robot."""
    class_name = robot_params.get("class_name", "Robot")
    robot_class = globals().get(class_name)
    if not isinstance(robot_class, type) or not issubclass(robot_class, Robot):
        raise ValueError(f"Unknown robot class: {class_name}")
    return robot_class(robot_ip=robot_params["ip"],
                       gripper_config=robot_params.get("gripper"),
                       **robot_params.get("init_kwargs", {}))


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

    def get_gripper_state(self, robot_ids: List[int]):
        state = dict()
        for robot_id in robot_ids:
            state[robot_id] = self.robots[robot_id].get_gripper_state()
        return state
    
    def move_gripper(self, mode: str, value: Dict[int, float]):
        for robot_id, pos in value.items():
            self.robots[robot_id].move_gripper(mode=mode, value=pos)

    def get_transformed_ft_sensor_data(self, robot_ids: List[int]):
        state = dict()
        for robot_id in robot_ids:
            state[robot_id] = self.robots[robot_id].get_transformed_ft_sensor_data()
        return state

    def get_ft_sensor_data(self, robot_ids: List[int]):
        state = dict()
        for robot_id in robot_ids:
            state[robot_id] = self.robots[robot_id].get_ft_sensor_data()
        return state

    def get_force_control_gain(self, robot_ids: List[int]):
        state = dict()
        for robot_id in robot_ids:
            state[robot_id] = self.robots[robot_id].get_force_control_gain()
        return state

    def set_ft_zero(self, robot_ids: List[int]):
        for robot_id in robot_ids:
            self.robots[robot_id].set_ft_zero()

    def get_force_mode(self, robot_ids: List[int]):
        state = dict()
        for robot_id in robot_ids:
            state[robot_id] = self.robots[robot_id].get_force_mode()
        return state

    def set_force_mode(self, robot_ids: List[int], force_mode_dicts: List):
        for robot_id, force_mode_dict in zip(robot_ids, force_mode_dicts):
            self.robots[robot_id].set_force_mode(force_mode_dict)
