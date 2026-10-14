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


class NamiRobot(Robot):
    """q22 = torso(2), head(2), left(7), right(7), dummy(4).

    q uses degrees; p contains head/left/right poses in mm and xyz Euler degrees.
    Joint commands and mode changes affect the whole controller.
    """

    JOINT_DOF = 22
    ARM_INDICES = (0, 1, 2)

    def get_control_state(self) -> Dict:
        """Read whole-body motion and torque signals from IndyDCP."""
        return self.robot_client.get_control_state()

    def enable_compliance_mode(self, stiffness: List[int] | None = None):
        """Enable and confirm compliance before starting Nami teleoperation."""
        if stiffness is not None:
            values = np.asarray(stiffness, dtype=float)
            if values.shape != (self.JOINT_DOF,) or not np.all(np.isfinite(values)):
                raise ValueError(f"compliance stiffness must contain {self.JOINT_DOF} finite values")
            if np.any(values != np.round(values)):
                raise ValueError("compliance stiffness values must be integers")
            stiffness = [int(value) for value in values]

        deadline = time.monotonic() + 10.
        last_error = None
        while time.monotonic() < deadline:
            try:
                if self.robot_client.get_compliance_mode()["enable"]:
                    return
            except Exception as exc:
                last_error = exc
            try:
                self.robot_client.set_compliance_mode(enable=True, stiffness=stiffness)
            except Exception as exc:
                # STEP may apply the transition and close the socket before replying.
                # Confirm the state on the next iteration, as in deprecated/nami.
                last_error = exc
            time.sleep(0.2)
        raise RuntimeError("Nami did not enter compliance mode within 10 seconds") from last_error

    @staticmethod
    def get_gripper_configs(config, robot_ip=None):
        """Return enabled grippers ordered by Nami arm index (1=left, 2=right).

        Legacy single-gripper configs default to the left arm; arm_index overrides it.
        """
        if not config:
            return {}
        per_arm = config.get("per_arm")
        if per_arm is None:
            per_arm = {config.get("arm_index", 1): config}
        elif "enable" in config:
            raise ValueError("Set enable on each Nami gripper, not on the per_arm parent")
        if any(type(arm) is not int or arm not in (1, 2) for arm in per_arm):
            raise ValueError("Nami grippers require arm indices 1 (left) or 2 (right)")
        enabled = {arm: per_arm[arm] for arm in sorted(per_arm)
                   if per_arm[arm].get("enable", False)}
        endports = set()
        for gripper in enabled.values():
            if "type" not in gripper:
                raise ValueError("Each enabled Nami gripper requires a client type")
            if gripper["type"] == "EndportDHGripperClient":
                params = gripper.get("params", {})
                tool_index = params.get("tool_index", 0)
                if type(tool_index) is not int or tool_index < 0:
                    raise ValueError("tool_index must be a non-negative integer")
                endport = (params.get("robot_ip", robot_ip), tool_index)
                if endport in endports:
                    raise ValueError("Nami grippers must use distinct physical endports")
                endports.add(endport)
        return enabled

    def __init__(self, robot_ip: str, gripper_config=None, **kwargs):
        configs = self.get_gripper_configs(gripper_config, robot_ip=robot_ip)
        super().__init__(robot_ip, gripper_config=None, **kwargs)
        self.gripper_arm_indices = tuple(configs)
        self.gripper_clients = {}
        module = importlib.import_module("communication.gripper")
        for arm, config in configs.items():
            client_class = getattr(module, config["type"])
            params = dict(config.get("params", {}))
            if config["type"] == "EndportDHGripperClient":
                params.setdefault("robot_ip", robot_ip)
            self.gripper_clients[arm] = client_class(**params)
        # Keep direct client access available for legacy single-gripper notebooks.
        self.gripper_client = (next(iter(self.gripper_clients.values()))
                               if len(self.gripper_clients) == 1 else None)

    def get_gripper_state(self, arm_index=None):
        """Return vectors in gripper_arm_indices order, or one arm's scalar state."""
        if arm_index is not None:
            if arm_index not in self.gripper_clients:
                raise ValueError(f"No enabled gripper for Nami arm {arm_index}")
            client = self.gripper_clients[arm_index]
            return {"gripper_pos": client.gripper_pos, "grasp_state": client.is_grasping}
        states = [self.get_gripper_state(arm) for arm in self.gripper_arm_indices]
        return {key: [state[key] for state in states]
                for key in ("gripper_pos", "grasp_state")}

    def move_gripper(self, mode: str, value, arm_index=None):
        """Move one arm, an ordered vector of grippers, or all with a scalar."""
        if arm_index is not None and arm_index not in self.gripper_clients:
            raise ValueError(f"No enabled gripper for Nami arm {arm_index}")
        arms = self.gripper_arm_indices if arm_index is None else (arm_index,)
        if not arms:
            return
        if mode not in ("thread", "no_thread"):
            raise ValueError(f"Unsupported gripper mode: {mode}")
        values = np.asarray(value, dtype=float)
        if values.ndim == 0:
            values = np.full(len(arms), values.item())
        if values.shape != (len(arms),) or not np.all(np.isfinite(values)):
            raise ValueError(f"Expected {len(arms)} finite Nami gripper commands")
        for arm, position in zip(arms, np.clip(values, 0., 1.)):
            client = self.gripper_clients[arm]
            if mode == "thread":
                client.MoveGripperWThread(gripper_value=float(position))
            else:
                client.MoveGripperWOThread(gripper_value=float(position))

    @staticmethod
    def _vector(value, size):
        vector = np.asarray(value, dtype=float)
        if vector.shape != (size,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"Expected {size} finite values")
        return vector.tolist()

    def validate_joint_command(self, value):
        return self._vector(value, self.JOINT_DOF)

    def validate_task_command(self, value, arm_index):
        if arm_index not in self.ARM_INDICES:
            raise ValueError("Nami arm_index must be 0 (head), 1 (left), or 2 (right)")
        return self._vector(value, 6)

    def get_task_pose(self, state, arm_index):
        poses = self._vector(state["p"], 18)
        return self.validate_task_command(poses[6 * arm_index:6 * arm_index + 6], arm_index)

    def make_joint_command_from_ik(self, jpos):
        """The model has 18 active joints; DCP requires four trailing dummy zeros."""
        return self._vector(jpos, 18) + [0.] * 4

    def set_servo_all(self, enable=True):
        return self.robot_client.set_servo_all(enable=enable)

    def move(self, target_pos, mode="joint_abs", wait=False,
             vel_ratio=50, acc_ratio=50, arm_index=1, **kwargs):
        params = dict(vel_ratio=vel_ratio, acc_ratio=acc_ratio,
                      blending_type=kwargs.get("blending_type", BlendingType.NONE))
        if mode == "joint_abs":
            result = self.robot_client.movej(
                jtarget=self.validate_joint_command(target_pos), **params)
        elif mode == "task_abs":
            result = self.robot_client.movel(
                ttarget=self.validate_task_command(target_pos, arm_index),
                arm_index=arm_index, **params)
        else:
            raise ValueError(f"Unsupported mode: {mode}")
        if wait:
            time.sleep(0.2)
            while self.get_state()["op_state"] == ROBOT_STATE.MOVE:
                time.sleep(0.1)
        return result

    def tele_move(self, action, mode="joint_abs", vel_scale=0.5, acc_scale=0.5,
                  arm_index=1, **kwargs):
        """Send a full q22 joint target or one selected chain's six-value pose."""
        if mode not in ("joint_abs", "task_abs"):
            raise ValueError(f"Unsupported mode: {mode}")

        if mode == "joint_abs":
            return self.robot_client.movetelej_abs(
                jpos=self.validate_joint_command(action),
                vel_ratio=vel_scale, acc_ratio=acc_scale)
        else:
            return self.robot_client.movetelel_abs(
                tpos=self.validate_task_command(action, arm_index), arm_index=arm_index,
                vel_ratio=vel_scale, acc_ratio=acc_scale)

    def compute_forward_kinematics(self, jpos, arm_index=1):
        self.validate_task_command([0.] * 6, arm_index)
        result = dict(self.robot_client.forward_kin(
            jpos=self.validate_joint_command(jpos), arm_index=arm_index))
        result["success"] = str(result.pop("response")["code"]) == "0"
        return result

    def compute_inverse_kinematics(self, tpos, init_jpos, arm_index=1):
        """Controller IK query; Vive joint teleoperation uses Pink instead."""
        result = dict(self.robot_client.inverse_kin(
            tpos=self.validate_task_command(tpos, arm_index),
            init_jpos=self.validate_joint_command(init_jpos), arm_index=arm_index))
        result["success"] = str(result.pop("response")["code"]) == "0"
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
