from typing import List, Dict
import time
import numpy as np
from threading import Thread
from collections import deque
from scipy.spatial.transform import Rotation, Slerp

from config.robot import ROBOT_CONFIG, CONTROL

from neuromeka import IndyDCP3 as RobotClient
from neuromeka import BlendingType

from deploy.helper.extra_utils import ROBOT_CONTROL_MODE, ROBOT_STATE
from deploy.helper.math_utils import MathFunc


class Robot:
    def __init__(self, 
                 robot_id: int=0, 
                 robot_ip: str | None=None, 
                 env=None):
        self.robot_id = robot_id
        self.env = env

        if robot_ip is not None:
            self.robot_client = RobotClient(robot_ip=robot_ip)
        else:
            self.robot_client = RobotClient(robot_ip=ROBOT_CONFIG[robot_id]["ip"])
            
        self._exec_idle_traj_thread: None | Thread = None
        self._idle_traj_triggered = False
        self._exec_tele_traj_thread: None | Thread = None
        self._tele_traj_triggered = False

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
        """ 
        Execute emergency
        """
        from neuromeka import StopCategory
        
        if mode == "emergency":
            self.robot_client.stop_motion(stop_category=StopCategory.CAT0)
        elif mode == "smooth":
            self.robot_client.stop_motion(stop_category=StopCategory.CAT2)
        else:
            raise ValueError(f"Unsupported stop mode: {mode}")

    def recover(self):
        """
        Recover after executing emergency
        """
        self.robot_client.recover()

    def set_direct_teaching(self, enable: bool):
        self.robot_client.set_direct_teaching(enable)

    def move(self, target_pos: List[float], mode: str = "joint_abs", wait=False, tele_mode=False, **kwargs):
        if tele_mode:
            if wait:
                self.tele_move_to_target(
                    target=target_pos,
                    mode=mode,
                    **kwargs
                )
            else:
                if (self._exec_tele_traj_thread is not None) \
                    and (self._exec_tele_traj_thread.is_alive()):
                    self._tele_traj_triggered = False
                    self._exec_tele_traj_thread.join()

                self._exec_tele_traj_thread = Thread(
                    target=self.tele_move_to_target, 
                    daemon=True,
                    kwargs={
                        "target": target_pos,
                        "mode": mode,
                        **kwargs
                    })
                self._exec_tele_traj_thread.start()
        else:
            if mode == "joint_abs":
                self.robot_client.movej(
                    jtarget=target_pos, 
                    vel_ratio=kwargs.get("vel_ratio", CONTROL["move_vel_scale"]), 
                    acc_ratio=kwargs.get("acc_ratio", CONTROL["move_acc_scale"]), 
                    blending_type=kwargs.get("blending_type", BlendingType.NONE))
                
                if wait:
                    wait_type = kwargs.get("wait_type", "op")
                    if wait_type == "op":
                        # Operation state based checking
                        time.sleep(0.2)  # wait for state transfer
                        while self.get_state()["op_state"] == ROBOT_STATE.MOVE:
                            time.sleep(0.1)
                    elif wait_type == "error":
                        # Error based checking
                        while True:
                            current_joint_pos = self.get_state()["q"]
                            error = np.linalg.norm(np.array(target_pos) - np.array(current_joint_pos))
                            if error < 5.:  # 0.2
                                # self.stop_motion(mode="smooth")
                                # time.sleep(0.1)
                                break
                            time.sleep(0.05)
                    else:
                        raise ValueError(f"Unavailable wait type: {wait_type}")
            elif mode == "task_abs":
                self.robot_client.movel(
                    ttarget=target_pos,
                    vel_ratio=kwargs.get("vel_ratio", CONTROL["move_vel_scale"]), 
                    acc_ratio=kwargs.get("acc_ratio", CONTROL["move_acc_scale"]), 
                    blending_type=kwargs.get("blending_type", BlendingType.NONE))

                if wait:
                    # Operation state based checking
                    time.sleep(0.2)  # wait for state transfer
                    while self.get_state()["op_state"] == ROBOT_STATE.MOVE:
                        time.sleep(0.1)
            else:
                raise NotImplementedError

    def tele_move_to_target(self, target, mode: str, **kwargs):
        self._tele_traj_triggered = True

        # Set start and end pos
        if mode in ["joint_abs", "joint_abs_w_safety"]:
            start_pos = self.get_state()["q"]
        elif mode in ["task_abs", "task_abs_w_safety"]:
            start_pos = self.get_state()["p"]
        else:
            raise ValueError(f"Unavailable control mode {mode}")
        end_pos = target
        assert len(start_pos) == len(end_pos), "Start and end position dimenstion mismatch"

        # Set parameter for interpolation
        if mode in ["joint_abs", "joint_abs_w_safety"]:
            current_error = np.linalg.norm(np.array(start_pos) - np.array(target))
            n_interpolate_traj = min(int(2 * current_error), int(1.5 / CONTROL["period"]))  # Tuning parameter: 2 / Max: 2.5s
        else:
            current_error = np.linalg.norm(np.array(start_pos[:3]) - np.array(target[:3]))
            n_interpolate_traj = min(int(0.4 * current_error), int(1.5 / CONTROL["period"]))  # Tuning parameter: 0.4 / Max: 2.5s

        n_interpolate_traj = max(n_interpolate_traj, 2)
        n_margin_traj = int(0.1 / CONTROL["period"])  # 0.5s
        # n_interpolate_traj = 3
        # n_margin_traj = 3
        n_traj = n_interpolate_traj + n_margin_traj

        interpolated_traj = np.linspace(start_pos, end_pos, n_interpolate_traj)
        if mode in ["task_abs", "task_abs_w_safety"]:
            # interpolate euler angle in rotation matrix space
            R_start = Rotation.from_euler(seq="XYZ", angles=start_pos[3:], degrees=True)
            R_end = Rotation.from_euler(seq="XYZ", angles=end_pos[3:], degrees=True)

            key_times = [0, 1]
            rotations = Rotation.concatenate([R_start, R_end])
            slerp = Slerp(key_times, rotations)
            interp_times = np.linspace(0, 1, n_interpolate_traj)
            interp_rots = slerp(interp_times)
            interpolated_traj[:, 3:] = interp_rots.as_euler("XYZ", degrees=True)

        end_extension = np.tile(end_pos, (n_margin_traj, 1))
        interpolated_traj = np.vstack((interpolated_traj, end_extension))

        command_traj = deque([], maxlen=n_traj)
        command_traj.extend(interpolated_traj)

        while len(command_traj) > 0 and self._tele_traj_triggered:
            control_start = time.time()

            command = command_traj.popleft()

            success = self.tele_move(
                action=command.tolist(),
                mode=mode,
                vel_scale=kwargs.get("vel_scale", CONTROL["vel_scale"]), 
                acc_scale=kwargs.get("acc_scale", CONTROL["acc_scale"]),
                safety_controller=kwargs.get("safety_controller", None)
            )

            if not success:
                command_traj.appendleft(command)
            elif len(command_traj) == 0:
                state = self.get_state()

                if mode in ["joint_abs", "joint_abs_w_safety"]:
                    curr_joint_pos = np.array(state["q"])
                    curr_joint_vel = np.array(state["qdot"])
                    pos_error = np.linalg.norm(curr_joint_pos - command)
                    vel = np.linalg.norm(curr_joint_vel)

                    if pos_error > 0.5 or vel > 0.2:
                        command_traj.appendleft(command)

                elif mode in ["task_abs", "task_abs_w_safety"]:
                    target_task_pos = MathFunc.mm_to_m(np.array(command[:3]))
                    curr_task_pos = MathFunc.mm_to_m(np.array(state["p"][:3]))
                    curr_joint_vel = np.array(state["qdot"])

                    pos_error = np.linalg.norm(curr_task_pos - target_task_pos)
                    vel = np.linalg.norm(curr_joint_vel)

                    if pos_error > 0.01 or vel > 0.2:
                        command_traj.appendleft(command)

            control_end = time.time()
            wait_time = kwargs.get("period", CONTROL["period"]) - (control_end - control_start)
            if wait_time > 0:
                time.sleep(wait_time)
        
        self._tele_traj_triggered = False

    def tele_move(self, action: List[float], mode: str = "joint_abs", **kwargs):
        if mode == "joint_abs":
            self.robot_client.movetelej_abs(
                jpos=action,
                vel_ratio=kwargs.get("vel_scale", CONTROL["vel_scale"]),
                acc_ratio=kwargs.get("acc_scale", CONTROL["acc_scale"])
            )
            return True
        elif mode == "task_abs":
            self.robot_client.movetelel_abs(
                tpos=action,
                vel_ratio=kwargs.get("vel_scale", CONTROL["vel_scale"]),
                acc_ratio=kwargs.get("acc_scale", CONTROL["acc_scale"])
            )
            return True
        else:
            raise NotImplementedError

         
    def compute_forward_kinematics(self, jpos: List[float]):
        fk_return = self.robot_client.forward_kin(jpos=jpos)
        if fk_return["response"]["code"] == "0":
            fk_return["success"] = True
        else:
            fk_return["tpos"] = self.get_state()["p"]  # Just return current p
            fk_return["success"] = False
        del fk_return["response"]
        return fk_return
    
    def compute_inverse_kinematics(self, tpos: List[float], init_jpos: List[float]):
        ik_return = self.robot_client.inverse_kin(tpos=tpos, init_jpos=init_jpos)
        if ik_return["response"]["code"] == "0":
            ik_return["success"] = True
        else:
            ik_return["jpos"] = self.get_state()["q"]  # Just return current q
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

    def move(self, target_pos: Dict[int, List[float]], mode: str = "joint_abs", wait=False, tele_mode=False, **kwargs):
        for robot_id, pos in target_pos.items():
            self.robots[robot_id].move(
                target_pos=pos,
                mode=mode,
                wait=wait,
                tele_mode=tele_mode,
                **kwargs
            )

    def tele_move(self, action: Dict[int, List[float]], mode: str = "joint_abs", **kwargs):
        for robot_id, pos in action.items():
            self.robots[robot_id].tele_move(
                action=pos,
                mode=mode,
                **kwargs
            )
