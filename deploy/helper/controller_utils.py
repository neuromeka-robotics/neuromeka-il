import time
from typing import Dict, List

# help functions
from helper.extra_utils import ROBOT_STATE

# config
from config.robot import ROBOT_HOME_POS, CONTROL, ROBOT_ID

# communication
from communication.robot import Robot

class Controller:
    def __init__(self, **kwargs):
        # set robot connection
        self.robot: Dict[int, Robot] = dict()
        # for robot_id in ROBOT_ID:
        #     self.robot[robot_id] = Robot(robot_id)

    def exec_home_pos(self, task, wait=False, tele_mode=False, mode="joint_abs", **kwargs):
        assert task in ROBOT_HOME_POS.keys(), f"Robot home position for '{task}' not set."
        assert mode in ["joint_abs", "joint_abs_w_safety"]

        for robot_id in ROBOT_HOME_POS[task].keys():
            if tele_mode:
                self.set_teleop(robot_id, mode="joint_abs")
            else:
                self.set_idle(robot_id)
            time.sleep(0.1)

            self.robot[robot_id].move(
                target_pos=ROBOT_HOME_POS[task][robot_id],
                mode=mode,
                wait=wait,
                tele_mode=tele_mode,
                **kwargs
            )

    def exec_set_teleop(self):
        for robot_id in ROBOT_ID:
            self.set_teleop(robot_id)

    def exec_set_idle(self):
        for robot_id in ROBOT_ID:
            self.set_idle(robot_id)

    def exec_direct_teaching(self, enable: bool):
        for robot_id in ROBOT_ID:
            robot_state = self.robot[robot_id].get_state()["op_state"]
            if enable:
                if robot_state != ROBOT_STATE.DIRECT_TEACHING:
                    if robot_state == ROBOT_STATE.TELE_OP:
                        self.set_no_teleop(robot_id)
                        time.sleep(0.1)
                        self.robot[robot_id].set_direct_teaching(enable=True)
                    elif ROBOT_STATE.in_failure_state(robot_state=robot_state):
                        self.set_recovery(robot_id)
                        time.sleep(0.1)
                        self.robot[robot_id].set_direct_teaching(enable=True)
                    elif robot_state == ROBOT_STATE.IDLE:
                        self.robot[robot_id].set_direct_teaching(enable=True)
            else:
                if robot_state == ROBOT_STATE.DIRECT_TEACHING:
                    self.robot[robot_id].set_direct_teaching(enable=False)

    def exec_emergency_stop(self, enable: bool):
        for robot_id in ROBOT_ID:
            robot_state = self.robot[robot_id].get_state()["op_state"]
            if enable:
                if robot_state != ROBOT_STATE.STOP:
                    if robot_state == ROBOT_STATE.TELE_OP:  # nn controller running
                        self.set_no_teleop(robot_id)
                        time.sleep(0.1)
                    self.robot[robot_id].stop_motion()
            else:
                if robot_state == ROBOT_STATE.STOP:
                    self.robot[robot_id].recover()
                    time.sleep(3.)

    def exec_soft_stop(self, task_robot_ids: List[int], last_action: Dict[int, List[float]], control_period: float, mode: str):
        assert mode in ["joint_abs", "task_abs", "joint_abs_w_safety", "task_abs_w_safety"], f"Unavailable control mode {mode}"

        soft_stop_start = time.time()

        while time.time() - soft_stop_start < 0.2:
            soft_stop_control_start = time.time()

            for robot_id in task_robot_ids:
                self.robot[robot_id].tele_move(
                    action=last_action[robot_id],
                    mode=mode,
                    vel_scale=CONTROL["vel_scale"], acc_scale=CONTROL["acc_scale"]
                )

            soft_stop_control_end = time.time()
            wait_time = control_period - (soft_stop_control_end - soft_stop_control_start)
            if wait_time > 0.:
                time.sleep(wait_time)

    def set_teleop(self, robot_id, mode="task_abs"):
        assert robot_id in ROBOT_ID, f"Unavailable robot ID {robot_id}"

        time_limit = 10.  # sec
        start_time = time.time()

        while time.time() - start_time < time_limit:
            robot_state = self.robot[robot_id].get_state()["op_state"]
            if robot_state == ROBOT_STATE.TELE_OP:
                break
            else:
                self.robot[robot_id].stop_teleop()
                self.robot[robot_id].start_teleop(mode=mode)
                time.sleep(0.2)

    def set_no_teleop(self, robot_id):
        assert robot_id in ROBOT_ID, f"Unavailable robot ID {robot_id}"

        time_limit = 10.  # sec
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            robot_state = self.robot[robot_id].get_state()["op_state"]
            if robot_state == ROBOT_STATE.IDLE:
                break
            else:
                self.robot[robot_id].stop_teleop()
                time.sleep(0.2)

    def set_recovery(self, robot_id):
        assert robot_id in ROBOT_ID, f"Unavailable robot ID {robot_id}"

        while True:
            op_state = self.robot[robot_id].get_state()["op_state"]
            if ROBOT_STATE.in_failure_state(robot_state=op_state):
                self.robot[robot_id].recover()
                time.sleep(3.)
            else:
                break

    def set_idle(self, robot_id):
        assert robot_id in ROBOT_ID, f"Unavailable robot ID {robot_id}"
        
        robot_state = self.robot[robot_id].get_state()["op_state"]
        if robot_state == ROBOT_STATE.TELE_OP:
            self.set_no_teleop(robot_id)
        elif robot_state == ROBOT_STATE.DIRECT_TEACHING:
            self.robot[robot_id].set_direct_teaching(enable=False)
            time.sleep(0.1)
        elif ROBOT_STATE.in_failure_state(robot_state=robot_state):
            self.set_recovery(robot_id)
            time.sleep(0.1)


