# base
from __future__ import annotations
from typing import Dict, List, Union
import time

# helper functions
from helper.extra_utils import ROBOT_STATE

# communication
from communication.robot import Robot, RobotCluster

# config
from helper.extra_utils import NN_CONTROL_STATE
from helper.config_utils import ROBOT_CONFIG, TASK_CONFIG, BASE_ROBOT_CONFIG, BASE_TASK_CONFIG


class Controller:
    robot_config = BASE_ROBOT_CONFIG
    robot_ids = BASE_ROBOT_CONFIG.robot_ids
    task_config = BASE_TASK_CONFIG
    task_name = BASE_TASK_CONFIG.name
        
    def __init__(self, robot: Dict[int, Robot] | None = None, **kwargs):
        # set robot
        self.robot: Dict[int, Robot] = dict()
        if robot is not None:
            self.robot = robot
            for robot_id in self.robot_ids:
                assert robot_id in self.robot.keys(), f"Robot id {robot_id} does not exist"
                assert isinstance(self.robot[robot_id], Robot), f"Wrong robot instance for id {robot_id}"
        else:
            for robot_id in self.robot_ids:
                self.robot[robot_id] = Robot(
                    robot_ip=self.robot_config.robot_params[robot_id]["ip"], 
                    gripper_config=self.robot_config.robot_params[robot_id].get("gripper", None),
                    **self.robot_config.robot_params[robot_id].get("init_kwargs", {})
                )
                
        self.robot_cluster = RobotCluster(robots=self.robot)
        self.exec_set_idle()
        
    def exec_home_movement(self, wait: bool = False):
        #############################################################
        # Define process to run for home pos execution in config.py #
        #############################################################
        self.task_config.extra_config.home_movement_fn(self, wait)
        
    def exec_start_movement(self):
        #######################################################################
        # Define process to run BEFORE main controller execution in config.py #
        #######################################################################
        self.task_config.extra_config.start_movement_fn(self)
        
    def exec_finish_movement(self):
        ######################################################################
        # Define process to run AFTER main controller execution in config.py #
        ######################################################################
        self.task_config.extra_config.finish_movement_fn(self)

    def exec_home_pos(self, wait=False):
        for robot_id in self.robot_ids:
            self.set_idle(robot_id)
            time.sleep(0.1)
        
        target_pos_dict = dict()
        for robot_id in self.robot_ids:
            if self.robot_config.robot_params[robot_id]["home_pos"] is not None:
                target_pos_dict[robot_id] = self.robot_config.robot_params[robot_id]["home_pos"]

        self.robot_cluster.move(
            target_pos=target_pos_dict,
            mode="joint_abs",
            wait=wait,
            vel_ratio={robot_id: self.robot_config.robot_params[robot_id]["control"]["move_vel_scale"] for robot_id in self.robot_config.robot_ids},
            acc_ratio={robot_id: self.robot_config.robot_params[robot_id]["control"]["move_acc_scale"] for robot_id in self.robot_config.robot_ids},
        )

    def exec_set_teleop(self):
        for robot_id in self.robot_ids:
            self.set_teleop(robot_id)

    def exec_set_idle(self):
        for robot_id in self.robot_ids:
            self.set_idle(robot_id)

    def exec_direct_teaching(self, enable: bool):
        for robot_id in self.robot_ids:
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
        for robot_id in self.robot_ids:
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

    def exec_soft_stop(self, 
                       last_action: Dict[int, List[float]], 
                       control_period: float, 
                       mode: str = "task_abs"):
        assert mode in ["joint_abs", "task_abs"], f"Unavailable control mode {mode}"

        soft_stop_start = time.time()

        while time.time() - soft_stop_start < 0.2:
            soft_stop_control_start = time.time()

            self.robot_cluster.tele_move(
                action=last_action,
                mode=mode,
                vel_scale={robot_id: self.robot_config.robot_params[robot_id]["control"]["vel_scale"] for robot_id in self.robot_config.robot_ids},
                acc_scale={robot_id: self.robot_config.robot_params[robot_id]["control"]["acc_scale"] for robot_id in self.robot_config.robot_ids},
            )

            soft_stop_control_end = time.time()
            wait_time = control_period - (soft_stop_control_end - soft_stop_control_start)
            if wait_time > 0.:
                time.sleep(wait_time)

    def set_teleop(self, robot_id: int, mode: str = "task_abs"):
        assert mode in ["joint_abs", "task_abs"], f"Unavailable control mode {mode}"
        assert robot_id in self.robot_ids, f"Unavailable robot ID {robot_id}"

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

    def set_no_teleop(self, robot_id: int):
        assert robot_id in self.robot_ids, f"Unavailable robot ID {robot_id}"

        time_limit = 10.  # sec
        start_time = time.time()
        
        while time.time() - start_time < time_limit:
            robot_state = self.robot[robot_id].get_state()["op_state"]
            if robot_state == ROBOT_STATE.IDLE:
                break
            else:
                self.robot[robot_id].stop_teleop()
                time.sleep(0.2)

    def set_recovery(self, robot_id: int):
        assert robot_id in self.robot_ids, f"Unavailable robot ID {robot_id}"

        while True:
            op_state = self.robot[robot_id].get_state()["op_state"]
            if ROBOT_STATE.in_failure_state(robot_state=op_state):
                self.robot[robot_id].recover()
                time.sleep(3.)
            else:
                break
    
    def set_idle(self, robot_id: int):
        assert robot_id in self.robot_ids, f"Unavailable robot ID {robot_id}"
        
        robot_state = self.robot[robot_id].get_state()["op_state"]
        if robot_state == ROBOT_STATE.TELE_OP:
            self.set_no_teleop(robot_id)
        elif robot_state == ROBOT_STATE.DIRECT_TEACHING:
            self.robot[robot_id].set_direct_teaching(enable=False)
            time.sleep(0.1)
        elif ROBOT_STATE.in_failure_state(robot_state=robot_state):
            self.set_recovery(robot_id)
            time.sleep(0.1)
            

class Base_NN_controller(Controller):
    def __init__(self, robot: Dict[int, Robot] | None = None, **kwargs):
        # set robot
        super(Base_NN_controller, self).__init__(robot=robot, **kwargs)
        
        # set empty camera
        self.camera = {}
        
        # set empty nn policy
        self.nn_policy: Union[Empty_NN_policy] = Empty_NN_policy(robot_config=self.robot_config, task_config=self.task_config)
        
    def load_policy(self):
        raise NotImplementedError
    
    def exec_nn_control(self, duration: float):
        raise NotImplementedError
    
    def exec_nn_control_stop(self):
        raise NotImplementedError
    
    def exec_start_movement(self):
        #######################################################################
        # Define process to run BEFORE main controller execution in config.py #
        #######################################################################
        self.task_config.extra_config.start_movement_fn(self)
        return NN_CONTROL_STATE.TASK_IN_PROGRESS
        
    def exec_finish_movement(self):
        ######################################################################
        # Define process to run AFTER main controller execution in config.py #
        ######################################################################
        self.task_config.extra_config.finish_movement_fn(self)
        if self.control_state == NN_CONTROL_STATE.TASK_FINISH:
            return NN_CONTROL_STATE.TASK_SUCCESS
        else:
            return NN_CONTROL_STATE.TASK_FAIL
    
    def _reset_control(self):
        self.nn_policy.reset()
        self.control_state = NN_CONTROL_STATE.TASK_IN_PROGRESS


class Empty_NN_policy:
    """
    Policy without loaded weights
    """
    def __init__(self, robot_config: ROBOT_CONFIG, task_config: TASK_CONFIG):
        self.robot_config = robot_config
        self.task_config = task_config
        self.n_robots = len(self.robot_config.robot_ids)
        self.device = self.task_config.model_config.device

    def reset(self):
        pass

    def __call__(self, **kwargs):
        pass
