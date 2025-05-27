from typing import List, Dict, Union
from threading import Thread
import time
import numpy as np
import torch

from imitation_learning.model import Empty_NN_policy, NN_policy

from config.robot import CONTROL, ROBOT_HOME_POS
from config.model import POLICY
from config.env import CAMERA

from helper.math_utils import MathFunc
from helper.extra_utils import ROBOT_STATE, NN_CONTROL_STATE
from helper.controller_utils import Controller

from perception_module.realsense import RealsenseCamHandler

from communication.robot import Robot, RobotCluster


class NN_controller(Controller):
    def __init__(self, robot: Dict[int, Robot], tasks: List[str], **kwargs):
        # set robot
        super().__init__(robot=robot)
        
        # set task
        self.nn_tasks = tasks
        
        # set nn controller
        self.nn_policy: Dict[str, Union[Empty_NN_policy, NN_policy]] = dict()
        for task in self.nn_tasks:
            self.nn_policy[task] = Empty_NN_policy(task)

        # set camera
        self.camera: Dict[str, RealsenseCamHandler] = dict()
        for serial in CAMERA["imitation"]:
            if serial not in self.camera.keys():
                self.camera[serial] = RealsenseCamHandler(
                    serial_number=serial, 
                    align=True, 
                    clipping_distance_m=1., 
                    exposure=CAMERA["exposure"][serial]
                )
                self.camera[serial].start()
                print(f"Initialized camera (serial: {serial}, exposure: {CAMERA['exposure'][serial]})")

        # set empty variables
        self._control_triggered = False
        self._control_thread = None
        
    def get_available_tasks(self):
        return self.nn_tasks
        
    def load_policy(self, task: str):
        assert task in self.nn_tasks, f"Unavailable task {task}"
        
        if isinstance(self.nn_policy[task], Empty_NN_policy):
            self.nn_policy[task] = NN_policy(task)
        
    def exec_nn_control(self, task: str, duration: float):
        assert task in self.nn_tasks, f"Unavailable task {task}"
        
        if not self._control_triggered:
            self._reset_control(task)
            
            self._control_triggered = True
            
            self._control_thread = Thread(target=self._nn_control_fn, args=(task, duration,), daemon=True)
            self._control_thread.start()
        else:
            print("Control loop is already triggered")
            
    def exec_nn_control_stop(self):
        self._control_triggered = False
        if self._control_thread is not None:
            self._control_thread.join()
            self._control_thread = None
            
    def _reset_control(self, task: str):
        self.nn_policy[task].reset()

    def exec_start_movement(self, task: str, task_robot_ids: List[int]):
        for robot_id in task_robot_ids:
            self.set_teleop(robot_id, mode="task_abs")

    def exec_finish_movement(self, task: str, task_robot_ids: List[str]):
        for robot_id in task_robot_ids:
            self.set_idle(robot_id)

    def _nn_control_fn(self, task: str, duration: float):
        # initialize
        TASK_ROBOT_ID = ROBOT_HOME_POS[task].keys()
        n_task_robots = self.nn_policy[task].n_robots
        assert len(TASK_ROBOT_ID) == n_task_robots, \
            f"Check robot home pos configuration. Number of robots does not match ({len(TASK_ROBOT_ID)} vs. {n_task_robots})"

        # check task-specific home position for ALL robots!
        states = self.robot_cluster.get_state(robot_ids=TASK_ROBOT_ID)
        for robot_id in TASK_ROBOT_ID:
            home_joint_pos = np.array(ROBOT_HOME_POS[task][robot_id])
            current_joint_pos = np.array(states[robot_id]["q"])
            error = np.linalg.norm(home_joint_pos - current_joint_pos)

            if error > 0.5:
                print(f"Move robot {robot_id} to task-specific home position. (error: {error})")
                self._control_triggered = False
                return
        
        # pre execution
        self.exec_start_movement(task, TASK_ROBOT_ID)

        # start control loop
        control_loop_init = time.time()
        while self._control_triggered and (time.time() - control_loop_init < duration):
            control_start = time.time()
            
            # get proprioception    
            op_state = []
            qpos = None
            qvel = None
            end_pos = None
            end_vel = None

            for robot_id in TASK_ROBOT_ID:
                control_dat = self.robot[robot_id].get_state()
                
                op_state.append(control_dat["op_state"])
                
                if qpos is None:
                    qpos = np.asarray(control_dat["q"])
                else:
                    qpos = np.concatenate((qpos, np.asarray(control_dat["q"])), axis=-1)
                    
                if qvel is None:
                    qvel = np.asarray(control_dat["qdot"])
                else:
                    qvel = np.concatenate((qvel, np.asarray(control_dat["qdot"])), axis=-1)

                if end_pos is None:
                    end_pos = np.asarray(control_dat["p"])
                else:
                    end_pos = np.concatenate((end_pos, np.asarray(control_dat["p"])), axis=-1)
                
                if end_vel is None:
                    end_vel = np.asarray(control_dat["pdot"])
                else:
                    end_vel = np.concatenate((end_vel, np.asarray(control_dat["pdot"])), axis=-1)

            # get exteroception  (TODO: Currently, HARDCODE for single camera)
            cam_serial = POLICY[task]["deploy"]["camera_serial"]
            cam_data = self.camera[cam_serial].get_all()
            color_img = cam_data["rgb"][np.newaxis]
            depth_img = cam_data["depth"][np.newaxis, :, :, np.newaxis]

            # prepare input for nn policy
            policy_input = {
                "qpos": qpos,
                "qvel": qvel,
                "end_pos": end_pos,
                "end_vel": end_vel,
                "color_image": color_img,
                "depth_image": depth_img
            }

            # forward pass neural network
            with torch.no_grad():
                nn_control = self.nn_policy[task](**policy_input)

            nn_robot_action = dict()
            for idx, robot_id in enumerate(TASK_ROBOT_ID):
                nn_robot_action[robot_id] = nn_control[f"robot_action_{idx}"].tolist()

            # update control state
            control_state = nn_control["control_state"]
            for op in op_state:
                if ROBOT_STATE.in_failure_state(robot_state=op):
                    control_state = NN_CONTROL_STATE.ROBOT_FAIL
                    break

            # finish control loop / execute control
            if NN_CONTROL_STATE.get_out_control_loop(control_state):
                break
            else:
                self.robot_cluster.tele_move(
                    action=nn_robot_action,
                    mode="task_abs",
                )

            # sync control frequency                         
            control_end = time.time()
            wait_time = CONTROL["period"] - (control_end - control_start)
            if wait_time > 0:
                time.sleep(wait_time)
        
        # soft stop
        self.exec_soft_stop(
            task_robot_ids=TASK_ROBOT_ID,
            last_action=nn_robot_action,
            control_period=CONTROL["period"],
            mode="task_abs")

        # post execution
        self.exec_finish_movement(task=task, task_robot_ids=TASK_ROBOT_ID)
        self._control_triggered = False
