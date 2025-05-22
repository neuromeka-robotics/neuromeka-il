from typing import List, Dict, Union
from threading import Thread
import time
import numpy as np
import torch

from imitation_learning.model import Empty_NN_policy, NN_policy
from config.robot import CONTROL, ROBOT_HOME_POS
from config.model import POLICY
from helper.math_utils import MathFunc
from helper.extra_utils import ROBOT_STATE, NN_CONTROL_STATE
from perception_module.realsense import RealsenseCamHandler
from config.env import CAMERA
from helper.controller_utils import Controller
from communication.robot import Robot, RobotCluster


class NN_controller(Controller):
    def __init__(self, tasks: List[str], **kwargs):
        self.nn_tasks = tasks
        
        self.nn_policy: Dict[str, Union[Empty_NN_policy, NN_policy]] = dict()
        for task in self.nn_tasks:
            self.nn_policy[task] = Empty_NN_policy(task)
        
        self._control_triggered = False
        self._control_thread = None
        self.camera: Dict[str, RealsenseCamHandler] = dict()
        if isinstance(CAMERA["imitation"], str):
            serial = CAMERA["imitation"]
            self.camera[serial] = RealsenseCamHandler(
                serial_number=serial, 
                align=True, 
                clipping_distance_m=1., 
                exposure=CAMERA["exposure"][serial]
            )
            self.camera[serial].start()
        elif isinstance(CAMERA["imitation"], list):
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
        else:
            raise NotImplementedError

        self.robot: Dict[int, Robot] = kwargs.get("robot", dict())
        #for robot_id in ROBOT_ID:
        #    self.robot[robot_id] = Robot(robot_id, env=None)
        self.robot_cluster = RobotCluster(robots=self.robot)
        
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
        for robot_id in self.robot:
            if (self.robot[robot_id]._exec_tele_traj_thread is not None) and (self.robot[robot_id]._exec_tele_traj_thread.is_alive()):
                self.robot[robot_id]._tele_traj_triggered = False
                self.robot[robot_id]._exec_tele_traj_thread.join()

        self._control_triggered = False
        if self._control_thread is not None:
            self._control_thread.join()
            self._control_thread = None
            
    def _reset_control(self, task: str):
        self.nn_policy[task].reset()

    def exec_finish_movement(self, task: str, task_robot_ids: List[str]):
        for robot_id in task_robot_ids:
            self.set_no_teleop(robot_id)
        return NN_CONTROL_STATE.TASK_SUCCESS

    def exec_start_movement(self, task: str, task_robot_ids: List[int]):
        pass

    def _nn_control_fn(self, task: str, duration: float):
        TASK_ROBOT_ID = ROBOT_HOME_POS[task].keys()
        n_task_robots = self.nn_policy[task].n_robots
        assert len(TASK_ROBOT_ID) == n_task_robots, \
            f"Check robot home pos configuration. Number of robots does not match ({len(TASK_ROBOT_ID)} vs. {n_task_robots})"
        control_loop_init = time.time()

        self.exec_start_movement(task, TASK_ROBOT_ID)

        for robot_id in TASK_ROBOT_ID:
            self.set_teleop(robot_id, mode="task_abs")
        
        while self._control_triggered and (time.time() - control_loop_init < duration):
            control_start = time.time()
            
            #####################################################
            # 1. Get proprioception and exteroception            
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

            if (self.nn_policy[task].init_end_pos is None) \
                or (self.nn_policy[task].init_end_ori is None):
                assert len(TASK_ROBOT_ID) == 1
                init_end_pos = MathFunc.mm_to_m(end_pos[:3])
                init_end_ori = MathFunc.degree_to_rad(end_pos[3:])
                init_end_ori = MathFunc.euler_to_rotMat(
                    euler_x=init_end_ori[0],
                    euler_y=init_end_ori[1],
                    euler_z=init_end_ori[2]
                )
                self.nn_policy[task].init_end_pos = init_end_pos.astype(np.float32)
                self.nn_policy[task].init_end_ori = init_end_ori.astype(np.float32)

            # get exteroception
            cam_serial = POLICY[task]["deploy"].get("camera_serial", None)
            assert isinstance(cam_serial, str), f"Unappropriate camera serial number {cam_serial}"
            cam_data = self.camera[cam_serial].get_all()
            color_img = cam_data["rgb"][np.newaxis]  # TODO: Currently, HARDCODE for single camera
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
            # 2. Forward pass neural network (i.e., NN_policy)  
            with torch.no_grad():
                nn_control = self.nn_policy[task](**policy_input)

            # update control state
            self.control_state = nn_control["control_state"]
            for op in op_state:
                if ROBOT_STATE.in_failure_state(robot_state=op):
                    self.control_state = NN_CONTROL_STATE.ROBOT_FAIL
                    break

            # Finish control loop
            if NN_CONTROL_STATE.get_out_control_loop(self.control_state):
                break

            nn_robot_action = dict()
            for idx, robot_id in enumerate(TASK_ROBOT_ID):
                nn_robot_action[robot_id] = nn_control[f"robot_action_{idx}"].tolist()

            # 3. Execute to robot (i.e., MoveTele)              
            self.robot_cluster.tele_move(
                action=nn_robot_action,
                mode="task_abs",
            )
            print(nn_robot_action)

            # 4. Sync control frequency                         
            
            control_end = time.time()
            wait_time = CONTROL["period"] - (control_end - control_start)
            if wait_time > 0:
                time.sleep(wait_time)
                
        # if self.robot[robot_id].get_state()["op_state"] == ROBOT_STATE.TELE_OP:
        #     self.exec_soft_stop(
        #         task_robot_ids=TASK_ROBOT_ID,
        #         last_action=nn_robot_action,
        #         control_period=control_period,
        #         mode="task_abs")

        if self.control_state == NN_CONTROL_STATE.TASK_FINISH or self.control_state == NN_CONTROL_STATE.TASK_IN_PROGRESS:
            self.control_state = self.exec_finish_movement(task=task, task_robot_ids=TASK_ROBOT_ID)
            print("Executed task finish movement")
        #elif self.control_state == NN_CONTROL_STATE.TASK_IN_PROGRESS:
        #    self.control_state = self.exec_finish_movement(task=task, task_robot_ids=TASK_ROBOT_ID)

        self._control_triggered = False
