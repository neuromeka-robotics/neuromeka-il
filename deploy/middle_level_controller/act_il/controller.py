# base
from typing import Dict, Union
from threading import Thread
import time
import numpy as np
import torch

# helper functions
from helper.extra_utils import ROBOT_STATE, NN_CONTROL_STATE
from helper.controller_utils import Base_NN_controller, Empty_NN_policy

# communication
from communication.robot import Robot
from perception_module.realsense import RealsenseCamHandler

# nn models
from middle_level_controller.act_il.model import *

# config
from middle_level_controller.act_il.config import CUSTOM_ROBOT_CONFIG, CUSTOM_TASK_CONFIG


class NN_controller(Base_NN_controller):
    ROBOT_CONFIG = CUSTOM_ROBOT_CONFIG
    ROBOT_IDS = CUSTOM_ROBOT_CONFIG.robot_ids
    TASK_CONFIG = CUSTOM_TASK_CONFIG
    TASK_NAME = CUSTOM_TASK_CONFIG.name
    
    def __init__(self, robot: Dict[int, Robot] | None = None, **kwargs):
        super(NN_controller, self).__init__(robot=robot, **kwargs)

        # set camera
        self.camera: Dict[str, RealsenseCamHandler] = kwargs.get("camera", dict())
        for cam_name in self.TASK_CONFIG.camera_config.cam_names:
            if cam_name in self.camera.keys():
                if not self.camera[cam_name]._thread_running:
                    self.camera[cam_name].start()
            else:
                cam_config = self.TASK_CONFIG.camera_config.cam_params[cam_name]
                
                self.camera[cam_name] = RealsenseCamHandler(
                    serial_number=cam_config["serial"], 
                    align=True, 
                    clipping_distance_m=1., 
                    exposure=cam_config.get("exposure", None)
                )
                self.camera[cam_name].start()
                
        # set empty variables
        self._control_triggered = False
        self._control_thread = None

    def load_policy(self):
        if isinstance(self.nn_policy, Empty_NN_policy):
            self.nn_policy: NN_policy = NN_policy(self.TASK_NAME)
        
    def exec_nn_control(self, duration: float):
        if not self._control_triggered:
            self._reset_control()
            
            self._control_triggered = True
            
            self._control_thread = Thread(target=self._nn_control_fn, args=(duration,), daemon=True)
            self._control_thread.start()
        else:
            print("Control loop is already triggered")
            
    def exec_nn_control_stop(self):
        self._control_triggered = False
        if self._control_thread is not None:
            self._control_thread.join()
            self._control_thread = None
            
    def _reset_control(self):
        self.nn_policy.reset()

    def exec_start_movement(self):
        for robot_id in self.ROBOT_IDS:
            self.set_teleop(robot_id, mode="task_abs")

    def exec_finish_movement(self):
        for robot_id in self.ROBOT_IDS:
            self.set_idle(robot_id)

    def _nn_control_fn(self, duration: float):
        # check task-specific home position for ALL robots!
        states = self.robot_cluster.get_state(robot_ids=self.ROBOT_IDS)
        for robot_id in self.ROBOT_IDS:
            home_joint_pos = self.ROBOT_CONFIG.robot_params[robot_id]["home_pos"]
            
            if home_joint_pos is not None:
                home_joint_pos = np.array(home_joint_pos)
                current_joint_pos = np.array(states[robot_id]["q"])
                error = np.linalg.norm(home_joint_pos - current_joint_pos)

                if error > 0.5:
                    print(f"Move robot {robot_id} to task-specific home position. (error: {error})")
                    self._control_triggered = False
                    return
        
        # pre execution
        self.exec_start_movement()

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

            for robot_id in self.ROBOT_IDS:
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

            # get exteroception
            cam_data_dict  = dict()
            for cam_name in self.TASK_CONFIG.camera_config.cam_names:
                cam_output = self.camera[cam_name].get_all()
                cam_data_dict[f"images.rgb.{cam_name}"] = cam_output["rgb"]  # (H, W, C)
                if self.TASK_CONFIG.camera_config.cam_params[cam_name].get("enable_depth", False):
                    cam_data_dict[f"images.depth.{cam_name}"] = cam_output["depth"]  # (H, W)
                cam_data_dict[f"images.intrinsics.{cam_name}"] = cam_output["intrinsics"]  # (3, 3)

            # prepare input for nn policy
            policy_input = {
                "qpos": qpos,
                "qvel": qvel,
                "end_pos": end_pos,
                "end_vel": end_vel,
                **cam_data_dict
            }

            # forward pass neural network
            with torch.no_grad():
                nn_control = self.nn_policy(**policy_input)

            nn_robot_action = dict()
            for idx, robot_id in enumerate(self.ROBOT_IDS):
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
                    vel_scale={robot_id: self.ROBOT_CONFIG.robot_params[robot_id]["control"]["vel_scale"] for robot_id in self.ROBOT_CONFIG.robot_ids},
                    acc_scale={robot_id: self.ROBOT_CONFIG.robot_params[robot_id]["control"]["acc_scale"] for robot_id in self.ROBOT_CONFIG.robot_ids},
                )

            # sync control frequency                         
            control_end = time.time()
            wait_time = self.ROBOT_CONFIG.control_dt - (control_end - control_start)
            if wait_time > 0:
                time.sleep(wait_time)
        
        # soft stop
        self.exec_soft_stop(
            task_robot_ids=self.ROBOT_IDS,
            last_action=nn_robot_action,
            control_period=self.ROBOT_CONFIG.control_dt,
            mode="task_abs")

        # post execution
        self.exec_finish_movement()
        self._control_triggered = False
