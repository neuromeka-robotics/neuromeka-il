# base
from typing import Dict
import pickle
from ruamel.yaml import YAML
from enum import Enum
import os
import re
import numpy as np
import torch
import torchvision.transforms as transforms

# helper functions
from helper.config_utils import ROBOT_CONFIG, TASK_CONFIG
from helper.controller_utils import Empty_NN_policy
from helper.math_utils import MathFunc
from helper.extra_utils import NN_CONTROL_STATE
from nrmk_il.policies.selection import ControlMode, GripperMode

# remote communication modules for external nn models
# No nn model dependencies are required in "model.py"
from helper.remote_utils import PickleClient


class NN_policy(Empty_NN_policy):
    """
    Policy with loaded weights
    """
    def __init__(self, robot_config: ROBOT_CONFIG, task_config: TASK_CONFIG):
        super(NN_policy, self).__init__(robot_config=robot_config, task_config=task_config)
        
        # Set client to communicate with the nn server
        self.client = PickleClient(host="localhost", port=self.task_config.model_config.port)
        
        # Request nn model loading
        response = self.client.send_data(
            data={
                "operation": "load",
                "data": {
                    "model_config": {
                        "model_type": self.task_config.model_config.model_type,
                        "model_dir": self.task_config.model_config.model_dir,
                        "model_file": self.task_config.model_config.model_file,
                        "device": self.task_config.model_config.device
                    }
                }
            }
        )
        assert response["result"] == "SUCCESS", "NN Server-Client communication failed"
        
        self.n_robots = response["num_robots"]
        self.control_mode = ControlMode.name_to_mode(name=response["control_mode"])
        self.gripper_mode = GripperMode.name_to_mode(name=response["gripper_mode"])
        if self.gripper_mode in [GripperMode.BINARY, GripperMode.CONTINUOUS]:
            self.use_gripper = True
        else:
            self.use_gripper = False

    def reset(self):
        response = self.client.send_data(data={"operation": "reset"})
        assert response["result"] == "SUCCESS", "NN Server-Client communication failed"

        # Used for relaitve transformation
        self.init_relative_end_pos = None
        self.init_relative_end_ori = None
        
        # Gripper
        if self.use_gripper:
            self.prev_gripper_cmd = np.ones(self.n_robots)

    def __call__(self, **kwargs):
        # Prepare data
        data = {
            "proprioception": {
                "qpos": kwargs["qpos"],
                "qvel": kwargs["qvel"],
                "end_pose": kwargs["end_pose"],
                "end_vel": kwargs["end_vel"]
            },
            "images": {}
        }
        
        for cam_name in self.task_config.camera_config.cam_names:
            data["images"][cam_name] = {}
            data["images"][cam_name]["rgb"] = kwargs[f"images.rgb.{cam_name}"]
            if kwargs.get(f"images.depth.{cam_name}") is not None:
                data["images"][cam_name]["depth"] = kwargs[f"images.depth.{cam_name}"]
                
        if self.use_gripper:
            data["proprioception"]["gripper_pos"] = kwargs["gripper_pos"]
            data["proprioception"]["grasp_state"] = kwargs["grasp_state"]
            data["proprioception"]["prev_gripper_command"] = self.prev_gripper_cmd
        
        # Request nn model control
        response = self.client.send_data(data={"operation": "step", "data": data})
        assert response["result"] == "SUCCESS", "NN Server-Client communication failed"
        for key in ["action", "success_probability"]:
            assert key in response.keys(), f"{key} key is missing"
        assert "is_new_chunk" in response.keys(), "is_new_chunk key required for action chunking models"
        
        if response["is_new_chunk"]:
            end_pos = MathFunc.mm_to_m(kwargs["end_pose"][:3])
            end_ori = MathFunc.degree_to_rad(kwargs["end_pose"][3:])
            end_ori = MathFunc.euler_to_rotMat(
                euler_x=end_ori[0], euler_y=end_ori[1], euler_z=end_ori[2]
            )
            self.init_relative_end_pos = end_pos
            self.init_relative_end_ori = end_ori

        action = response["action"]
        success_prob = response["success_probability"]

        # post-process
        action_dict = dict()
        action_dict["action"] = action

        if success_prob > self.task_config.model_config.success_threshold:
            action_dict["control_state"] = NN_CONTROL_STATE.TASK_FINISH
        else:
            action_dict["control_state"] = NN_CONTROL_STATE.TASK_IN_PROGRESS

        for robot_id in range(self.n_robots):
            pos_action = action[3 * robot_id : 3 * (robot_id + 1)]
            rot_action = action[
                3 * self.n_robots + 9 * robot_id : 
                3 * self.n_robots + 9 * (robot_id + 1)
            ].reshape(3, 3)

            if self.control_mode == ControlMode.RELATIVE_DELTA_TASK_SPACE:
                pos_action = self.init_relative_end_ori @ pos_action + self.init_relative_end_pos
                rot_action = self.init_relative_end_ori @ rot_action

            pos_action = MathFunc.m_to_mm(pos_action)
            euler_action = MathFunc.rotMat_to_euler(rot_action)
            euler_action = MathFunc.rad_to_degree(euler_action)
            action_dict[f"robot_action_{robot_id}"] = np.concatenate(
                (pos_action, euler_action), axis=-1
            )  # (6,)
            
        if self.use_gripper:
            for robot_id in range(self.n_robots):
                gripper_cmd = action[-(self.n_robots - robot_id)]
                gripper_cmd = np.clip(gripper_cmd, a_min=0., a_max=1.)
                
                if self.gripper_mode == GripperMode.BINARY:
                    gripper_cmd = np.round(gripper_cmd)  # 0 or 1

                action_dict[f"gripper_action_{robot_id}"] = gripper_cmd
                self.prev_gripper_cmd[robot_id] = gripper_cmd

        return action_dict
