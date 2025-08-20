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

from nrmk_il.policies import ACTConfig, ACTPolicy
from nrmk_il.policies.selection import ControlMode, GripperMode


class NN_policy(Empty_NN_policy):
    """
    Policy with loaded weights
    """
    def __init__(self, robot_config: ROBOT_CONFIG, task_config: TASK_CONFIG):
        super(NN_policy, self).__init__(robot_config=robot_config, task_config=task_config)

        # load policy config
        cfg: Dict[str, Dict] = YAML().load(
            open(
                os.path.join(self.task_config.model_config.model_dir, "config.yaml"),
                "r",
            )
        )
        
        if self.task_config.model_config.model_type == "act":
            policy_config = ACTConfig(**cfg["policy"])
        else:
            raise NotImplementedError

        # set parameters
        assert self.n_robots == policy_config.num_robots, "Number of robots mismatch"
        self.n_obs_steps = policy_config.n_obs_steps
        self.action_update_count = policy_config.n_action_steps
        self.control_mode = policy_config.control_mode
        self.gripper_mode = GripperMode.robot_mode_to_gripper_mode(policy_config.robot_mode)
        if self.gripper_mode in [GripperMode.BINARY, GripperMode.CONTINUOUS]:
            self.use_gripper = True
        else:
            self.use_gripper = False

        # load dataset statistics
        with open(
            os.path.join(self.task_config.model_config.model_dir, "dataset_stats.pkl"),
            "rb",
        ) as f:
            policy_stats = pickle.load(f)

        for k in policy_stats.keys():
            policy_stats[k]["mean"] = torch.from_numpy(policy_stats[k]["mean"])
            policy_stats[k]["std"] = torch.from_numpy(policy_stats[k]["std"])

        # set image pre-processor
        candidate_word = "observation.images"
        for key, value in policy_config.input_shapes.items():
            if key.startswith(candidate_word):
                image_size = value[-2:]  # Assume same size for all images
        self.image_resize = transforms.Resize(image_size)
        
        self.image_crop = dict()
        if cfg.get("extra") is not None and cfg["extra"].get("image_crop") is not None:
            for image_name, crop_area in cfg["extra"]["image_crop"].items():
                self.image_crop[image_name] = \
                    lambda img: img[crop_area[0][0]:crop_area[0][1], crop_area[1][0]:crop_area[1][1]]

        # load policy
        if self.task_config.model_config.model_type == "act":
            self.policy = ACTPolicy(policy_config, policy_stats)
        else:
            raise NotImplementedError

        policy_weight_path = os.path.join(self.task_config.model_config.model_dir, self.task_config.model_config.model_file)
        self.policy.load_state_dict(
            torch.load(
                policy_weight_path,
                weights_only=True,
            )
        )
        self.policy.to(self.device)
        self.policy.eval()
        print(
            f"Loaded (policy): {policy_weight_path}"
        )

    def reset(self):
        self.policy.reset()

        # Used for relaitve transformation
        self.init_relative_end_pos = None
        self.init_relative_end_ori = None
        
        # Gripper
        if self.use_gripper:
            self.prev_gripper_cmd = torch.ones(self.n_robots).unsqueeze(0)  # (B, n_robots)

    def __call__(self, **args):
        # pre-process input data
        qpos = args["qpos"].astype(np.float32)  # (n_joints * n_robots)
        qvel = args["qvel"].astype(np.float32)  # (n_joints * n_robots)
        end_pose = args["end_pos"].astype(np.float32)  # (6 * n_robots)
        end_vel = args["end_vel"].astype(np.float32)  # (6 * n_robots)
        
        cam_data_dict = dict()
        for cam_name in self.task_config.camera_config.cam_names:
            cam_data_dict[f"observation.images.rgb.{cam_name}"] = args[f"images.rgb.{cam_name}"].astype(np.float32)  # (H, W, C)
            if args.get(f"images.depth.{cam_name}") is not None:
                cam_data_dict[f"observation.images.depth.{cam_name}"] = args[f"images.depth.{cam_name}"][..., np.newaxis]  # (H, W, 1)
                
        if self.use_gripper:  ###################
            gripper_pos = args["gripper_pos"].astype(np.float32)  # (n_robots)
            grasp_state = args["grasp_state"].astype(np.float32)  # (n_robots)
            gripper_pos_data = torch.from_numpy(gripper_pos).unsqueeze(0)  # (B, n_robots)
            grasp_state_data = torch.from_numpy(grasp_state).unsqueeze(0)  # (B, n_robots)
                
        # Image crop
        for key in cam_data_dict.keys():
            if self.image_crop.get(key) is not None:
                cam_data_dict[key] = self.image_crop[key](cam_data_dict[key])

        # Change unit
        qpos = MathFunc.degree_to_rad(qpos)
        qvel = MathFunc.degree_to_rad(qvel)
        end_pos = MathFunc.mm_to_m(end_pose[:3])
        end_ori = MathFunc.degree_to_rad(end_pose[3:])
        end_ori = MathFunc.euler_to_rotMat(
            euler_x=end_ori[0], euler_y=end_ori[1], euler_z=end_ori[2]
        )
        end_linVel = MathFunc.mm_to_m(end_vel[:3])
        end_angVel = MathFunc.degree_to_rad(end_vel[3:])
        
        # Pre-process
        for key in cam_data_dict.keys():
            image_data = torch.from_numpy(cam_data_dict[key])
            image_data = torch.einsum('h w c -> c h w', image_data)
            image_data = self.image_resize(image_data)
            image_data /= 255.
            cam_data_dict[key] = image_data.unsqueeze(0)  # (B, C, H, W)

        # prepare available data
        available_data = {
            "observation.qpos": torch.from_numpy(qpos).unsqueeze(0),
            "observation.qvel": torch.from_numpy(qvel).unsqueeze(0),
            "observation.end_position": torch.from_numpy(end_pos).unsqueeze(0),
            "observation.end_orientation": torch.from_numpy(end_ori.reshape(-1)).unsqueeze(0),  # (B, 9)
            "observation.end_linear_velocity": torch.from_numpy(end_linVel).unsqueeze(0),
            "observation.end_angular_velocity": torch.from_numpy(end_angVel).unsqueeze(0),
            **cam_data_dict
        }
        if self.use_gripper:
            available_data["observation.gripper_position"] = gripper_pos_data
            available_data["observation.grasp_state"] = grasp_state_data
            available_data["observation.prev_gripper_command"] = self.prev_gripper_cmd

        # select data from the available data that the policy requires
        batch_data_keys = list(self.policy.config.input_shapes.keys())
        batch_data = dict()
        for key in batch_data_keys:
            batch_data[key] = available_data[key].to(self.device).contiguous()

        if len(self.policy._action_queue) == 0:
            self.init_relative_end_pos = end_pos
            self.init_relative_end_ori = end_ori

        # forward pass neural network
        output = self.policy.select_action(batch_data)  # (B, 12 * n_robots) or (B, (12 + 1) * n_robots)
        action = output["actions"].squeeze(0).cpu().numpy()
        if output["success_probability"] is None:
            success_prob = 0.  # No success detector
        else:
            success_prob = output["success_probability"].squeeze(0).cpu().numpy().item()

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
                self.prev_gripper_cmd[:, robot_id] = gripper_cmd

        return action_dict
