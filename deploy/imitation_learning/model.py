from typing import Dict

import pickle
from ruamel.yaml import YAML
from enum import Enum
import os
import re
import sys
import numpy as np
import torch
import torchvision.transforms as transforms

from config.model import POLICY

from helper.math_utils import MathFunc
from helper.extra_utils import NN_CONTROL_STATE

match = re.search(r'(.*/neuromeka-il/)', os.path.abspath(__file__))
sys.path.append(os.path.join(match.group(1), "train"))
from policies import ACTConfig, ACTPolicy
from policies.selection import ControlMode

class Empty_NN_policy:
    """
    Policy without loaded weights
    """

    def __init__(self, task):
        self.task = task
        self.n_robots = 0

    def reset(self):
        pass

    def __call__(self, **kwargs):
        pass


class NN_policy:
    """
    Policy with loaded weights
    """

    def __init__(self, task_name, device='cuda'):
        self.task_name = task_name

        # load policy config
        cfg: Dict[str, Dict] = YAML().load(
            open(
                f'{POLICY[task_name]["model"]["directory"]}/{POLICY[task_name]["model"]["configuration"]}',
                "r",
            )
        )
        if POLICY[task_name]["model"]["type"] == "act":
            policy_config = ACTConfig(**cfg["policy"])
        else:
            raise NotImplementedError

        # set crucial parameters
        self.n_robots = policy_config.num_robots
        self.n_obs_steps = policy_config.n_obs_steps
        self.action_update_count = policy_config.n_action_steps
        self.control_mode = policy_config.control_mode

        # load dataset statistics
        with open(
            f'{POLICY[task_name]["model"]["directory"]}/{POLICY[task_name]["model"]["datastats"]}',
            "rb",
        ) as f:
            policy_stats = pickle.load(f)

        for k in policy_stats.keys():
            policy_stats[k]["mean"] = torch.from_numpy(policy_stats[k]["mean"])
            policy_stats[k]["std"] = torch.from_numpy(policy_stats[k]["std"])

        # set image pre-processor
        camera_names = []
        candidate_words = ["observation.images", "observation.depths"]
        for k in policy_config.input_shapes:
            for candidate_word in candidate_words:
                if k.startswith(candidate_word):
                    camera_names.append(k.split(".")[-1])
                    image_size = policy_config.input_shapes[
                        f"{candidate_word}.{camera_names[0]}"
                    ][
                        -2:
                    ]  # Assume same size for all images

        self.image_resize = transforms.Resize(image_size)
        if cfg.get("extra") is not None and cfg["extra"].get("image_crop") is not None:
            assert len(cfg["extra"]["image_crop"]) == 1, "Single camera only supported."
            for image_name, crop_area in cfg["extra"]["image_crop"].items():
                self.image_crop = lambda img: img[
                    :,
                    crop_area[0][0] : crop_area[0][1],
                    crop_area[1][0] : crop_area[1][1],
                ]
        else:
            self.image_crop = None

        # load policy
        if POLICY[task_name]["model"]["type"] == "act":
            self.policy = ACTPolicy(policy_config, policy_stats)
        else:
            raise NotImplementedError

        self.policy.load_state_dict(
            torch.load(
                f'{POLICY[task_name]["model"]["directory"]}/{POLICY[task_name]["model"]["weight"]}',
                weights_only=True,
            )
        )
        self.policy.to(device)
        self.policy.eval()
        print(
            f'Loaded (policy): {POLICY[task_name]["model"]["directory"]}/{POLICY[task_name]["model"]["weight"]}'
        )

    def reset(self):
        self.policy.reset()

        # Used for relaitve transformation
        self.init_relative_end_pos = None
        self.init_relative_end_ori = None

    def __call__(self, **args):
        # pre-process input data
        qpos = args["qpos"].astype(np.float32)  # (n_joints * n_robots)
        qvel = args["qvel"].astype(np.float32)  # (n_joints * n_robots)
        end_pose = args["end_pos"].astype(np.float32)  # (6 * n_robots)
        color_image = args["color_image"].astype(np.float32)  # (N_CAM, H, W, C)

        qpos = MathFunc.degree_to_rad(qpos)
        qvel = MathFunc.degree_to_rad(qvel)
        end_pos = MathFunc.mm_to_m(end_pose[:3])
        end_ori = MathFunc.degree_to_rad(end_pose[3:])
        end_ori = MathFunc.euler_to_rotMat(
            euler_x=end_ori[0], euler_y=end_ori[1], euler_z=end_ori[2]
        )
        qpos_data = torch.from_numpy(qpos).unsqueeze(0)  # (B, 6 * n_robots)
        qvel_data = torch.from_numpy(qvel).unsqueeze(0)  # (B, 6 * n_robots)

        if self.image_crop is not None:
            color_image = self.image_crop(color_image)
        image_data = torch.from_numpy(color_image)
        image_data = torch.einsum("k h w c -> k c h w", image_data)
        image_data = self.image_resize(image_data)
        image_data /= 255.0
        image_data = image_data.unsqueeze(0)  # (B, N_CAM, C, H, W)
        image_data = image_data[:, 0]  # TODO: Currently, HARDCODE for single camera  # (B, C, H, W)

        # prepare available data
        available_data = {
            "observation.images.top": image_data,
            "observation.qpos": qpos_data,
            "observation.qvel": qvel_data,
        }

        # select data from the available data that the policy requires
        batch_data_keys = list(self.policy.config.input_shapes.keys())
        batch_data = dict()
        for key in batch_data_keys:
            batch_data[key] = available_data[key].cuda().contiguous()

        if len(self.policy._action_queue) == 0:
            self.init_relative_end_pos = end_pos
            self.init_relative_end_ori = end_ori

        # forward pass neural network
        output = self.policy.select_action(batch_data)  # (B, 12 * n_robots) or (B, (12 + 1) * n_robots)
        action = output["actions"].squeeze(0).cpu().numpy()
        success_prob = output["success_probability"].squeeze(0).cpu().numpy().item()

        # post-process output data
        action_dict = dict()
        action_dict["action"] = action

        if success_prob > POLICY[self.task_name]["deploy"].get("success_threshold", 1.):
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

        return action_dict
