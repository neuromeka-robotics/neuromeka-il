import pickle
from train.policies import ACTConfig, ACTPolicy
from deploy.config.model import POLICY
from ruamel.yaml import YAML


class Empty_NN_policy:
    """
    Policy without loaded weights
    """

    def __init__(self, task):
        self.task = task

    def reset(self):
        pass

    def __call__(self, **kwargs):
        pass


class NN_policy:
    """
    Policy with loaded weights
    """

    def __init__(self, task):
        self.task = task
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
        self.gripper_mode = GripperMode.robot_mode_to_gripper_mode(
            policy_config.robot_mode
        )
        if self.gripper_mode in [GripperMode.BINARY, GripperMode.CONTINUOUS]:
            self.use_gripper = True
        else:
            self.use_gripper = False

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
        if self.use_gripper:
            self.prev_trigger_value = torch.zeros(self.n_robots).unsqueeze(
                0
            )  # (B, n_robots)

        # Used for relaitve transformation
        self.init_end_pos = None
        self.init_end_ori = None
        self.init_relative_end_pos = None
        self.init_relative_end_ori = None

    def __call__(self, **kwargs):
        # pre-process input data
        qpos = args["qpos"].astype(np.float32)  # (n_joints * n_robots)
        qvel = args["qvel"].astype(np.float32)  # (n_joints * n_robots)
        end_pose = args["end_pos"].astype(np.float32)  # (6 * n_robots)
        end_vel = args["end_vel"].astype(np.float32)  # (6 * n_robots)
        color_image = args["color_image"].astype(np.float32)  # (N_CAM, H, W, C)
        depth_image = args["depth_image"].astype(np.float32)  # (N_CAM, H, W, 1)

        qpos = MathFunc.degree_to_rad(qpos)
        qvel = MathFunc.degree_to_rad(qvel)
        end_pos = MathFunc.mm_to_m(end_pose[:3])
        end_ori = MathFunc.degree_to_rad(end_pose[3:])
        end_ori = MathFunc.euler_to_rotMat(
            euler_x=end_ori[0], euler_y=end_ori[1], euler_z=end_ori[2]
        )
        end_linVel = MathFunc.mm_to_m(end_vel[:3])
        end_angVel = MathFunc.degree_to_rad(end_vel[3:])
        qpos_data = torch.from_numpy(qpos).unsqueeze(0)  # (B, 6 * n_robots)
        qvel_data = torch.from_numpy(qvel).unsqueeze(0)  # (B, 6 * n_robots)

        if self.image_crop is not None:
            color_image = self.image_crop(color_image)
        image_data = torch.from_numpy(color_image)
        image_data = torch.einsum("k h w c -> k c h w", image_data)
        image_data = self.image_resize(image_data)
        image_data /= 255.0
        image_data = image_data.unsqueeze(0)  # (B, N_CAM, C, H, W)
        image_data = image_data[
            :, 0
        ]  # TODO: Currently, HARDCODE for single camera  # (B, C, H, W)

        if self.image_crop is not None:
            depth_image = self.image_crop(depth_image)
        depth_data = torch.from_numpy(depth_image)
        depth_data = torch.einsum("k h w c -> k c h w", depth_data)
        depth_data = self.image_resize(depth_data)
        depth_data = depth_data.unsqueeze(0)  # (B, N_CAM, 1, H, W)
        depth_data = depth_data[
            :, 0
        ]  # TODO: Currently, HARDCODE for single camera  # (B, 1, H, W)

        if self.use_gripper:
            gripper_position = (
                args["gripper_position"].astype(np.float32) / 255.0
            )  # (n_robots)
            gripper_object_detected = args["gripper_object_detected"].astype(
                np.float32
            )  # (n_robots)

            gripper_position_data = torch.from_numpy(gripper_position).unsqueeze(
                0
            )  # (B, n_robots)
            gripper_object_detected_data = torch.from_numpy(
                gripper_object_detected
            ).unsqueeze(
                0
            )  # (B, n_robots)

        ##############
        ## Relaitve ##
        rel_end_pos = self.init_end_ori.T @ (end_pos - self.init_end_pos)
        rel_xy_end_pos = end_pos.copy()
        rel_xy_end_pos[:2] = rel_xy_end_pos[:2] - self.init_end_pos[:2]
        rel_end_ori = (self.init_end_ori.T @ end_ori).reshape(-1)
        rel_end_linVel = self.init_end_ori.T @ end_linVel
        rel_end_angVel = self.init_end_ori.T @ end_angVel

        rel_end_pos_data = torch.from_numpy(rel_end_pos).unsqueeze(0)  # (B, 3)
        rel_xy_end_pos_data = torch.from_numpy(rel_xy_end_pos).unsqueeze(0)  # (B, 3)
        rel_end_ori_data = torch.from_numpy(rel_end_ori).unsqueeze(0)  # (B, 9)
        rel_end_linVel_data = torch.from_numpy(rel_end_linVel).unsqueeze(0)  # (B, 3)
        rel_end_angVel_data = torch.from_numpy(rel_end_angVel).unsqueeze(0)  # (B, 3)
        ##############

        # prepare available data
        available_data = {
            "observation.images.top": image_data,
            "observation.qpos": qpos_data,
            "observation.qvel": qvel_data,
        }

        if self.use_gripper:
            available_data["observation.prev_trigger_value"] = self.prev_trigger_value

        # select data from the available data that the policy requires
        batch_data_keys = list(self.policy.config.input_shapes.keys())
        batch_data = dict()
        for key in batch_data_keys:
            batch_data[key] = available_data[key].cuda().contiguous()

        if len(self.policy._action_queue) == 0:
            self.init_relative_end_pos = end_pos
            self.init_relative_end_ori = end_ori

        output = self.policy.select_action(
            batch_data
        )  # (B, 12 * n_robots) or (B, (12 + 1) * n_robots)
        action = output["actions"]
        success_prob = output["success_probability"]

        # post-process
        action = (
            action.squeeze(0).cpu().numpy()
        )  # (12 * n_robots) or (12 + 1) * n_robots)
        action_dict = dict()
        action_dict["action"] = action

        if (success_prob is not None) and (
            POLICY[self.task_name]["deploy"].get("success_threshold") is not None
        ):
            success_prob = success_prob.cpu().numpy().squeeze(0).item()
            if success_prob > POLICY[self.task_name]["deploy"]["success_threshold"]:
                action_dict["control_state"] = NN_CONTROL_STATE.TASK_FINISH
            else:
                action_dict["control_state"] = NN_CONTROL_STATE.TASK_IN_PROGRESS
        else:
            action_dict["control_state"] = NN_CONTROL_STATE.TASK_IN_PROGRESS

        for robot_id in range(self.n_robots):
            pos_action = action[3 * robot_id : 3 * (robot_id + 1)]
            rot_action = action[
                3 * self.n_robots
                + 9 * robot_id : 3 * self.n_robots
                + 9 * (robot_id + 1)
            ].reshape(3, 3)

            if self.control_mode == ControlMode.TASK_SPACE:
                # Apply heuristic (FIX ROLL/PITCH/YAW)
                if POLICY[self.task_name]["deploy"].get("fix"):
                    ori_fix_config: Dict = POLICY[self.task_name]["deploy"].get("fix")
                    if (
                        ori_fix_config.get("roll")
                        and ori_fix_config.get("pitch")
                        and ori_fix_config.get("yaw")
                    ):
                        rot_action = self.init_end_ori
            elif self.control_mode == ControlMode.RELATIVE_TASK_SPACE:
                pos_action = self.init_end_ori @ pos_action + self.init_end_pos
                rot_action = self.init_end_ori @ rot_action

            pos_action = MathFunc.m_to_mm(pos_action)  # (3,)
            euler_action = MathFunc.rotMat_to_euler(rot_action)
            euler_action = MathFunc.rad_to_degree(euler_action)  # (3,)
            action_dict[f"robot_action_{robot_id}"] = np.concatenate(
                (pos_action, euler_action), axis=-1
            )  # (6,)

        if self.use_gripper:
            # TODO: Fix required for {0: no gripper, 1: gripper}
            # Current implementation only works for {0: gripper, 1: no gripper}
            for robot_id in range(self.n_robots):
                trigger_value = action[-(self.n_robots - robot_id)]

                if self.gripper_mode == GripperMode.BINARY:
                    if trigger_value < 0.5:
                        action_dict[f"gripper_action_{robot_id}"] = 3
                        self.prev_trigger_value[:, robot_id] = 0.0
                    else:
                        action_dict[f"gripper_action_{robot_id}"] = 228
                        self.prev_trigger_value[:, robot_id] = 1.0
                elif self.gripper_mode == GripperMode.CONTINUOUS:
                    trigger_value = np.clip(trigger_value, a_min=0.0, a_max=1.0)
                    action_dict[f"gripper_action_{robot_id}"] = int(
                        3 + trigger_value * (228 - 3)
                    )
                    self.prev_trigger_value[:, robot_id] = trigger_value
                else:
                    raise NotImplementedError(
                        f"Not implemented gripper mode ({self.gripper_mode})"
                    )

        return action_dict
