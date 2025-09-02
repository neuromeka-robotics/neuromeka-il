import time
import importlib
import numpy as np
import torch

from helper.controller_utils import Empty_NN_policy

"""
Unit test to debug implementation of NN policy
"""

if __name__ == "__main__":
    demo_task = "act_il"
    
    # Import config
    module = importlib.import_module(f"middle_level_controller.{demo_task}.config")
    robot_config = module.CUSTOM_ROBOT_CONFIG
    task_config = module.CUSTOM_TASK_CONFIG
    
    # Import NN policy
    module = importlib.import_module(f"middle_level_controller.{demo_task}.model")
    nn_policy: Empty_NN_policy = module.NN_policy(robot_config=robot_config, task_config=task_config)

    for _ in range(1000):
        # Generate random camera data
        cam_data_dict = dict()
        for cam_name in task_config.camera_config.cam_names:
            cam_data_dict[f"images.rgb.{cam_name}"] = np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
            if task_config.camera_config.cam_params[cam_name].get("enable_depth", False):
                cam_data_dict[f"images.depth.{cam_name}"] = np.random.uniform(size=(480, 640)).astype(np.float32)
        
        # Generate random policy input
        policy_input = {
            "qpos": np.random.normal(size=(6 * len(robot_config.robot_ids),)),
            "qvel": np.random.normal(size=(6 * len(robot_config.robot_ids),)),
            "end_pose": np.random.normal(size=(6 * len(robot_config.robot_ids),)),
            "end_vel": np.random.normal(size=(6 * len(robot_config.robot_ids),)),
            "gripper_pos": np.random.uniform(size=(1 * len(robot_config.robot_ids),)),  # 0 ~ 1
            "grasp_state": np.random.uniform(size=(1 * len(robot_config.robot_ids),)) > 0.5,  # bool
            "prev_gripper_command": np.round(np.random.uniform(size=(1 * len(robot_config.robot_ids),))),  # 0 or 1
            **cam_data_dict
        }
        
        inference_start = time.time()
        
        with torch.no_grad():
            nn_control = nn_policy(**policy_input)
        
        inference_end = time.time()
        print(f"Inference time: {(inference_end - inference_start):.3f}")
        
    time.sleep(0.1)