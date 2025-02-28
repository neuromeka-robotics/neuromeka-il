import argparse
import os
import h5py
import numpy as np
import shutil
from tqdm import tqdm

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from policies.selection import RobotMode
from helper.utils import check_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic data generation")
    parser.add_argument("--task", required=True, type=str, help="Task name to generate")
    parser.add_argument("--robot", required=True, type=str, 
                        choices=["single_robot", "dual_robot", "single_robot_gripper", "dual_robot_gripper"], 
                        help="Robot mode to generate")
    args = parser.parse_args()
    
    task = args.task
    robot_mode = args.robot
    
    # Constant
    traj_len = 300  # Episode length
    n_data = 20  # Number of total episodes
    n_robot_joints = 6  # Number of joints in a single robot
    
    DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", task)
    exist = check_dir(DATA_DIR, generate=False)
    if exist:
        print("Data directory already exists. Automatically removing them.")
        shutil.rmtree(DATA_DIR)
    check_dir(DATA_DIR, generate=True)
    
    robot_mode = RobotMode.name_to_mode(name=robot_mode)
    n_robots = RobotMode.get_num_robots(mode=robot_mode)
    robot_ids = [robot_id for robot_id in range(n_robots)]
    
    for data_id in tqdm(range(n_data)):
        # Empty container
        data = dict()
        
        # Generate proprioception
        for robot_id in robot_ids:
            data[f"q_{robot_id}"] = np.random.normal(size=(traj_len, n_robot_joints)).astype(np.float32)
            data[f"qdot_{robot_id}"] = np.random.normal(size=(traj_len, n_robot_joints)).astype(np.float32)
            data[f"p_{robot_id}"] = np.random.normal(size=(traj_len, 6)).astype(np.float32)  # 0~2: position / 3~5: orientation (euler)
            data[f"pdot_{robot_id}"] = np.random.normal(size=(traj_len, 6)).astype(np.float32)  # 0~2: linear velocity / 3~5: angular velocity
            
        # Generate exteroception
        data["color"] = np.random.randint(0, 256, (traj_len, 480, 640, 3), dtype=np.uint8)  # (height, width, channel) image
        
        # Generate robot control
        for robot_id in robot_ids:
            data[f"tele_abs_control_{robot_id}"] = np.random.normal(size=(traj_len, 6)).astype(np.float32)  # 0~2: position / 3~5: orientation (euler)
            
        # Generate gripper control
        if robot_mode in [RobotMode.SINGLE_ROBOT_GRIPPER, RobotMode.DUAL_ROBOT_GRIPPER]:
            for robot_id in robot_ids:
                data[f"trigger_value_{robot_id}"] = np.random.uniform(low=0, high=1, size=(traj_len, 1)).astype(np.float32)
                
        # Save
        with h5py.File(f"{DATA_DIR}/{data_id}.h5", "w") as hf:
            for name, value in data.items():
                hf.create_dataset(name, data=value)
            
    # Print data properties
    for name, value in data.items():
        print(f"{name}: {value.shape} ({value.dtype})")
            
    
    
    