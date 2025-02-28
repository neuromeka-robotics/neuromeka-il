"""
Pre-process raw data in the format suitable for training.

Follow TODO 1~3 to add new types of data.
"""

import os
import shutil
import numpy as np
import argparse
import h5py
from tqdm import tqdm

from helper.utils import check_dir, MathFunc


class DataProcessor:
    def __init__(self, **kwargs):
        """
        [Data types and formats]
        
        # Proprioception (arm)
        q_{id}: (T, n_joints)  [float] [degree]
        qdot_{id}: (T, n_joints)  [float] [degree/s]
        p_{id}: (T, 6)  [float] [0~2: mm, 3~5: degree]
        pdot_{id}: (T, 6)  [float] [0~2: mm/s, 3~5: degree/s]
        
        # Proprioception (gripper) (Optional)
        trigger_value_{id}: (T, 1)  [float]
        
        # Exteroception (camera)
        color: (T, H, W, C)  [uint8]
        
        # Control (task space control)
        tele_abs_control_{id}: (T, 6)  [float] [0~2: mm, 3~5: degree]
        
        ### TODO 1: Add name and shape of new data
        """
        for key, value in kwargs.items():
            setattr(self, key, value)
    
    def process(self):
        # Check robot ids
        robot_ids = []        
        for name in self.__dict__.keys():
            if name not in ["color", "depth", "cam_intrinsics"]:
                robot_id = int(name.split("_")[-1])
                if robot_id not in robot_ids:
                    robot_ids.append(robot_id)
        robot_ids = sorted(robot_ids)
        
        # Exteroception (camera)
        color = self.color
        
        # Init proprioception and control
        # (1) Default (Must)
        q = dict()
        qdot = dict()
        end_pos = dict()
        end_ori = dict()
        end_linVel = dict()
        end_angVel = dict()
        control_end_pos = dict()
        control_end_ori = dict()
        
        # Gripper (Optional)
        trigger_value = dict()
        prev_trigger_value = dict()
        
        for robot_id in robot_ids:
            # Proprioception (arm)
            q[robot_id] = MathFunc.degree_to_rad(getattr(self, f"q_{robot_id}"))
            qdot[robot_id] = MathFunc.degree_to_rad(getattr(self, f"qdot_{robot_id}"))
            end_pos[robot_id] = MathFunc.mm_to_m(getattr(self, f"p_{robot_id}")[:, :3])
            end_euler_ang = MathFunc.degree_to_rad(getattr(self, f"p_{robot_id}")[:, 3:])
            end_linVel[robot_id] = MathFunc.mm_to_m(getattr(self, f"pdot_{robot_id}")[:, :3])
            end_angVel[robot_id] = MathFunc.degree_to_rad(getattr(self, f"pdot_{robot_id}")[:, 3:])
            
            if f"trigger_value_{robot_id}" in self.__dict__:
                trigger_value[robot_id] = getattr(self, f"trigger_value_{robot_id}")[..., np.newaxis]
                prev_trigger_value[robot_id] = np.roll(trigger_value[robot_id], 1, axis=0)
                prev_trigger_value[robot_id][0] = 0 if trigger_value[robot_id][0] == 0 else 1
            else:
                trigger_value[robot_id] = None
                prev_trigger_value[robot_id] = None
            
            # Control
            control_end_pos[robot_id] = MathFunc.mm_to_m(getattr(self, f"tele_abs_control_{robot_id}")[:, :3])
            control_end_euler_ang = MathFunc.degree_to_rad(getattr(self, f"tele_abs_control_{robot_id}")[:, 3:])
            
            ### TODO 2: Add processing parts for new data
            # Convert euler angle to rotation matrix
            end_ori[robot_id] = []
            control_end_ori[robot_id] = []

            n_steps = end_euler_ang.shape[0]
            
            # Compute for base frame
            for i in range(n_steps):
                rotMat = MathFunc.euler_to_rotMat(
                    euler_x=end_euler_ang[i, 0],
                    euler_y=end_euler_ang[i, 1],
                    euler_z=end_euler_ang[i, 2]
                )
                control_rotMat = MathFunc.euler_to_rotMat(
                    euler_x=control_end_euler_ang[i, 0],
                    euler_y=control_end_euler_ang[i, 1],
                    euler_z=control_end_euler_ang[i, 2]
                )
                end_ori[robot_id].append(rotMat.copy())
                control_end_ori[robot_id].append(control_rotMat.copy())

            end_ori[robot_id] = np.asarray(end_ori[robot_id]).astype(np.float32)
            control_end_ori[robot_id] = np.asarray(control_end_ori[robot_id]).astype(np.float32)
        
        # Concat
        q = np.concatenate([q[robot_id] for robot_id in robot_ids], axis=-1)
        qdot = np.concatenate([qdot[robot_id] for robot_id in robot_ids], axis=-1)
        end_pos = np.concatenate([end_pos[robot_id] for robot_id in robot_ids], axis=-1)
        end_ori = np.stack([end_ori[robot_id] for robot_id in robot_ids], axis=1)
        end_linVel = np.concatenate([end_linVel[robot_id] for robot_id in robot_ids], axis=-1)
        end_angVel = np.concatenate([end_angVel[robot_id] for robot_id in robot_ids], axis=-1)
        control_end_pos = np.concatenate([control_end_pos[robot_id] for robot_id in robot_ids], axis=-1)
        control_end_ori = np.stack([control_end_ori[robot_id] for robot_id in robot_ids], axis=1)
        
        trigger_value["total"] = None
        prev_trigger_value["total"] = None
        
        for robot_id in robot_ids:
            if trigger_value[robot_id] is not None:
                if trigger_value["total"] is None:
                    trigger_value["total"] = np.copy(trigger_value[robot_id])
                else:
                    trigger_value["total"] = np.concatenate([trigger_value["total"], trigger_value[robot_id]], axis=-1)
            
            if prev_trigger_value[robot_id] is not None:
                if prev_trigger_value["total"] is None:
                    prev_trigger_value["total"] = np.copy(prev_trigger_value[robot_id])
                else:
                    prev_trigger_value["total"] = np.concatenate([prev_trigger_value["total"], prev_trigger_value[robot_id]], axis=-1) 
        
        trigger_value = trigger_value["total"]
        prev_trigger_value = prev_trigger_value["total"]
        
        # Set processed data
        ### TODO 3: Add new data with appropriate key
        self.processed_data = {
            "observation.qpos": q,  # (T, n_joints * n_robots)
            "observation.qvel": qdot,  # (T, n_joints * n_robots)
            "observation.end_position": end_pos,  # (T, 3 * n_robots)
            "observation.end_orientation": end_ori,  # (T, n_robots, 3, 3)
            "observation.end_linear_velocity": end_linVel,  # (T, 3 * n_robots)
            "observation.end_angular_velocity": end_angVel,  # (T, 3 * n_robots)
            "observation.images.top": color,  # (T, H, W, C)
            "action.end_pos": control_end_pos,  # (T, 3 * n_robots)
            "action.end_ori": control_end_ori,  # (T, n_robots, 3, 3)
        }
        
        if trigger_value is not None:
            self.processed_data["action.trigger_value"] = trigger_value  # (T, 1 * n_robots)
        if prev_trigger_value is not None:
            self.processed_data["observation.prev_trigger_value"] = prev_trigger_value  # (T, 1 * n_robots)
        
    def save(self, file_name: str):
        with h5py.File(file_name, "w") as hf:
            for key, value in self.processed_data.items():
                hf[key] = value
                
    def check_shape(self):
        for key, value in self.processed_data.items():
            print(f"{key}: {value.shape}")


def get_parser():
    parser = argparse.ArgumentParser(description="Data processing")
    parser.add_argument("--task", required=True, type=str, help="Task name to process")
    return parser

def main(args):
    # Update root data directory
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_PARENT_DIR = os.path.join(BASE_DIR, "data")
    PROCESSED_DATA_PARENT_DIR = os.path.join(BASE_DIR, "processed_data")
    
    # Set target data directory
    TASK_DATA_DIR = f"{DATA_PARENT_DIR}/{args.task}"
    assert check_dir(TASK_DATA_DIR, generate=False), f"Data directory {TASK_DATA_DIR} does not exist"
    
    PROCESSED_TASK_DATA_DIR = f"{PROCESSED_DATA_PARENT_DIR}/{args.task}"
    exist = check_dir(PROCESSED_TASK_DATA_DIR, generate=False)
    if exist:
        print("Processed data directory already exists. Automatically removing them.")
        shutil.rmtree(PROCESSED_TASK_DATA_DIR)
    check_dir(PROCESSED_TASK_DATA_DIR, generate=True)
    
    for file in tqdm(os.listdir(TASK_DATA_DIR)):
        # Load raw data
        data = dict()
        with h5py.File(f"{TASK_DATA_DIR}/{file}", "r") as root:
            for k in root.keys():
                data[k] = root[k][:]
        
        # Process and save data
        processor = DataProcessor(**data)
        processor.process()
        processor.save(os.path.join(PROCESSED_TASK_DATA_DIR, file))
        
    processor.check_shape()  # Print data shape


if __name__ == '__main__':
    parser = get_parser()
    args = parser.parse_args()
    main(args)
    





