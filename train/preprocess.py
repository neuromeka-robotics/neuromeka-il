import os
import shutil
import numpy as np
import argparse
import h5py
from tqdm import tqdm
import ntpath
from shutil import copyfile

from nrmk_il.helper.utils import check_dir, get_base_dir, MathFunc

"""
Pre-process raw data in the format suitable for training.

Follow TODO 1~3 to add new types of data.
"""

N_FILTER_IDX = 10  # 0.05s * 10 = 0.5s
N_SUCCESS_IDX = 6  # 0.05s * 6 = 0.3s


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
        gripper_position_{id}: (T, 1)  [float]
        grasp_state_{id}: (T, 1)  [float]
        
        # Exteroception (camera)
        images.rgb.{cam_name}: (T, H, W, C)  [uint8]
        images.depth.{cam_name}: (T, H, W)  [float]
        images.intrinsics.{cam_name}: (3, 3)  [float]
        
        # Control (robot - task space control)
        tele_abs_control_{id}: (T, 6)  [float] [0~2: mm, 3~5: degree]
        
        # Control (gripper)
        gripper_command_{id}: (T, 1)  [float, range from 0 to 1]
        
        ### TODO 1: Add name and shape of new data
        """
        self.cam_data_dict = dict()
        for key, value in kwargs.items():
            if key.startswith("images."):
                if "rgb" in key:
                    self.cam_data_dict[f"observation.{key}"] = value[N_FILTER_IDX:]  # (T, H, W, C)
                elif "depth" in key:
                    self.cam_data_dict[f"observation.{key}"] = value[N_FILTER_IDX:][..., np.newaxis]  # (T, H, W, 1)
                elif "intrinsics" in key:
                    cam_name = key.split(".")[-1]
                    n_steps = kwargs[f"images.rgb.{cam_name}"].shape[0]
                    self.cam_data_dict[f"extra.{key}"] = np.repeat(value[np.newaxis], n_steps, axis=0)[N_FILTER_IDX:]  # (T, 3, 3)
            else:
                setattr(self, key, value)
    
    def process(self):
        # Check robot ids
        robot_ids = []        
        for name in self.__dict__.keys():
            if name == "cam_data_dict":
                continue
            
            robot_id = int(name.split("_")[-1])
            if robot_id not in robot_ids:
                robot_ids.append(robot_id)
        robot_ids = sorted(robot_ids)
        
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
        gripper_position = dict()
        grasp_state = dict()
        gripper_command = dict()
        prev_gripper_command = dict()
        
        for robot_id in robot_ids:
            # Proprioception (arm)
            q[robot_id] = MathFunc.degree_to_rad(getattr(self, f"q_{robot_id}"))
            qdot[robot_id] = MathFunc.degree_to_rad(getattr(self, f"qdot_{robot_id}"))
            end_pos[robot_id] = MathFunc.mm_to_m(getattr(self, f"p_{robot_id}")[:, :3])
            end_euler_ang = MathFunc.degree_to_rad(getattr(self, f"p_{robot_id}")[:, 3:])
            end_linVel[robot_id] = MathFunc.mm_to_m(getattr(self, f"pdot_{robot_id}")[:, :3])
            end_angVel[robot_id] = MathFunc.degree_to_rad(getattr(self, f"pdot_{robot_id}")[:, 3:])
            
            # Proprioception (gripper)
            gripper_position[robot_id] = getattr(self, f"gripper_position_{robot_id}").astype(np.float32) \
                if f"gripper_position_{robot_id}" in self.__dict__ else None
            grasp_state[robot_id] = getattr(self, f"grasp_state_{robot_id}").astype(np.float32) \
                if f"grasp_state_{robot_id}" in self.__dict__ else None
            
            # Control (robot - task space control)
            control_end_pos[robot_id] = MathFunc.mm_to_m(getattr(self, f"tele_abs_control_{robot_id}")[:, :3])
            control_end_euler_ang = MathFunc.degree_to_rad(getattr(self, f"tele_abs_control_{robot_id}")[:, 3:])
            
            # Control (gripper)
            if f"gripper_command_{robot_id}" in self.__dict__:
                gripper_command[robot_id] = getattr(self, f"gripper_command_{robot_id}").astype(np.float32)
                prev_gripper_command[robot_id] = np.roll(gripper_command[robot_id], 1, axis=0)
                prev_gripper_command[robot_id][0] = 0 if gripper_command[robot_id][0] == 0 else 1
            else:
                gripper_command[robot_id] = None
                prev_gripper_command[robot_id] = None
            
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
        
        gripper_position["total"] = None
        grasp_state["total"] = None
        gripper_command["total"] = None
        prev_gripper_command["total"] = None
        
        for robot_id in robot_ids:
            if gripper_position[robot_id] is not None:
                if gripper_position["total"] is None:
                    gripper_position["total"] = np.copy(gripper_position[robot_id])
                else:
                    gripper_position["total"] = np.concatenate([gripper_position["total"], gripper_position[robot_id]], axis=-1)
                    
            if grasp_state[robot_id] is not None:
                if grasp_state["total"] is None:
                    grasp_state["total"] = np.copy(grasp_state[robot_id])
                else:
                    grasp_state["total"] = np.concatenate([grasp_state["total"], grasp_state[robot_id]], axis=-1)
                    
            if gripper_command[robot_id] is not None:
                if gripper_command["total"] is None:
                    gripper_command["total"] = np.copy(gripper_command[robot_id])
                else:
                    gripper_command["total"] = np.concatenate([gripper_command["total"], gripper_command[robot_id]], axis=-1)
            
            if prev_gripper_command[robot_id] is not None:
                if prev_gripper_command["total"] is None:
                    prev_gripper_command["total"] = np.copy(prev_gripper_command[robot_id])
                else:
                    prev_gripper_command["total"] = np.concatenate([prev_gripper_command["total"], prev_gripper_command[robot_id]], axis=-1) 
        
        gripper_position = gripper_position["total"]
        grasp_state = grasp_state["total"]
        gripper_command = gripper_command["total"]
        prev_gripper_command = prev_gripper_command["total"]
        
        # Generate success label
        episode_len = q.shape[0]
        is_success = np.zeros((episode_len, 1), dtype=np.float32)
        is_success[-N_SUCCESS_IDX:] = 1
        
        # Set processed data
        ### TODO 3: Add new data with appropriate key
        self.processed_data = {
            "observation.qpos": q[N_FILTER_IDX:],  # (T, n_joints * n_robots)
            "observation.qvel": qdot[N_FILTER_IDX:],  # (T, n_joints * n_robots)
            "observation.end_position": end_pos[N_FILTER_IDX:],  # (T, 3 * n_robots)
            "observation.end_orientation": end_ori[N_FILTER_IDX:],  # (T, n_robots, 3, 3)
            "observation.end_linear_velocity": end_linVel[N_FILTER_IDX:],  # (T, 3 * n_robots)
            "observation.end_angular_velocity": end_angVel[N_FILTER_IDX:],  # (T, 3 * n_robots)
            **self.cam_data_dict,  # (T, H, W, C) / (T, H, W)
            "action.end_pos": control_end_pos[N_FILTER_IDX:],  # (T, 3 * n_robots)
            "action.end_ori": control_end_ori[N_FILTER_IDX:],  # (T, n_robots, 3, 3)
            "is_success": is_success[N_FILTER_IDX:]  # (T, 1)
        }
        
        if gripper_position is not None:
            self.processed_data["action.gripper_position"] = gripper_position[N_FILTER_IDX:]  # (T, 1 * n_robots)
        if grasp_state is not None:
            self.processed_data["observation.grasp_state"] = grasp_state[N_FILTER_IDX:]  # (T, 1 * n_robots)
        if gripper_command is not None:
            self.processed_data["action.gripper_command"] = gripper_command[N_FILTER_IDX:]  # (T, 1 * n_robots)
        if prev_gripper_command is not None:
            self.processed_data["observation.prev_gripper_command"] = prev_gripper_command[N_FILTER_IDX:]  # (T, 1 * n_robots)
        
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
    parser.add_argument("--n_filter_idx", default=0, type=int, help="Remove first n_filter_idx")
    parser.add_argument("--n_success_idx", default=5, type=int, help="Label last n_success_idx as success")
    return parser

def main(args):
    # Update root data directory
    BASE_DIR = get_base_dir()
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
    
    # Copy DAGGER progress file if exists
    dagger_progress_file = f"{DATA_PARENT_DIR}/{args.task}_progress.json"
    if os.path.exists(dagger_progress_file):
        base_file_name = ntpath.basename(dagger_progress_file)
        copyfile(dagger_progress_file, f"{PROCESSED_DATA_PARENT_DIR}/{base_file_name}")
        print(f"Copying also DAGGER progress file: {base_file_name}")
        
    # Set filter idx
    global N_FILTER_IDX
    N_FILTER_IDX = args.n_filter_idx
    print(f"Process start idx: {N_FILTER_IDX}")
    
    # Set success idx
    global N_SUCCESS_IDX
    N_SUCCESS_IDX = args.n_success_idx
    print(f"Success stard idx: {-N_SUCCESS_IDX}")
    
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
    





