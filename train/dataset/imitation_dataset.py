from typing import List, Dict, Tuple, Any

import os
import h5py
import numpy as np
import cv2
import torch
import torch.utils
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from config.configuration_base import BaseConfig


class ImageLoadDataset(torch.utils.data.Dataset):
    def __init__(self, 
                 episode_ids: List[int], 
                 dataset_dir: str, 
                 camera_names: List[str], 
                 image_size: List[int], 
                 action_horizon: int, 
                 **kwargs):
        super(ImageLoadDataset).__init__()
        self.episode_ids = np.sort(episode_ids)
        self.dataset_dir = dataset_dir
        self.camera_names = camera_names
        self.image_size = image_size
        self.action_horizon = action_horizon

        self.image_crop = dict() 
        if "image_crop" in kwargs.keys():
            for image_name, crop_area in kwargs["image_crop"].items():
                self.image_crop[image_name] = lambda img: img[crop_area[0][0]:crop_area[0][1], crop_area[1][0]:crop_area[1][1]]
        
        self.image_resize = lambda img: cv2.resize(img, dsize=image_size) if image_size is not None else None

        self.use_augmentation = True
        self.augmentation = transforms.Compose([
            transforms.ColorJitter(
                brightness=0.5,
                contrast=0.5,
                saturation=0.5
            ),
        ])

        self.dataset_size_dict: dict = {}
        self.get_replay_length()
        self.make_replay_start_index()

    def __len__(self):
        return sum(l for _, l in self.dataset_size_dict.items())

    def __getitem__(self, index: int):
        # Get episode id and episode-wise index
        episode_id = self.episode_ids[index >= self.start_indices][-1]
        start_ts = index - self.start_indices[index >= self.start_indices][-1]

        # Sample demonstration
        dataset_path = os.path.join(self.dataset_dir, f"{episode_id}.h5")
        root = h5py.File(dataset_path, "r")

        out_data = dict()

        # Sample trajectory in the demonstration
        episode_len = root[list(root.keys())[0]][:].shape[0]
        end_ts = min(start_ts + self.action_horizon, episode_len)
        action_len = end_ts - start_ts
        
        # Load data
        image_dict = {}
        for key, value in root.items():
            if "images" in key:
                temp_img = value[start_ts]
                
                # Crop and Resize
                if key in self.image_crop.keys():
                    temp_img = self.image_crop[key](temp_img)

                if self.image_resize is not None:
                    temp_img = self.image_resize(temp_img)
                
                image_dict[key.split('.')[-1]] = temp_img
            elif "observation" in key:
                value_ = value[start_ts]
                if value_.ndim == 2 or value_.ndim == 3:
                    if key == "observation.end_orientation":
                        n_robots = value_.shape[0]
                    value_ = value_.reshape(-1)
                elif value_.ndim != 1:
                    raise ValueError("Wrong Data Dimension")
                out_data[key] = torch.from_numpy(value_).to(torch.float32)
            elif "action" in key:
                value_ = value[start_ts:end_ts]
                if value_.ndim == 1:
                    value_ = value_[..., np.newaxis]
                elif value_.ndim == 3 or value_.ndim == 4:
                    value_ = value_.reshape(action_len, -1)
                elif value_.ndim != 2:
                    raise ValueError("Wrong Data Dimension")
                padded_value = np.tile(value_[-1, :], (self.action_horizon, 1))
                padded_value[:action_len] = value_
                is_pad = np.zeros(self.action_horizon)
                is_pad[action_len:] = 1
                
                out_data[key] = torch.from_numpy(padded_value).to(torch.float32)
                out_data["is_pad"] = torch.from_numpy(is_pad).to(torch.bool)
        
        # Construct image
        all_cam_images = [image_dict[cam_name] for cam_name in self.camera_names]
        all_cam_images = np.stack(all_cam_images, axis=0)  # (n_cam, H, W, C)
        out_data["images"] = torch.from_numpy(all_cam_images).to(torch.float32)
        out_data["images"] = torch.einsum('k h w c -> k c h w', out_data["images"])

        image_data = out_data["images"]
        
        # Augment image
        if self.use_augmentation:
            image_data = image_data.to(torch.uint8)
            for i in range(len(self.camera_names)):
                image_data[i] = self.augmentation(image_data[i])  # apply different augmentation for each camera image
            image_data = image_data.to(torch.float32)
            
        # Normalize image
        image_data /= 255.
            
        # Compute action labels for relative_delta task space control
        for idx in range(n_robots):
            current_end_pos_w = out_data["observation.end_position"][3 * idx : 3 * (idx + 1)]  # (T, 3 * num_robots)
            current_end_ori_w = out_data["observation.end_orientation"][9 * idx : 9 * (idx + 1)].reshape(3, 3)
            target_end_pos_w = out_data["action.end_pos"][:, 3 * idx : 3 * (idx + 1)]  # (T, 3 * num_robots)
            target_end_ori_w = out_data["action.end_ori"][:, 9 * idx : 9 * (idx + 1)].reshape(self.action_horizon, 3, 3)
            
            relative_delta_end_pos = \
                torch.bmm(torch.transpose(current_end_ori_w, 0, 1).unsqueeze(0).expand(self.action_horizon, -1, -1), (target_end_pos_w - current_end_pos_w).unsqueeze(-1)).squeeze(-1)
            relative_delta_end_ori = \
                torch.bmm(
                    torch.transpose(current_end_ori_w, 0, 1).unsqueeze(0).expand(self.action_horizon, -1, -1), 
                    target_end_ori_w).reshape(self.action_horizon, -1)

            if idx == 0:
                out_data["action.relative_delta.end_pos"] = relative_delta_end_pos
                out_data["action.relative_delta.end_ori"] = relative_delta_end_ori
            else:
                out_data["action.relative_delta.end_pos"] = np.concatenate((out_data["action.relative_delta.end_pos"], relative_delta_end_pos), axis=-1)
                out_data["action.relative_delta.end_ori"] = np.concatenate((out_data["action.relative_delta.end_ori"], relative_delta_end_ori), axis=-1)
            
        for cam_idx, cam_name in enumerate(self.camera_names):
            out_data[f"observation.images.{cam_name}"] = image_data[cam_idx]
        del out_data["images"]
        
        return out_data

    def get_replay_length(self):
        for episode_id in self.episode_ids:
            dataset_path = os.path.join(self.dataset_dir, f"{episode_id}.h5")
            with h5py.File(dataset_path, 'r') as root:
                sample_key = list(root.keys())[0]
                sample_data = root[sample_key][:]
                self.dataset_size_dict[episode_id] = len(sample_data)

    def make_replay_start_index(self):
        prev_start_idx = 0
        self.start_indices = [0]
        for episode_id in self.episode_ids[:-1]:
            cur_start_idx = prev_start_idx + self.dataset_size_dict[episode_id]
            self.start_indices.append(cur_start_idx)
            prev_start_idx = cur_start_idx
        self.start_indices = np.array(self.start_indices)


def get_norm_stats(dataset_dir: str) -> Dict[str, Dict[str, torch.tensor]]:
    """
    Compute dataset statistics and use them during training and testing
    """ 
    import psutil
    def get_memory_usage(mode="used"):
        assert mode in ["used", "total"]
        memory_info = psutil.virtual_memory()
        if mode == "used":
            return memory_info.used / (1024 * 1024)  # Return used memory in MB
        elif mode == "total":
            return memory_info.total / (1024 * 1024)  # Return total memory in MB
    
    total_memory = get_memory_usage(mode="total")

    total_data = {}
    for file in os.listdir(dataset_dir):
        memory_usage = get_memory_usage(mode="used")
        if memory_usage > total_memory * 0.5:
            print("[WARNING] Not enough memory to load all data for statistics computation\n\n")
            break
        
        with h5py.File(os.path.join(dataset_dir, file), "r") as root:
            for key, value in root.items():
                if "image" in key:
                    continue
                
                value = value[:]
                assert type(value) is np.ndarray, f"{key}:{type(value)}. Only numpy array is allowed"
                value_ = torch.from_numpy(value)
                episode_len = value_.shape[0]
                
                if value_.dim() == 1:
                    value_ = value_.unsqueeze(-1)
                elif value_.dim() == 3 or value_.dim() == 4:
                    value_ = value_.reshape(episode_len, -1)
                elif value_.dim() != 2:
                    raise ValueError("Wrong Data Dimension")
                
                if key in total_data:
                    total_data[key].append(value_)
                else:
                    total_data[key] = [value_]
        
    stats = {key:{} for key in total_data}
    for key, value in total_data.items():
        concatenated_value = torch.concat(value, dim=0) # (T1+T2+..., n_joints)  or  (T1+T2+..., n_joints * n_robots) or ((T1 + T2 + ...) * n_dim,)
        stats[key]["mean"] = torch.mean(concatenated_value, dim=0)
        stats[key]["std"] = torch.clip(torch.std(concatenated_value, dim=0), min=1e-5, max=np.inf)
        stats[key]["max"] = torch.max(concatenated_value, dim=0).values
        stats[key]["min"] = torch.min(concatenated_value, dim=0).values
    return stats


def load_data(base_cfg: BaseConfig, 
              train_config: Dict[str, Any], 
              camera_names: List[str], 
              image_size: List[int], 
              action_horizon: int, 
              compute_relative_delta_norm: bool = False, 
              **kwargs) -> Tuple[torch.utils.data.DataLoader, Dict[str, Dict[str, torch.tensor]]]:
    
    print(f'\nData from: {base_cfg.dataset_dir}\n')
    
    # Set train indexes
    num_episodes = len(os.listdir(base_cfg.dataset_dir))
    shuffled_indices = np.random.permutation(num_episodes)
    
    # Compute dataset statistics
    norm_stats = get_norm_stats(base_cfg.dataset_dir)
    
    # Set consistent generator for the dataloader
    torch_gen = torch.Generator()
    torch_gen.manual_seed(base_cfg.seed)
    
    # Construct dataset and dataloader
    image_dataset = ImageLoadDataset(
        episode_ids=shuffled_indices,
        dataset_dir=base_cfg.dataset_dir,
        camera_names=camera_names,
        image_size=image_size,
        action_horizon=action_horizon,
        **kwargs
    )
    image_dataloader = DataLoader(
        image_dataset, 
        batch_size=train_config["batch_size"],
        shuffle=True, 
        pin_memory=True, 
        num_workers=base_cfg.num_workers, 
        prefetch_factor=1,
        generator=torch_gen,
        drop_last=True
    )
        
    # Compute stats for relative_delta task space control
    if compute_relative_delta_norm:
        temp_end_pos = []
        temp_end_ori = []
        for data in image_dataloader:
            temp_end_pos.append(data["action.relative_delta.end_pos"])
            temp_end_ori.append(data["action.relative_delta.end_ori"])
        temp_end_pos = torch.cat(temp_end_pos, dim=0)
        temp_end_ori = torch.cat(temp_end_ori, dim=0)
        norm_stats["action.relative_delta.end_pos"] = dict()
        norm_stats["action.relative_delta.end_ori"] = dict()
        norm_stats["action.relative_delta.end_pos"]["mean"] = torch.mean(temp_end_pos, dim=0)
        norm_stats["action.relative_delta.end_pos"]["std"] = torch.clip(torch.std(temp_end_pos, dim=0), min=1e-5, max=np.inf)
        norm_stats["action.relative_delta.end_pos"]["max"] = torch.max(temp_end_pos, dim=0).values
        norm_stats["action.relative_delta.end_pos"]["min"] = torch.min(temp_end_pos, dim=0).values
        norm_stats["action.relative_delta.end_ori"]["mean"] = torch.mean(temp_end_ori, dim=0)
        norm_stats["action.relative_delta.end_ori"]["std"] = torch.clip(torch.std(temp_end_ori, dim=0), min=1e-5, max=np.inf)
        norm_stats["action.relative_delta.end_ori"]["max"] = torch.max(temp_end_ori, dim=0).values
        norm_stats["action.relative_delta.end_ori"]["min"] = torch.min(temp_end_ori, dim=0).values
        
    return image_dataloader, norm_stats
