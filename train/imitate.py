from typing import Union, Dict, List, Tuple, Any

import hydra
from hydra.core.hydra_config import HydraConfig
import os
import pickle
from tqdm import tqdm
import ntpath
from shutil import copyfile

import numpy as np
import torch

from dataset.imitation_dataset import load_data
from config.configuration_base import BaseConfig
from policies import ACTConfig
from policies import ACTPolicy
from policies.selection import ControlMode
from helper.utils import (
    set_seed, detach_dict, check_dir, update_pbar,
    Logger
)


def build_stats(stats: Dict[str, Dict[str, torch.tensor]], 
                vision_backbone: Union[str, None], 
                key_names: List[str]) -> Dict[str, Dict[str, torch.tensor]]:
    """
    Add input statistics for pretrained vision backbones
    """
    final_stats = stats
    for key_name in key_names:
        if vision_backbone == "resnet18":
                final_stats[key_name] = {
                    "mean": torch.tensor([[[0.485]], [[0.456]], [[0.406]]], dtype=torch.float32),
                    "std": torch.tensor([[[0.229]], [[0.224]], [[0.225]]], dtype=torch.float32)
                }
        else:
            raise NotImplementedError
    return final_stats

def make_policy(policy_config: Union[ACTConfig], 
                policy_class: str, 
                stats: Dict[str, Dict[str, torch.tensor]]) -> torch.nn.Module:
    """
    Generate policy
    """
    if policy_class == "act":
        policy = ACTPolicy(policy_config, stats)
    else:
        raise NotImplementedError
    return policy

def make_optimizer(train_config: Dict[str, Any], 
                   policy_class: str, 
                   policy: torch.nn.Module) -> Tuple[torch.optim.Optimizer, Union[None, torch.optim.lr_scheduler.LRScheduler]]:
    """
    Generate optimizer for the policy
    """
    if policy_class == "act":
        param_dicts = [
            {"params": [p for n, p in policy.named_parameters() if "backbone" not in n and p.requires_grad]},
            {
                "params": [p for n, p in policy.named_parameters() if "backbone" in n and p.requires_grad],
                "lr": train_config["lr_backbone"],
            },
        ]
        optimizer = torch.optim.AdamW(
            param_dicts, lr=train_config["lr"], weight_decay=train_config["weight_decay"])
        lr_scheduler = None
    else:
        raise NotImplementedError
    return optimizer, lr_scheduler

def forward_pass(data: Dict[str, torch.tensor], 
                 policy: Union[ACTPolicy]) -> Tuple[Dict[str, torch.tensor], torch.tensor]:
    """
    Forward pass the policy and compute loss
    """
    batch_data_keys = list(policy.config.input_shapes.keys()) + list(policy.config.output_shapes.keys())
    batch_data = {}
    for key in batch_data_keys:
        batch_data[key] = data[key].cuda()
    return policy(batch_data)

def train_bc(
    config: BaseConfig, 
    train_config: Dict[str, Any], 
    train_dataloader, 
    policy: torch.nn.Module, 
    optimizer: torch.optim.Optimizer, 
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler,
    **kwargs):
    # Set logger
    if config.logging:
        train_logger = Logger(config, project_name="imitation learning")

    # Set progress bar
    pbar = tqdm(total=train_config["num_epochs"])
    
    # Start training
    policy.train()
    grad_clip_norm = train_config.get("grad_clip_norm")

    for epoch in range(train_config["num_epochs"]):
        # Save model
        if epoch % 10 == 0:
            ckpt_path = os.path.join(config.ckpt_dir, "policy_last.ckpt")
            torch.save(policy.state_dict(), ckpt_path)
                
        # Train model
        for data in train_dataloader:
            forward_dict, _ = forward_pass(data, policy, **kwargs)
            loss = forward_dict["loss"]
            loss.backward()
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(policy.parameters(), grad_clip_norm, error_if_nonfinite=False)
            optimizer.step()
            optimizer.zero_grad()
            
            if lr_scheduler is not None:
                lr_scheduler.step()

            iter_summary = detach_dict(forward_dict)
            
            # Update logger
            if config.logging:
                train_logger.store_log(iter_summary)
            iter_summary["epoch"] = epoch
                     
        # Update monitoring metrics
        update_pbar(pbar, iter_summary)
    
    # Conclude training
    ckpt_path = os.path.join(config.ckpt_dir, "policy_last.ckpt")
    torch.save(policy.state_dict(), ckpt_path)
    pbar.close()

@hydra.main(version_base=None, config_path="config", config_name="configuration.yaml")
def main(cfg):
    # Load config
    base_config = BaseConfig(**cfg["base"])
    if base_config.policy_class == "act":
        policy_config = ACTConfig(**cfg['policy'])

        img_key_names = [k for k in policy_config.input_shapes if k.startswith("observation.images")]
        camera_names = []
        candidate_words = ["observation.images"]
        for k in policy_config.input_shapes:
            for candidate_word in candidate_words:
                if k.startswith(candidate_word):
                    camera_names.append(k.split(".")[-1])
                    image_size = policy_config.input_shapes[f"{candidate_word}.{camera_names[0]}"][-2:]  # Assume same size for all images

        camera_names = list(set(camera_names))  # Remove duplicates, Assume camera name is same for image
        action_horizon = policy_config.chunk_size
        vision_backbone = policy_config.vision_backbone
    else:
        raise ValueError
    train_config = cfg["train"]
    extra_config = cfg["extra"] if "extra" in cfg.keys() else {}
    
    # Set seed
    set_seed(base_config.seed)
    
    # Set dataloader and compute dataset statistics
    compute_relative_delta_norm = (policy_config.control_mode == ControlMode.RELATIVE_DELTA_TASK_SPACE)
    train_dataloader, stats = load_data(
        base_cfg=base_config,
        train_config=train_config,
        camera_names=camera_names,
        image_size=image_size,
        action_horizon=action_horizon,
        compute_relative_delta_norm=compute_relative_delta_norm,
        **extra_config
    )
    stats = build_stats(stats, vision_backbone, img_key_names)

    # Save dataset statistics and critical files
    check_dir(base_config.ckpt_dir, generate=True)
    print(f"checkpoint dir: {base_config.ckpt_dir}\n")
    
    if base_config.pretrained_ckpt_dir is not None:
        # Load pre-trained statistics
        pretrained_stats_path = os.path.join(base_config.pretrained_ckpt_dir, "dataset_stats.pkl")
        with open(pretrained_stats_path, "rb") as f:
            stats_np = pickle.load(f)
        
        for k in stats_np.keys():
            stats[k]["mean"] = torch.from_numpy(stats_np[k]["mean"])
            stats[k]["std"] = torch.from_numpy(stats_np[k]["std"])
        print(f"Loaded stats: {pretrained_stats_path}")
    else:
        # Set statistics of current dataset
        stats_np = {}
        for key in stats.keys():
            stats_np[key] = {}
            stats_np[key]["mean"] = stats[key]["mean"].numpy()
            stats_np[key]["std"] = stats[key]["std"].numpy()
            
    stats_path = os.path.join(base_config.ckpt_dir, "dataset_stats.pkl")
    with open(stats_path, 'wb') as f:
        pickle.dump(stats_np, f)
        
    given_config_file = HydraConfig.get().job.config_name
    overrided_config_path = os.path.join(HydraConfig.get().runtime.output_dir, '.hydra', 'config.yaml')
    
    save_items = [overrided_config_path]
    for save_item in save_items:
        base_file_name = ntpath.basename(save_item)
        copyfile(save_item, f"{base_config.ckpt_dir}/{base_file_name}")

    # Make policy and optimizer
    policy = make_policy(policy_config, base_config.policy_class, stats)
    
    # Load pre-trained model
    if base_config.pretrained_ckpt_dir is not None:
        pretrained_ckpt_path = os.path.join(base_config.pretrained_ckpt_dir, "policy_last.ckpt")
        policy.load_state_dict(torch.load(pretrained_ckpt_path, weights_only=True))
        print(f"Loaded model: {pretrained_ckpt_path}")
        
    policy.cuda()
    optimizer, lr_scheduler = make_optimizer(train_config, base_config.policy_class, policy)
    
    # Train policy
    train_bc(
        config=base_config, 
        train_config=train_config, 
        train_dataloader=train_dataloader, 
        policy=policy, 
        optimizer=optimizer, 
        lr_scheduler=lr_scheduler, 
    )


if __name__ == '__main__':
    main()
