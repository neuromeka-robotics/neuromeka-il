from typing import List, Dict

import os
from pathlib import Path
import datetime
from dataclasses import dataclass


@dataclass
class BaseConfig:
    policy_class: str = "act"  # act
    task_name: str = "test"
    seed: int = 0
    num_workers: int = 1
    logging: bool = True

    # directory
    dataset_dir: str | Path | None = None
    ckpt_dir: str | Path | None = None
    pretrained_ckpt_dir: str | Path | None = None
    
    def __post_init__(self):
        self.model_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            
        TRAIN_DIR_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                    
        if self.dataset_dir is None:
            self.dataset_dir = f"{TRAIN_DIR_PATH}/processed_data/{self.task_name}"
            
        if self.ckpt_dir is None:
            self.ckpt_dir = f"{TRAIN_DIR_PATH}/weights/{self.task_name}/{self.model_name}"
            
        if self.pretrained_ckpt_dir is not None:
            assert os.path.isdir(self.pretrained_ckpt_dir), "Pre-trained model does not exist."
                