import os
from pathlib import Path
import datetime
from dataclasses import dataclass

from nrmk_il.helper.utils import get_base_dir

@dataclass
class BaseConfig:
    task_name: str = "test"

    policy_class: str = "act"  # act

    seed: int = 0
    num_workers: int = 1
    logging: bool = True

    model_name: str | None = None  # Only required for dagger_mode
    dagger_mode: bool = False

    # pretrained model directory
    pretrained_ckpt_dir: str | Path | None = None
    
    def __post_init__(self):
        if self.dagger_mode:
            assert self.model_name is not None, "In DAGGER mode, trained model should be provided"
        else:
            assert self.model_name is None, "In TRAIN FROM SCRATCH mode, model_name is automatically generated"
            self.model_name = datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
            
        BASE_DIR = get_base_dir()
        self.dataset_dir = os.path.join(BASE_DIR, "processed_data", self.task_name)
        self.ckpt_dir = os.path.join(BASE_DIR, "weights", self.task_name, self.model_name)
            
        if self.pretrained_ckpt_dir is not None:
            assert os.path.isdir(self.pretrained_ckpt_dir), "Pre-trained model does not exist."
                