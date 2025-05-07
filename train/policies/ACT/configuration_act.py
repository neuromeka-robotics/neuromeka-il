#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field

from policies.selection import RobotMode, ControlMode

@dataclass
class ACTConfig:
    """
    Configuration class for the Action Chunking Transformers policy.
    """
    # Robot / Control mode
    robot_mode: str | RobotMode
    control_mode: str | ControlMode
    
    # Number of robots to control (post-initialized using robot_mode)
    num_robots: int | None = None

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 20
    n_action_steps: int = 20
    
    input_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "observation.images.top": [3, 480, 640],
            "observation.qpos": [6],
            "observation.qvel": [6],
        }
    )
    output_shapes: dict[str, list[int]] = field(
        default_factory=lambda: {
            "action.pos": [3],
            "action.rot": [6],
            "is_pad": [1],
        }
    )
    action_dim: int = 9
    
    # Normalization / Unnormalization
    input_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {
            "observation.images.top": "mean_std",
            "observation.qpos": "mean_std",
            "observation.qvel": "mean_std",
        }
    )
    output_normalization_modes: dict[str, str] = field(
        default_factory=lambda: {
            "action.pos": "mean_std",
        }
    )

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: str | None = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: int = False
    
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512 # hidden_dim in original code
    n_heads: int = 8 # nheads in original code
    dim_feedforward: int = 3200 # dim_feedforward in original code
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4 # enc_layers in original code
    n_decoder_layers: int = 1 # dec_layers in original code
    
    # VAE.
    use_vae: bool = False
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4
    
    # Success detector
    use_success_detector: bool = False
    
    # Inference.
    temporal_ensemble_momentum: float | None = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 20.0 # kl_weight in original code

    def __post_init__(self):
        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_momentum is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

        assert isinstance(self.robot_mode, str), "Robot mode should be initially defined in string"
        self.robot_mode = RobotMode.name_to_mode(self.robot_mode)
        
        assert self.num_robots is None, "Num robots should not be set manually. It is automatically defined in the post-initialization"
        self.num_robots = RobotMode.get_num_robots(self.robot_mode)

        assert isinstance(self.control_mode, str), "Control mode should be initially defined in string"
        self.control_mode = ControlMode.name_to_mode(self.control_mode)
        
        if (self.robot_mode == RobotMode.SINGLE_ROBOT) \
            and (self.control_mode in ControlMode.get_candidate(name="task")):
            assert self.action_dim >= 3 + 6
        elif (self.robot_mode == RobotMode.SINGLE_ROBOT_GRIPPER) \
            and (self.control_mode in ControlMode.get_candidate(name="task")):
            assert self.action_dim >= 3 + 6 + 1
        elif (self.robot_mode == RobotMode.DUAL_ROBOT) \
            and (self.control_mode in ControlMode.get_candidate(name="task")):
            assert self.action_dim >= (3 + 6) * 2
        elif (self.robot_mode == RobotMode.DUAL_ROBOT_GRIPPER) \
            and (self.control_mode in ControlMode.get_candidate(name="task")):
            assert self.action_dim >= (3 + 6 + 1) * 2

        if self.control_mode in ControlMode.get_candidate("task") and self.temporal_ensemble_momentum is not None:
            raise NotImplementedError("Task space control does not currently support temporal smoothing")