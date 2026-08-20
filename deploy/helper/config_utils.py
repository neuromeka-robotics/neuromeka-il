from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Dict, Callable
import os

from helper.extra_utils import default_home_movement, default_start_movement, default_finish_movement

from nrmk_il.helper.utils import get_base_dir

@dataclass
class ROBOT_CONFIG:
    robot_ids: List[int] | None = None
    
    robot_params: Dict[int, Dict] = field(
        default_factory=lambda: {
            0: {
                "ip": "192.168.0.135",
                "home_pos": [-5.0032735, -20.997824, -84.92132, -0.020232247, -73.95311, -3.4574845],
                "gripper": {
                    "enable": False,
                    "type": "RobotiqUSBClient",
                    "params": {
                        "port": "/dev/robotiq_2f85"
                    }
                },
                "control": {
                    "vel_scale": 1.,  # 0 ~ 1
                    "acc_scale": 10.,  # 0 ~ 10
                    "move_vel_scale": 50.,  # 0 ~ 100
                    "move_acc_scale": 50.  # 0 ~ 1000
                }
            }
        }
    )
    
    control_dt: float = 0.05
    # Optional Robot subclass used by Controller when constructing connections.
    # Individual robot entries may override this with a ``robot_class`` key.
    robot_class: type | None = None
    
    def __post_init__(self):
        self.robot_ids = list(self.robot_params.keys())
        
        for robot_id in self.robot_ids:
            assert "ip" in self.robot_params[robot_id].keys()
            
            assert "home_pos" in self.robot_params[robot_id].keys()
            home_pos = self.robot_params[robot_id]["home_pos"]
            if home_pos is not None and not isinstance(home_pos, list):
                raise TypeError(
                    f"home_pos for robot {robot_id} must be a list or None")
            robot_class = self.robot_params[robot_id].get(
                "robot_class", self.robot_class)
            expected_dof = getattr(robot_class, "JOINT_DOF", None)
            if (home_pos is not None and expected_dof is not None
                    and len(home_pos) != expected_dof):
                raise ValueError(
                    f"home_pos for robot {robot_id} must contain exactly "
                    f"{expected_dof} values; got {len(home_pos)}")
            
            assert "gripper" in self.robot_params[robot_id].keys()
            assert "enable" in self.robot_params[robot_id]["gripper"].keys() and isinstance(self.robot_params[robot_id]["gripper"]["enable"], bool)
            
            assert "control" in self.robot_params[robot_id].keys()
            for control_param_type in ["vel_scale", "acc_scale", "move_vel_scale", "move_acc_scale"]:
                assert control_param_type in self.robot_params[robot_id]["control"].keys()


@dataclass
class CAMERA_CONFIG:
    cam_names: List[str] | None = None
    cam_params: Dict[str, Dict] = field(
        default_factory=lambda: {
            "wrist": {
                "serial": "233522079515",
                "exposure": 50,
                "enable_depth": False 
            }
        }
    )
    
    def __post_init__(self):
        self.cam_names = list(self.cam_params.keys())
        
        for cam_name in self.cam_names:
            assert "serial" in self.cam_params[cam_name].keys(), f"Camera serial number is not provided for {cam_name} camera"
            

@dataclass
class MODEL_CONFIG:
    model_type: str = "act"
    model_dir: str = "/GLOBAL/PATH/TO/MODEL/DIRECTORY"
    model_file: str = "policy_last.ckpt"
    success_threshold: float = 1.
    device: str = "cuda"
    port: int = 5555  # Required for external nn model usage via server-client communication
    
    def __post_init__(self):
        assert self.device in ["cpu", "cuda"], f"Unavailable device {self.device}"
    

@dataclass
class DATA_CONFIG:
    data_dir: str = os.path.join(get_base_dir(), "data")
    data_viz_dir: str =  os.path.join(get_base_dir(), "data_viz")
    
    device_type: str = "vive"
    device_params: Dict = field(
        default_factory=lambda: {
            "calib_uvw": [-1.5413670975757867, 3.1056404500490942, 1.1692256106471774]
        }
    )
    # Optional BaseDevice subclass for project-specific task- or joint-space
    # devices. When set, it takes precedence over device_type.
    device_class: type | None = None
    
    def __post_init__(self):
        if self.device_class is None:
            assert self.device_type in ["vive", "spacemouse"], \
                f"Unavailable device {self.device_type}"
        
@dataclass
class EXTRA_CONFIG:
    home_movement_fn: Callable = default_home_movement
    start_movement_fn: Callable = default_start_movement
    finish_movement_fn: Callable = default_finish_movement
    
    control_post_process_fn : Callable = lambda x: x


@dataclass
class COMPLIANCE_CONFIG:
    enable: bool = False
    stiffness: List[int] | None = None


@dataclass
class TELEOP_CONFIG:
    # Device output mode comes from BaseDevice.CONTROL_MODE. This is the mode
    # actually sent to the robot after FK/IK conversion.
    robot_control_mode: str = "task_abs"
    # arm_index can be:
    #   - int: single robot, single arm (backward compatible)
    #   - Dict[int, int]: multiple robots, each with one arm
    #   - List[int]: single robot, multiple arms (e.g., dual-arm humanoid teleop)
    arm_index: int | Dict[int, int] | List[int] = 0
    # IK is used only when a task-space device drives joint-space teleop.
    # STEP does not support locking; Pink locks joints inside its optimization.
    ik_type: str = "step"
    pink_config_path: str | None = None
    # During task->joint conversion, keep all joints outside the selected
    # head/arm chain at their current values.
    lock_non_selected_joints: bool = False
    compliance: COMPLIANCE_CONFIG = field(default_factory=COMPLIANCE_CONFIG)

    def __post_init__(self):
        if self.robot_control_mode not in ["joint_abs", "task_abs"]:
            raise ValueError(
                f"Unavailable robot control mode {self.robot_control_mode}")
        if isinstance(self.arm_index, int):
            if self.arm_index < 0:
                raise ValueError("arm_index must be non-negative")
        elif isinstance(self.arm_index, dict):
            for idx in self.arm_index.values():
                if idx < 0:
                    raise ValueError("arm_index values must be non-negative")
        elif isinstance(self.arm_index, list):
            for idx in self.arm_index:
                if idx < 0:
                    raise ValueError("arm_index values must be non-negative")
        else:
            raise TypeError(
                "arm_index must be an int, a List[int], or a Dict[int, int]")
        if self.ik_type not in ["step", "pink"]:
            raise ValueError(f"Unavailable IK type {self.ik_type}")
        if not isinstance(self.lock_non_selected_joints, bool):
            raise TypeError("lock_non_selected_joints must be a bool")
        if self.ik_type == "step" and self.lock_non_selected_joints:
            raise ValueError(
                "lock_non_selected_joints is unavailable for STEP IK; "
                "use ik_type='pink' or disable locking")
        if self.ik_type == "pink" and not self.pink_config_path:
            raise ValueError("pink_config_path is required for Pink IK")

@dataclass
class TASK_CONFIG:
    name: str = "base"
    camera_config: CAMERA_CONFIG | None = field(default_factory=CAMERA_CONFIG)
    model_config: MODEL_CONFIG | None = field(default_factory=MODEL_CONFIG)
    data_config: DATA_CONFIG | None = field(default_factory=DATA_CONFIG)
    extra_config: EXTRA_CONFIG = field(default_factory=EXTRA_CONFIG)
    teleop_config: TELEOP_CONFIG = field(default_factory=TELEOP_CONFIG)
    
    def __post_init__(self):
        assert self.camera_config is not None, \
            "Camera configuration is not provided"
        assert self.model_config is not None or self.data_config is not None, \
            "Either model or data configuration must be provided"
    
    
BASE_ROBOT_CONFIG = ROBOT_CONFIG()
BASE_TASK_CONFIG = TASK_CONFIG()
