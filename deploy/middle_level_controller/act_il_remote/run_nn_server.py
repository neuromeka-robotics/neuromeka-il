from typing import Dict

import sys
import os
import argparse
from ruamel.yaml import YAML
import pickle
import time
import numpy as np
import torch
import torchvision.transforms as transforms

# helper functions
def get_base_dir():
    """
    Check absolute directory path to "deploy"
    """
    import re
    BASE_DIR = re.search(r'(.*/deploy/)', os.path.abspath(__file__)).group(1)
    return BASE_DIR

sys.path.append(get_base_dir())
from helper.remote_utils import BaseRequestHandler, PickleServer, PickleClient
from helper.math_utils import MathFunc

# external nn models
# In this example, we use ACT from "nrmk_il" package
from nrmk_il.policies import ACTConfig, ACTPolicy
from nrmk_il.policies.selection import ControlMode, GripperMode


class RequestHandler(BaseRequestHandler):
    def __init__(self):
        self.policy: ACTPolicy | None = None

    def handle(self, received_data):
        if received_data["operation"] == "load":
            data = received_data["data"]
            assert "model_config" in data.keys(), "Model config should be provided."
            
            model_type = data["model_config"]["model_type"]
            model_dir = data["model_config"]["model_dir"]
            model_file = data["model_config"]["model_file"]
            self.device = data["model_config"]["device"]
            
            # load policy config
            cfg: Dict[str, Dict] = YAML().load(
                open(
                    os.path.join(model_dir, "config.yaml"),
                    "r",
                )
            )
            
            if model_type == "act":
                policy_config: ACTConfig = ACTConfig(**cfg["policy"])
            else:
                raise NotImplementedError
            
            # In external model config, "n_robots [int]", "control_mode [str]", "grippper_mode [str]" should be provided
            self.n_robots: int = policy_config.num_robots
            self.control_mode: str = ControlMode.mode_to_name(policy_config.control_mode)
            self.gripper_mode: str = GripperMode.mode_to_name(GripperMode.robot_mode_to_gripper_mode(policy_config.robot_mode))
            if self.gripper_mode in ["binary", "continuous"]:
                self.use_gripper = True
            else:
                self.use_gripper = False
                
            # load dataset statistics
            with open(
                os.path.join(model_dir, "dataset_stats.pkl"),
                "rb",
            ) as f:
                policy_stats = pickle.load(f)

            for k in policy_stats.keys():
                policy_stats[k]["mean"] = torch.from_numpy(policy_stats[k]["mean"])
                policy_stats[k]["std"] = torch.from_numpy(policy_stats[k]["std"])
                
            # set image pre-processor
            candidate_word = "observation.images"
            for key, value in policy_config.input_shapes.items():
                if key.startswith(candidate_word):
                    image_size = value[-2:]  # Assume same size for all images
            self.image_resize = transforms.Resize(image_size)
            
            self.image_crop = dict()
            if cfg.get("extra") is not None and cfg["extra"].get("image_crop") is not None:
                for image_name, crop_area in cfg["extra"]["image_crop"].items():
                    self.image_crop[image_name] = \
                        lambda img, crop=crop_area: img[crop[0][0]:crop[0][1], crop[1][0]:crop[1][1]]
            
            # load policy
            if model_type == "act":
                self.policy = ACTPolicy(policy_config, policy_stats)
            else:
                raise NotImplementedError
            
            policy_weight_path = os.path.join(model_dir, model_file)
            self.policy.load_state_dict(
                torch.load(
                    policy_weight_path,
                    weights_only=True,
                )
            )
            self.policy.to(self.device)
            self.policy.eval()
            print(
                f"Loaded (policy): {policy_weight_path}"
            )
            
            return self.success_response(message={
                "num_robots": self.n_robots, 
                "control_mode": self.control_mode, 
                "gripper_mode": self.gripper_mode})
            
        elif received_data["operation"] == "reset":
            self.policy.reset()
            return self.success_response(message={})
        
        elif received_data["operation"] == "step":
            data = received_data["data"]

            if not ("proprioception" in data.keys() and isinstance(data["proprioception"], dict)) \
                or not ("images" in data.keys() and isinstance(data["images"], dict)):
                return self.error_response(message={"comment": "No required input data"})
            
            # pre-process input data
            qpos = data["proprioception"]["qpos"].astype(np.float32)  # (n_joints * n_robots)
            qvel = data["proprioception"]["qvel"].astype(np.float32)  # (n_joints * n_robots)
            end_pose = data["proprioception"]["end_pose"].astype(np.float32)  # (6 * n_robots)
            end_vel = data["proprioception"]["end_vel"].astype(np.float32)  # (6 * n_robots)
            
            cam_data_dict = dict()
            for cam_name in data["images"].keys():
                cam_data_dict[f"observation.images.rgb.{cam_name}"] = data["images"][cam_name]["rgb"].astype(np.float32)  # (H, W, C)
                if data["images"][cam_name].get("depth") is not None:
                    cam_data_dict[f"observation.images.depth.{cam_name}"] = data["images"][cam_name]["depth"][..., np.newaxis]  # (H, W, 1)
                    
            if self.use_gripper:
                gripper_pos = data["proprioception"]["gripper_pos"].astype(np.float32)  # (n_robots)
                grasp_state = data["proprioception"]["grasp_state"].astype(np.float32)  # (n_robots)
                prev_gripper_cmd = data["proprioception"]["prev_gripper_command"].astype(np.float32)  # (n_robots)
                gripper_pos_data = torch.from_numpy(gripper_pos).unsqueeze(0)  # (B, n_robots)
                grasp_state_data = torch.from_numpy(grasp_state).unsqueeze(0)  # (B, n_robots)
                prev_gripper_cmd_data = torch.from_numpy(prev_gripper_cmd).unsqueeze(0)  # (B, n_robots)
                    
            # Image crop
            for key in cam_data_dict.keys():
                if self.image_crop.get(key) is not None:
                    cam_data_dict[key] = self.image_crop[key](cam_data_dict[key])

            # Change unit
            qpos = MathFunc.degree_to_rad(qpos)
            qvel = MathFunc.degree_to_rad(qvel)
            end_pos = MathFunc.mm_to_m(end_pose[:3])
            end_ori = MathFunc.degree_to_rad(end_pose[3:])
            end_ori = MathFunc.euler_to_rotMat(
                euler_x=end_ori[0], euler_y=end_ori[1], euler_z=end_ori[2]
            )
            end_linVel = MathFunc.mm_to_m(end_vel[:3])
            end_angVel = MathFunc.degree_to_rad(end_vel[3:])
            
            # Pre-process
            for key in cam_data_dict.keys():
                image_data = torch.from_numpy(cam_data_dict[key])
                image_data = torch.einsum('h w c -> c h w', image_data)
                image_data = self.image_resize(image_data)
                image_data /= 255.
                cam_data_dict[key] = image_data.unsqueeze(0)  # (B, C, H, W)

            # prepare available data
            available_data = {
                "observation.qpos": torch.from_numpy(qpos).unsqueeze(0),
                "observation.qvel": torch.from_numpy(qvel).unsqueeze(0),
                "observation.end_position": torch.from_numpy(end_pos).unsqueeze(0),
                "observation.end_orientation": torch.from_numpy(end_ori.reshape(-1)).unsqueeze(0),  # (B, 9)
                "observation.end_linear_velocity": torch.from_numpy(end_linVel).unsqueeze(0),
                "observation.end_angular_velocity": torch.from_numpy(end_angVel).unsqueeze(0),
                **cam_data_dict
            }
            if self.use_gripper:
                available_data["observation.gripper_position"] = gripper_pos_data
                available_data["observation.grasp_state"] = grasp_state_data
                available_data["observation.prev_gripper_command"] = prev_gripper_cmd_data
                
            # select data from the available data that the policy requires
            batch_data_keys = list(self.policy.config.input_shapes.keys())
            batch_data = dict()
            for key in batch_data_keys:
                batch_data[key] = available_data[key].to(self.device).contiguous()
                
            is_new_chunk: bool = len(self.policy._action_queue) == 0
                
            # forward pass neural network
            with torch.no_grad():
                output = self.policy.select_action(batch_data)  # (B, 12 * n_robots) or (B, (12 + 1) * n_robots)
            action = output["actions"].squeeze(0).cpu().numpy()
            if output["success_probability"] is None:
                success_prob = 0.  # No success detector
            else:
                success_prob = output["success_probability"].squeeze(0).cpu().numpy().item()
            
            return self.success_response(message={"action": action, "success_probability": success_prob, "is_new_chunk": is_new_chunk})
        else:
            return self.error_response(message={"comment": f"Unsupported request {received_data['operation']}"})


def run_fake_client(args):
    # Set client to communicate with the nn server
    client = PickleClient(host="localhost", port=args.port)
    
    # Request nn model loading
    print("Request to load model\n")
    response = client.send_data(
        data={
            "operation": "load",
            "data": {
                "model_config": {
                    "model_type": "act",
                    "model_dir": "/home/awesomericky/neuromeka_opensource/neuromeka-il/train/weights/broom/2025-08-21-21-31-21",
                    # "model_dir": "/home/awesomericky/neuromeka_opensource/neuromeka-il/train/weights/pick_and_place/2025-08-21-21-31-23",
                    "model_file": "policy_last.ckpt",
                    "device": "cuda"
                }
            }
        }
    )
    if response["result"] == "ERROR":
        raise ValueError("ZMQ error")
    
    n_robots = response["num_robots"]
    control_mode = response["control_mode"]
    gripper_mode = response["gripper_mode"]
    print(f"Num robots: {n_robots} / Control mode: {control_mode} / Gripper mode: {gripper_mode}\n")
    
    # Request nn model reset
    print("Request to reset model\n")
    response = client.send_data(data={"operation": "reset"})
    if response["result"] == "ERROR":
        raise ValueError("ZMQ error")
    
    try:
        while True:
            data = {
                "operation": "step",
                "data": {
                    "proprioception": {
                        "qpos": np.random.normal(size=(6,)),
                        "qvel": np.random.normal(size=(6,)),
                        "end_pose": np.random.normal(size=(6,)),
                        "end_vel": np.random.normal(size=(6,)),
                        "gripper_pos": np.random.uniform(size=(1,)),  # 0 ~ 1
                        "grasp_state": np.random.uniform(size=(1,)) > 0.5,  # bool
                        "prev_gripper_command": np.round(np.random.uniform(size=(1,))),  # 0 or 1
                    },
                    "images": {
                        "left": {
                            "rgb": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
                        },
                        "right": {
                            "rgb": np.random.randint(0, 256, (480, 640, 3), dtype=np.uint8)
                        }
                    }
                }
            }
            
            start = time.time()
            
            # Request nn model control
            response = client.send_data(data=data)
            
            end = time.time()
            
            if args.debug:
                print(f"Communication time: {(end - start):.3f}")
            
            if response["result"] == "ERROR":
                raise ValueError("ZMQ error")
        
            print(
                f"Action shape: {response['action'].shape} / Success probability: {response['success_probability']:.2f} / Is new chunk: {response['is_new_chunk']}")
                
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("Client interrupted, shutting down...")

    finally:
        client.close()
        print("Client shut down.")
        
        
def run_server(args):
    #################################################
    # External nn model specific code               #
    # In this example, the external nn model is ACT #
    #################################################
    request_handler = RequestHandler()
    server = PickleServer(host="localhost", port=args.port, request_handler=request_handler)
    server.serve()
    

def get_parser():
    parser = argparse.ArgumentParser(description="Data processing")
    parser.add_argument("--port", default=5555, type=int, help="NN server port")
    parser.add_argument("--fake_client", action="store_true", help="Run fake client for unit test and debugging")
    parser.add_argument("--debug", action="store_true", help="Turn on debug mode")
    return parser


if __name__ == "__main__":
    parser = get_parser()
    args = parser.parse_args()
    
    if args.fake_client:
        run_fake_client(args)
    else:
        run_server(args)