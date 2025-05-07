from typing import Dict, Union

import os
import math
import random
import numpy as np
import torch
import wandb

from config.configuration_base import BaseConfig


class MathFunc:
    @staticmethod
    def m_to_mm(length: np.ndarray) -> np.ndarray:
        """
        Convert m to mm
        """
        return length * 1000.
    
    @staticmethod
    def mm_to_m(length: np.ndarray) -> np.ndarray:
        """
        Convert mm to m
        """
        return length / 1000.

    @staticmethod
    def degree_to_rad(angle: np.ndarray) -> np.ndarray:
        """
        Convert degree to rad
        """
        return angle * np.pi / 180.
    
    @staticmethod
    def rad_to_degree(angle: np.ndarray) -> np.ndarray:
        """
        Convert rad to degree
        """
        return angle * 180. / np.pi
    
    @staticmethod
    def single_axis_rotMat(axis: str, angle: float) -> np.ndarray:
        """
        Generate single axis rotation matrix
        
        axis: x / y / z
        angle: [rad]
        """
        assert axis in ['x', 'y', 'z'], "Unavailable axis"
        
        c = np.cos(angle)
        s = np.sin(angle)

        if axis == 'x':
            return np.array([[1, 0, 0],
                            [0, c, -s],
                            [0, s, c]])
        elif axis == 'y':
            return np.array([[c, 0, s],
                            [0, 1, 0],
                            [-s, 0, c]])
        else:
            return np.array([[c, -s, 0],
                            [s, c, 0],
                            [0, 0, 1]])

    @staticmethod
    def euler_to_rotMat(euler_x: float, euler_y: float, euler_z: float) -> np.ndarray:
        """
        Convert euler angles to rotation matrix
        
        euler_x, euler_y, euler_z: [rad]
        """
        R_x = MathFunc.single_axis_rotMat('x', euler_x)
        R_y = MathFunc.single_axis_rotMat('y', euler_y)
        R_z = MathFunc.single_axis_rotMat('z', euler_z)
        return R_z @ R_y @ R_x
    
    @staticmethod
    def rotMat_to_euler(rotMat: np.ndarray) -> np.ndarray:
        """
        Convert rotation matrix to euler angles
        """
        assert(MathFunc.is_rotMat(rotMat)), "Given matrix is not rotation matrix."
        sy = math.sqrt(rotMat[0,0] * rotMat[0,0] + rotMat[1,0] * rotMat[1,0])
        singular = sy < 1e-6
        if  not singular :
            x = math.atan2(rotMat[2,1] , rotMat[2,2])
            y = math.atan2(- rotMat[2,0], sy)
            z = math.atan2(rotMat[1,0], rotMat[0,0])
        else :
            x = math.atan2(- rotMat[1,2], rotMat[1,1])
            y = math.atan2(- rotMat[2,0], sy)
            z = 0
        return np.array([x, y, z])
    
    @staticmethod
    def is_rotMat(rotMat: np.ndarray) -> bool:
        """
        Check whether the rotation matrix is correct
        """
        Rt = np.transpose(rotMat)
        shouldBeIdentity = np.dot(Rt, rotMat)
        I = np.identity(3, dtype = rotMat.dtype)
        n = np.linalg.norm(I - shouldBeIdentity)
        return n < 1e-4
    

class TorchRotMatFunc:
    @staticmethod
    def normalize_vector(v: torch.tensor) -> torch.tensor:
        """
        Normalize vector in batch
        
        v: (n_batch, 3)
        """
        batch = v.shape[0]
        v_mag = torch.sqrt(v.pow(2).sum(1))
        v_mag = torch.max(v_mag, torch.autograd.Variable(torch.FloatTensor([1e-8]).cuda()))
        v_mag = v_mag.view(batch, 1).expand(batch,v.shape[1])
        v = v / v_mag
        return v
        
    @staticmethod
    def cross_product(u: torch.tensor, v: torch.tensor) -> torch.tensor:
        """
        Compute cross product between vectors in batch
        
        u: (n_batch, 3)
        v: (n_batch, 3)
        """
        batch = u.shape[0]
        i = u[:,1] * v[:,2] - u[:,2] * v[:,1]
        j = u[:,2] * v[:,0] - u[:,0] * v[:,2]
        k = u[:,0] * v[:,1] - u[:,1] * v[:,0]
            
        out = torch.cat((i.view(batch,1), j.view(batch,1), k.view(batch,1)),1)
        return out
            
    @staticmethod
    def compute_rotation_matrix_from_ortho6d(poses: torch.tensor) -> torch.tensor:
        """
        Compute rotation matrix from two vectors via orthogonalization
        
        poses: (n_batch, 6)
        """
        x_raw = poses[:,0:3]
        y_raw = poses[:,3:6]
            
        x = TorchRotMatFunc.normalize_vector(x_raw)
        z = TorchRotMatFunc.cross_product(x,y_raw)
        z = TorchRotMatFunc.normalize_vector(z)
        y = TorchRotMatFunc.cross_product(z,x)
            
        x = x.view(-1,3,1)
        y = y.view(-1,3,1)
        z = z.view(-1,3,1)
        matrix = torch.cat((x,y,z), 2)
        return matrix
    
    
class Logger:
    def __init__(self, base_cfg: BaseConfig, project_name: str="imitation learning"):
        wandb.init(
            project=project_name,
        )
        wandb.run.name = f"{base_cfg.model_name}/{base_cfg.task_name}/{base_cfg.policy_class}"
        
    def store_log(self, epoch_summary: Dict[str, float], name: str="Train"):
        log_data = dict()
        for k, v in epoch_summary.items():
            log_data[f"{name}/{k}"] = v
        wandb.log(log_data)


def cat_batch_state(batch: Dict[str, torch.tensor], state_name="observation.state") -> bool:
    """
    Concatenate observations into a single state
    
    In-place operation for "batch"
    """
    states = None
    key_to_pop = []
    batch.keys()
    for key, value in batch.items():  # TODO: Convert to fixed key order
        if ("observation" in key) and ("images" not in key):
            if states is None:
                states = value
            else:
                states = torch.cat((states, value), dim=-1)
            key_to_pop.append(key)
            
    [batch.pop(key) for key in key_to_pop]
    batch[state_name] = states
    return states is not None

def cat_batch_action(batch: Dict[str, torch.tensor], action_name = "action") -> bool:
    """
    Concatenate actions into a single action
    
    In-place operation for "batch"
    """
    actions = None
    key_to_pop = []
    for key, value in batch.items():  # TODO: Convert to fixed key order
        if "action" in key:
            if actions is None:
                actions = value
            else:
                actions = torch.cat((actions, value), dim=2)
            key_to_pop.append(key)
    [batch.pop(key) for key in key_to_pop]
    batch[action_name] = actions
    return actions is not None
    
def decode_6d_action_head(actions_hat: torch.Tensor, num_robots: int) -> torch.Tensor:
    """
    Generate full task space action (i.e., position + rotation matrix) based on https://arxiv.org/abs/1812.07035
    """
    batch_size, chunk_size, _ = actions_hat.shape
    actions_hat = actions_hat.reshape(batch_size * chunk_size, -1)  # (B * T, 9 * n_robots)
    a_hat_pos = actions_hat[:, :3 * num_robots]  # (B * T, 3 * n_robots)
    a_hat_ori = actions_hat[:, 3 * num_robots:]  # (B * T, 6 * n_robots)

    a_hat_rot = dict()
    for robot_id in range(num_robots):
        a_hat_rot[robot_id] = TorchRotMatFunc.compute_rotation_matrix_from_ortho6d(a_hat_ori[:, 6 * robot_id:6 * (robot_id + 1)])  # (B * T, 3, 3)
        a_hat_rot[robot_id] = a_hat_rot[robot_id].view(batch_size * chunk_size, -1)  # (B * T, 9)
    a_hat_rot = torch.cat([a_hat_rot[robot_id] for robot_id in range(num_robots)], dim=-1)  # (B * T, 9 * num_robots)
    actions_hat = torch.cat((a_hat_pos, a_hat_rot), dim=-1).view(batch_size, chunk_size, -1)  # (B, T, 12 * num_robots)
    return actions_hat

def check_dir(folder: str, generate: bool=False) -> bool:
    """
    Check directory existence and generate if necessary
    """
    if os.path.isdir(folder):
        return True
    else:
        if generate:
            os.makedirs(folder)
            return True
        else:
            return False
        
def detach_dict(d: Dict[str, Union[np.ndarray, torch.tensor]]) -> Dict[str, Union[np.ndarray, torch.tensor]]:
    """
    Detach tensors in dict
    """
    new_d = dict()
    for k, v in d.items():
        if type(v) == torch.Tensor:
            new_d[k] = v.detach()
        else:
            new_d[k] = v
    return new_d

def set_seed(seed: int):
    """
    Set seed
    """
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # # Just use for initial debugging and reproduction
        # # because below conditions may slow down the training
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    
def update_pbar(pbar, forward_dict: Dict[str, float]):
    """
    Update status of the progress bar
    """
    loss_dict = {}
    for k, v in forward_dict.items():
        loss_dict[k] = round(float(v), 2)
    pbar.set_postfix(**loss_dict)
    pbar.update(1)
