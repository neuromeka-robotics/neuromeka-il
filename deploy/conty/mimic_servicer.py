from typing import Dict, List

import os
import re
import sys
match = re.search(r'(.*/deploy/)', os.path.abspath(__file__))
BASE_DIR = match.group(1)
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "conty"))
sys.path.append(os.path.join(BASE_DIR, "conty/grpc"))

import importlib
import copy

# helper functions
from helper.controller_utils import Base_NN_controller
from helper.config_utils import ROBOT_CONFIG, TASK_CONFIG
from helper.extra_utils import load_NN_controller

# communication
from perception_module.realsense import RealsenseCamHandler

# grpc protocols for conty
import conty.grpc.mimic_msgs_pb2 as mimic_data
import conty.grpc.mimic_msgs_pb2 as mimic_grpc

PC_DEVICE_PORT = 20500
MAX_CONTROL_DURATION = 4  # s


class MoveMimicPC_servicer(mimic_grpc.MoveMimicServicer):
    def __init__(self):
        # Set empty variables
        self.robot_id: int = 0
        self.robot_ip: str | None = None
        self.nn_controllers: Dict[str, Base_NN_controller] = {}
        
        self._robot_skills: List[str] = []
        self._robot_ips: Dict[str, str] = {}
        self._robot_home_pos: Dict[str, List[float]] = {}
        self._checked_feasibility = False
        self._camera: Dict[str, RealsenseCamHandler] | None = None
        
        # Initialize
        self._robot_skills = os.listdir(os.path.join(BASE_DIR, "middle_level_controller"))
        if not self._checked_feasibility:
            self._check_feasibility()
            self._checked_feasibility = True
            
    def __del__(self):
        # Stop camera
        for cam_name in self._camera.keys():
            del self._camera[cam_name]
    
    def _check_feasibility(self):
        cam_configs: Dict[str, Dict] = {}
        for skill in self._robot_skills:
            # Import robot and task configurations
            module = importlib.import_module(f"middle_level_controller.{skill}.config")
            robot_config: ROBOT_CONFIG = module.CUSTOM_ROBOT_CONFIG
            task_config: TASK_CONFIG = module.CUSTOM_TASK_CONFIG
            
            # Check robot config
            assert len(robot_config.robot_ids) == 1, "Single robot is only supported for conty"
            assert robot_config.robot_ids[0] == self.robot_id, f"Single robot ID should be {self.robot_id}"
            self._robot_ips[skill] = robot_config.robot_params[self.robot_id]["ip"]
            self._robot_home_pos[skill] = robot_config.robot_params[self.robot_id]["home_pos"]
            
            # Check camera config
            for cam_name in task_config.camera_config.cam_names:
                if cam_name in cam_configs.keys():
                    assert cam_configs[cam_name] == task_config.camera_config.cam_params[cam_name], \
                        f"Camera configuration mismatch for camera {cam_name} in skill {skill}"
                else:
                    cam_configs[cam_name] = copy.deepcopy(task_config.camera_config.cam_params[cam_name])
            
        # Set camera connection
        for cam_name, cam_config in cam_configs.items():
            self._camera[cam_name] = RealsenseCamHandler(
                serial_number=cam_config["serial"],
                align=True,
                clipping_distance_m=1.,
                exposure=cam_config.get("exposure", None)
            )
            self._camera[cam_name].start()
        
    def SetRobotAddress(self, request: mimic_data.Address, context) -> mimic_data.Response:
        self.robot_ip = request.ip
        self.robot_port = request.port

    def GetSkillList(self, request: mimic_data.Empty, context) -> mimic_data.MimicSkillList:
        assert self._checked_feasibility, "Robot and camera feasibility are not checked" 
        
        # Set robot ip
        peer = context.peer()
        if peer.startswith('ipv4:'):
            self.robot_ip = peer.split(':')[1]
        else:
            self.robot_ip = 'unknown'
        print(f"Client is connected to server at IP: {self.robot_ip}")

        # Search skills corresponding to the robot IP
        available_skills = []
        for skill in self._robot_skills:
            if self._robot_ips[skill] == self.robot_ip:
               available_skills.append(skill)
        print(f"Available skills: {available_skills}")
               
        return mimic_data.MimicSkillList(skill_list=available_skills)
        
    def GetSkillHome(self, request: mimic_data.MimicSkillName, context) -> mimic_data.GetSkillHomeRes:
        print(f"Get Skill Home called with {request.name}")
        skill = request.name
        assert skill in self._robot_home_pos.keys(), f"Home positiion for '{skill}' not defined"
        
        return mimic_data.GetSkillHomeRes(jpos=self._robot_home_pos[skill])
    
    def RunSkill(self, request: mimic_data.MimicSkillName, context) -> mimic_data.Response:
        print("RunSkill called")
        skill = request.name
        
        # Set controller
        if skill not in self.nn_controllers.keys():
            controller_cls = load_NN_controller(controller_type=skill)
            self.nn_controllers[skill] = controller_cls(robot=None, camera=self._camera)
            self.nn_controllers[skill].load_policy()
        
        # Run model
        self.nn_controllers[skill].exec_nn_control(duration=MAX_CONTROL_DURATION)
        
        return mimic_data.Response()
    
    def StopSkill(self, request: mimic_data.Empty, context) -> mimic_data.Response:
        print("StopSkill called")
        for skill in self.nn_controllers.keys():
            self.nn_controllers[skill].exec_nn_control_stop()
            
        return mimic_data.Response()
        
        
if __name__ == "__main__":
    print("Running in main")
    import grpc
    from concurrent import futures

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10),
                         options=
                         [('grpc.max_send_message_length', 10 * 1024 * 1024),
                          ('grpc.max_receive_message_length', 10 * 1024 * 1024)]
                         )
    servicer = MoveMimicPC_servicer()
    mimic_grpc.add_MoveMimicServicer_to_server(servicer=servicer, server=server)

    server.add_insecure_port('[::]:{}'.format(PC_DEVICE_PORT))
    server.start()
    server.wait_for_termination()
