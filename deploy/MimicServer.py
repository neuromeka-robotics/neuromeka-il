import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "communication/impl"))
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(os.path.join(os.path.dirname(__file__), "../train"))

import communication.impl.mimic_msgs_pb2 as mimic_data
import communication.impl.mimic_pb2_grpc as mimic_grpc

from config.robot import ROBOT_HOME_POS
from imitation_learning.controller import NN_controller

PC_DEVICE_PORT = 20500
WEIGHT_DIR = os.path.join(os.path.dirname(__file__), "weights")
MAX_CONTROL_DURATION = 4  # s


class MimicPCServicer(mimic_grpc.MoveMimicServicer):
    robot_ip: str
    robot_port: str
    nn_controller: NN_controller
    
    def __init__(self):
        self.robot_ip = None
        self.robot_port = None
        self.nn_controller = None
        
    def SetRobotAddress(self, request: mimic_data.Address, context) -> mimic_data.Response:
        self.robot_ip = request.ip
        self.robot_port = request.port
        print(self.robot_ip)

    def GetSkillList(self, request: mimic_data.Empty, context) -> mimic_data.MimicSkillList:
        # Set robot ip
        peer = context.peer()
        if peer.startswith('ipv4:'):
            self.robot_ip = peer.split(':')[1]
        else:
            self.robot_ip = 'unknown'
        print(f"Client is connected to server at IP: {self.robot_ip}")

        # Get skill list
        skills = os.listdir(WEIGHT_DIR)
        
        # (1) Initialize neural network controller 
        # (2) Set connection to the robot and camera
        if self.nn_controller is None:
            self.nn_controller = NN_controller(tasks=skills)
        return mimic_data.MimicSkillList(skill_list=skills)
        
    def GetSkillHome(self, request: mimic_data.MimicSkillName, context) -> mimic_data.GetSkillHomeRes:
        print(f"Get Skill Home called with {request.name}")
        name = request.name
        assert name in ROBOT_HOME_POS.keys(), f"Home positiion for '{name}' not defined"

        return mimic_data.GetSkillHomeRes(jpos=ROBOT_HOME_POS[name])
    
    def RunSkill(self, request: mimic_data.MimicSkillName, context) -> mimic_data.Response:
        name = request.name
        assert name in self.nn_controller.get_available_tasks(), f"Task '{name}' not defined"
        
        # Load model
        self.nn_controller.load_policy(task=name)
        
        # Run model
        self.nn_controller.exec_nn_control(task=name, duration=MAX_CONTROL_DURATION)
        return mimic_data.Response()
    
    def StopSkill(self, request: mimic_data.Empty, context) -> mimic_data.Response:
        self.nn_controller.exec_nn_control_stop()
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
    servicer = MimicPCServicer()
    mimic_grpc.add_MoveMimicServicer_to_server(servicer=servicer, server=server)

    server.add_insecure_port('[::]:{}'.format(PC_DEVICE_PORT))
    server.start()
    server.wait_for_termination()
