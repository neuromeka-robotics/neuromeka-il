from typing import Dict, List

import time
import numpy as np
from collections import deque
from threading import Thread
from statemachine import StateMachine, State

from communication.robot import Robot
from helper.extra_utils import KeyboardListener, load_NN_controller, NN_CONTROL_STATE
from helper.controller_utils import Base_NN_controller
from perception.realsense import RealsenseCamHandler


class FSM(StateMachine):
    starting = State(initial=True)
    brooming = State()
    pick_and_placing = State()
    finishing = State()
    
    start = (
        starting.to(brooming, cond="target_reached")
        | starting.to(finishing)
    )
    
    broom = (
        brooming.to(pick_and_placing, cond="task_succeed")
        | brooming.to(brooming, cond="task_failed")
        | brooming.to.itself(internal=True)
    )
    
    pick_and_place = (
        pick_and_placing.to(finishing, cond="task_succeed")
        | pick_and_placing.to(pick_and_placing, cond="task_failed")
        | pick_and_placing.to.itself(internal=True)
    )
    
    def __init__(self,
                 robot: Dict[int, Robot],
                 controller: Dict[str, Base_NN_controller],
                 task_durations: Dict[str, float]):
        
        self.robot = robot
        self.controller = controller
        self.task_durations = task_durations
        
        self._target_joint_pos: Dict[int, List[float]] = {}
        self._current_task: str = None
        
        super().__init__()
    
    def on_enter_starting(self):
        # FSM init pos
        self._target_joint_pos = {
            0: [0., 0, -90., 0., -90., 0.]
        }
        
    def on_enter_brooming(self):
        # Update task
        self._current_task = "broom"
        
        # Start nn control
        self.start_nn_control_task()
        
    def on_exit_brooming(self):
        self.controller[self._current_task].exec_nn_control_stop()
        
    def on_enter_pick_and_placing(self):
        # Update task
        self._current_task = "pick_and_place"
        
        # Start nn control
        self.start_nn_control_task()
        
    def on_exit_pick_and_placing(self):
        self.controller[self._current_task].exec_nn_control_stop()
        
    def target_reached(self, event_data):
        error_threshold = event_data.trigger_data.kwargs.get("error", 0.1)
        assert isinstance(error_threshold, float)
        
        for robot_id, target in self._target_joint_pos.items():
            current_joint_pos = np.array(self.robot[robot_id].get_state()["q"])
            target_joint_pos = np.array(target)
            error = np.linalg.norm(current_joint_pos - target_joint_pos)
            if error > error_threshold:
                return False
        return True
    
    def start_nn_control_task(self):
        # Move to home pos
        self.controller[self._current_task].exec_home_movement(wait=True)
        
        # Execute task
        self.controller[self._current_task].exec_nn_control(
            duration=self.task_durations[self._current_task]
        )
        
    def task_succeed(self):
        return self.controller[self._current_task].control_state == NN_CONTROL_STATE.TASK_SUCCESS
    
    def task_failed(self):
        return self.controller[self._current_task].control_state in [NN_CONTROL_STATE.TASK_FAIL, NN_CONTROL_STATE.ROBOT_FAIL]
    

if __name__ == "__main__":
    # Set tasks
    demo_tasks = ["broom", "pick_and_place"]
    
    # Set maximum task duration
    task_durations = {"broom": 10., "pick_and_place": 10}
    
    # Set robot
    robot = {
        0: Robot(
            robot_ip = "192.168.0.111",
            gripper_config = {
                "enable": True,
                "type": "RobotiqUSBClient",
                "params": {
                    "port": "/dev/robotiq_2f85"
                }
            }
        )
    }
    
    # Set camera
    camera = {
        "left": RealsenseCamHandler(
            serial_number = "207222072747",
            align = True,
            clipping_distance_m = 1.,
        ),
        "right": RealsenseCamHandler(
            serial_number = "317622073859",
            align = True,
            clipping_distance_m = 1.,
        ),
    }
    for cam_name in camera.keys():
        camera[cam_name].start()
        
    # Set controller
    controller: Dict[str, Base_NN_controller] = dict()
    for task in demo_tasks:
        NN_controller = load_NN_controller(controller_type=task)
        controller[task] = NN_controller(robot=robot, camera=camera)
        controller[task].load_policy()
        
    # Set finite state machine
    fsm = FSM(robot=robot, controller=controller, task_durations=task_durations)
    
    FSM_PERIOD = 0.1
    current_state = None
    
    while True:
        fsm_start = time.time()
        
        prev_state = current_state
        current_state = fsm.current_state
        
        if current_state == FSM.starting:
            fsm.start(error=70.)
        elif current_state == FSM.brooming:
            fsm.broom()
        elif current_state == FSM.pick_and_placing:
            fsm.pick_and_place()
        elif current_state == FSM.finishing:
            break
        
        fsm_end = time.time()
        wait_time = FSM_PERIOD - (fsm_end - fsm_start)
        if wait_time > 0.:
            time.sleep(wait_time)
            
    if prev_state == FSM.starting:
        print("Cannot start the system because robots are far from initial position")
    else:
        print("Total task sequence finished")