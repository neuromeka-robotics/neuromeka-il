import time
from enum import Enum, auto

from helper.extra_utils import KeyboardListener, load_NN_controller
from helper.controller_utils import Base_NN_controller

class COMMAND_MACHINE(Enum):
    MOVE_TO_TASK_HOME = auto()
    EXECUTE_TASK = auto()
    SET_TELEOP = auto()
    TRIGGER_DIRECT_TEACHING = auto()
    EXECUTE_EMERGENCY_STOP = auto()
    EXECUTE_NN_STOP = auto()

    # default
    NO_COMMAND = auto()

    @staticmethod
    def process_io(io_data):
        if io_data['1']:
            return COMMAND_MACHINE.MOVE_TO_TASK_HOME
        elif io_data['2']:
            return COMMAND_MACHINE.EXECUTE_TASK
        elif io_data['7']:
            return COMMAND_MACHINE.SET_TELEOP
        elif io_data['8']:
            return COMMAND_MACHINE.TRIGGER_DIRECT_TEACHING
        elif io_data['9']:
            return COMMAND_MACHINE.EXECUTE_EMERGENCY_STOP
        elif io_data['0']:
            return COMMAND_MACHINE.EXECUTE_NN_STOP
        else:
            return COMMAND_MACHINE.NO_COMMAND


if __name__ == "__main__":
    # set task name
    demo_task = "act_il"
    
    # set nn controller and robot connection
    NN_controller = load_NN_controller(controller_type=demo_task)
    nn_controller: Base_NN_controller = NN_controller()
    
    # set keyboard listener
    task_command_listener = KeyboardListener(key_targets=['1', '2', '7', '8', '9', '0'])

    # Set keyboard commmand period.
    # Too small value may slow down other threading (e.g., _nn_control_fn), 
    # especially when using pynput for keyboard input.
    DEMO_COMMAND_PERIOD = 0.2 

    # initialize
    prev_emergency_state = False
    prev_direct_teaching_state = False

    # start
    while True:
        demo_command_start = time.time()

        task_command_data = task_command_listener.get_key_states()
        if task_command_data["updated"]:
            CURRENT_COMMAND = COMMAND_MACHINE.process_io(task_command_data["value"])

            if CURRENT_COMMAND == COMMAND_MACHINE.MOVE_TO_TASK_HOME:
                print(f"COMMAND: MOVE_TO_FIRST_TASK_HOME")
                nn_controller.exec_home_pos()
            elif CURRENT_COMMAND == COMMAND_MACHINE.EXECUTE_TASK:
                print(f"COMMAND: EXECUTE_FIRST_TASK")
                nn_controller.exec_nn_control(duration=1200.)
            elif CURRENT_COMMAND == COMMAND_MACHINE.EXECUTE_NN_STOP:
                print(f"COMMAND: EXECUTE_NN_STOP")
                nn_controller.exec_nn_control_stop()
            elif CURRENT_COMMAND == COMMAND_MACHINE.SET_TELEOP:
                print(f"COMMAND: SET_TELEOP")
                nn_controller.exec_set_teleop()

            if CURRENT_COMMAND == COMMAND_MACHINE.TRIGGER_DIRECT_TEACHING:
                prev_direct_teaching_state = not prev_direct_teaching_state
                print(f"COMMAND: TRIGGER_DIRECT_TEACHING ({prev_direct_teaching_state})")
                nn_controller.exec_direct_teaching(enable=prev_direct_teaching_state)
            
            if CURRENT_COMMAND == COMMAND_MACHINE.EXECUTE_EMERGENCY_STOP:
                prev_emergency_state = not prev_emergency_state
                print(f"COMMAND: EXECUTE_EMERGENCY_STOP ({prev_emergency_state})")
                nn_controller.exec_emergency_stop(enable=prev_emergency_state)
        
        demo_command_end = time.time()
        wait_time = DEMO_COMMAND_PERIOD - (demo_command_end - demo_command_start)
        if wait_time > 0.:
            time.sleep(wait_time)





