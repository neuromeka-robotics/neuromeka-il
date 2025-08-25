import time
from enum import Enum, auto

from helper.extra_utils import KeyboardListener, load_NN_controller
from helper.controller_utils import Base_NN_controller

class COMMAND_MACHINE(Enum):
    # Used for task execution
    MOVE_TO_TASK_HOME = auto()
    EXECUTE_TASK = auto()
    EXECUTE_NN_STOP = auto()
    
    # Used for DAGGER
    EXECUTE_START_STATE_DAGGER = auto()
    EXECUTE_CURRENT_STATE_DAGGER = auto()

    # default
    NO_COMMAND = auto()

    @staticmethod
    def process_io(io_data):
        if io_data['1']:
            return COMMAND_MACHINE.MOVE_TO_TASK_HOME
        elif io_data['2']:
            return COMMAND_MACHINE.EXECUTE_TASK
        elif io_data['0']:
            return COMMAND_MACHINE.EXECUTE_NN_STOP
        elif io_data['3']:
            return COMMAND_MACHINE.EXECUTE_START_STATE_DAGGER
        elif io_data['4']:
            return COMMAND_MACHINE.EXECUTE_CURRENT_STATE_DAGGER
        else:
            return COMMAND_MACHINE.NO_COMMAND


if __name__ == "__main__":
    # Set task name
    demo_task = "act_il"
    
    # Set nn controller and robot connection
    NN_controller = load_NN_controller(controller_type=demo_task)
    nn_controller: Base_NN_controller = NN_controller()
    
    # Load nn model weights
    nn_controller.load_policy()
    
    # Check DAGGER and set data_collector if necessary
    if nn_controller.TASK_CONFIG.data_config is not None:
        import importlib
        from helper.controller_utils import Controller
        from helper.config_utils import ROBOT_CONFIG, TASK_CONFIG
        from data_collector.collector import DataCollectionScheduler
        from data_collector.config import DATA_COLLECTOR_ROBOT_CONFIG, DATA_COLLECTOR_TASK_CONFIG
        
        # Overwrite robot and task configurations for data collector
        module = importlib.import_module(f"middle_level_controller.{demo_task}.config")
        DATA_COLLECTOR_ROBOT_CONFIG: ROBOT_CONFIG = module.CUSTOM_ROBOT_CONFIG
        DATA_COLLECTOR_TASK_CONFIG: TASK_CONFIG = module.CUSTOM_TASK_CONFIG
        
        data_collection_scheduler: Controller = DataCollectionScheduler(
            robot=nn_controller.robot,
            camera=nn_controller.camera,
            dagger_mode=True
        )
    else:
        data_collection_scheduler = None
    
    # Set keyboard listener
    task_command_listener = KeyboardListener(key_targets=['1', '2', '3', '4', '0'])

    # Set keyboard commmand period.
    # Too small value may slow down other threading (e.g., _nn_control_fn), 
    # especially when using pynput for keyboard input.
    DEMO_COMMAND_PERIOD = 0.2 

    # start
    while True:
        demo_command_start = time.time()

        task_command_data = task_command_listener.get_key_states()
        if task_command_data["updated"]:
            CURRENT_COMMAND = COMMAND_MACHINE.process_io(task_command_data["value"])

            if CURRENT_COMMAND == COMMAND_MACHINE.MOVE_TO_TASK_HOME:
                print(f"COMMAND: MOVE_TO_FIRST_TASK_HOME")
                nn_controller.exec_home_movement(wait=False)
            elif CURRENT_COMMAND == COMMAND_MACHINE.EXECUTE_TASK:
                print(f"COMMAND: EXECUTE_FIRST_TASK")
                nn_controller.exec_nn_control(duration=1200.)
            elif CURRENT_COMMAND == COMMAND_MACHINE.EXECUTE_NN_STOP:
                print(f"COMMAND: EXECUTE_NN_STOP")
                nn_controller.exec_nn_control_stop()
            elif CURRENT_COMMAND == COMMAND_MACHINE.EXECUTE_START_STATE_DAGGER:
                if data_collection_scheduler is not None:
                    print(f"COMMAND: EXECUTE_START_STATE_DAGGER")
                    # Stop nn controller
                    nn_controller.exec_nn_control_stop()
                    
                    # Start data collection from home pos
                    data_collection_scheduler.exec_collection(mode="start")
            elif CURRENT_COMMAND == COMMAND_MACHINE.EXECUTE_CURRENT_STATE_DAGGER:
                if data_collection_scheduler is not None:
                    print(f"COMMAND: EXECUTE_CURRENT_STATE_DAGGER")
                    # Stop nn controller
                    nn_controller.exec_nn_control_stop()
                    
                    # Start data collection from current pos
                    data_collection_scheduler.exec_collection(mode="current")

        demo_command_end = time.time()
        wait_time = DEMO_COMMAND_PERIOD - (demo_command_end - demo_command_start)
        if wait_time > 0.:
            time.sleep(wait_time)





