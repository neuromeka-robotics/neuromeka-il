import time
from enum import Enum, auto

from helper.extra_utils import KeyboardListener
from helper.controller_utils import Controller


class COMMAND_MACHINE(Enum):
    MOVE_TO_TASK_HOME = auto()
    EXECUTE_START_STATE_COLLECTION = auto()
    EXECUTE_CURRENT_STATE_COLLECTION = auto()

    # default
    NO_COMMAND = auto()

    @staticmethod
    def process_io(io_data):
        if io_data['1']:
            return COMMAND_MACHINE.MOVE_TO_TASK_HOME
        elif io_data['2']:
            return COMMAND_MACHINE.EXECUTE_START_STATE_COLLECTION
        elif io_data['3']:
            return COMMAND_MACHINE.EXECUTE_CURRENT_STATE_COLLECTION
        else:
            return COMMAND_MACHINE.NO_COMMAND


if __name__ == "__main__":
    # set keyboard listener
    task_command_listener = KeyboardListener(key_targets=['1', '2', '3'])

    # set data collector and robot connection
    from data_collector.collector import DataCollectionScheduler
    data_collection_scheduler: Controller = DataCollectionScheduler()

    # Set keyboard commmand period.
    # Too small value may slow down other threading (e.g., _collection_fn), 
    # especially when using pynput for keyboard input.
    COMMAND_PERIOD = 0.2 

    # start
    while True:
        command_start = time.time()

        task_command_data = task_command_listener.get_key_states()
        if task_command_data["updated"]:
            CURRENT_COMMAND = COMMAND_MACHINE.process_io(task_command_data["value"])

            if CURRENT_COMMAND == COMMAND_MACHINE.MOVE_TO_TASK_HOME:
                print(f"COMMAND: MOVE_TO_TASK_HOME")
                data_collection_scheduler.exec_home_movement(wait=False)
            elif CURRENT_COMMAND == COMMAND_MACHINE.EXECUTE_START_STATE_COLLECTION:
                print(f"COMMAND: EXECUTE_START_STATE_COLLECTION")
                data_collection_scheduler.exec_collection(mode="start")
            elif CURRENT_COMMAND == COMMAND_MACHINE.EXECUTE_CURRENT_STATE_COLLECTION:
                print(f"COMMAND: EXECUTE_CURRENT_STATE_COLLECTION")
                data_collection_scheduler.exec_collection(mode="current")

        command_end = time.time()
        wait_time = COMMAND_PERIOD - (command_end - command_start)
        if wait_time > 0.:
            time.sleep(wait_time)





