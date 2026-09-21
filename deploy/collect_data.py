import time
from enum import Enum, auto

from helper.controller_utils import Controller
import getch
import sys

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
    # set data collector and robot connection
    from data_collector.collector import DataCollectionScheduler
    data_collection_scheduler: Controller = DataCollectionScheduler(config_name=sys.argv[1])
    print("Commands:")
    print("1: Move to task home")
    print("2: Collect from start state")
    print("3: Collect from current state")
    print("Waiting for commands...")
    while True:
        char = getch.getch()
        if char == '1':
            print("Moving to task home")
            data_collection_scheduler.exec_home_movement(wait=False)
        elif char == '2':
            print("Collecting from start state")
            data_collection_scheduler.exec_collection(mode="start")
        elif char == '3':
            print("Collecting from current state")
            data_collection_scheduler.exec_collection(mode="current")
        else:
            print("Invalid command")
