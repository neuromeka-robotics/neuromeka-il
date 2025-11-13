import time
import sys
from enum import Enum, auto

from helper.extra_utils import load_NN_controller
from helper.controller_utils import Base_NN_controller
import getch
from multiprocessing import Process, Queue

def process_input(event_queue):
    print("Commands:")
    print("0: Stop execution")
    print("1: Move to first task home")
    print("2: Execute first task")
    print("3: Execute start state dagger")
    print("4: Execute current state dagger")
    print("q: Quit")
    print("Waiting for commands...")
    while True:
        k = getch.getch()
        if k in ["0", "1", "2", "3", "4", "q"]:
            event_queue.put(k)
            time.sleep(0.01) # Need to sleep to make sure other process can get the input
            if k == "q":
                print("Quitting...")
                break
        else:
            print("Invalid command")

if __name__ == "__main__":
    # Get task name from command line argument
    demo_task = sys.argv[1]
    
    # Set nn controller and robot connection
    NN_controller = load_NN_controller(controller_type=demo_task)
    nn_controller: Base_NN_controller = NN_controller()
    
    # Load nn model weights
    nn_controller.load_policy()
    
    # Check DAGGER and set data_collector if necessary
    if nn_controller.task_config.data_config is not None:
        ####################
        ## DAGGER enabled ##
        ####################
        import importlib
        from helper.controller_utils import Controller
        from helper.config_utils import ROBOT_CONFIG, TASK_CONFIG
        from data_collector.collector import DataCollectionScheduler
        from data_collector.config import DATA_COLLECTOR_ROBOT_CONFIG, DATA_COLLECTOR_TASK_CONFIG
        
        # Overwrite robot and task configurations for data collector
        module = importlib.import_module(f"middle_level_controller.{demo_task}.config")
        DataCollectionScheduler.robot_config = module.CUSTOM_ROBOT_CONFIG
        DataCollectionScheduler.task_config = module.CUSTOM_TASK_CONFIG
        
        data_collection_scheduler: Controller = DataCollectionScheduler(
            robot=nn_controller.robot,
            camera=nn_controller.camera,
            dagger_mode=True
        )
    else:
        #####################
        ## DAGGER disabled ##
        #####################
        data_collection_scheduler = None

    event_queue = Queue()
    p = Process(target=process_input, args=(event_queue,))
    p.start()

    while True:
        k = event_queue.get()
        if k == '1':
            print("Moving to first task home")
            nn_controller.exec_home_movement(wait=False)
        elif k == '2':
            print("Executing first task")
            nn_controller.exec_nn_control(duration=1200.)
        elif k == '3':
            if data_collection_scheduler is not None:
                print("Executing start state dagger")
                # Stop nn controller
                nn_controller.exec_nn_control_stop()
                
                # Start data collection from home pos
                data_collection_scheduler.exec_collection(mode="start")
        elif k == '4':
            if data_collection_scheduler is not None:
                print("Executing current state dagger")
                # Stop nn controller
                nn_controller.exec_nn_control_stop()
                
                # Start data collection from current pos
                data_collection_scheduler.exec_collection(mode="current")
        elif k == '0':
            print("Stopping execution")
            nn_controller.exec_nn_control_stop()
        elif k == 'q':
            break

    p.join()
