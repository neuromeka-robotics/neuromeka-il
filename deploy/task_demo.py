import argparse
import importlib
import time
from multiprocessing import Process, Queue

def process_input(event_queue, interactive_compliance=False):
    import getch

    print("Commands:")
    print("0: Stop execution")
    print("1: Move to first task home")
    print("2: Execute first task")
    print("3: Execute start state dagger")
    print("4: Execute current state dagger")
    if interactive_compliance:
        print("r: Toggle RL policy compliance command (both arms)")
    print("q: Quit")
    print("Waiting for commands...")
    while True:
        k = getch.getch()
        if k in ["0", "1", "2", "3", "4", "q"] or (k == "r" and interactive_compliance):
            event_queue.put(k)
            time.sleep(0.01) # Need to sleep to make sure other process can get the input
            if k == "q":
                print("Quitting...")
                break
        else:
            print("Invalid command")

def _start_visualizer(args, config):
    enabled = args.viser if args.viser is not None else getattr(config, "VISER_ENABLED", False)
    if not enabled:
        return None
    if args.demo_task not in ("box_lift_open_loop", "box_lift_rl"):
        raise ValueError("Policy Viser telemetry is supported for box_lift_open_loop and box_lift_rl")
    from data_collector.visualizer import CollectionVisualizer
    from helper.eir_pink_ik_visualizer import DEFAULT_CONFIG_PATH

    teleop = config.CUSTOM_TASK_CONFIG.control_config.teleop_config
    config_path = (
        getattr(teleop, "pink_config_path", None) or DEFAULT_CONFIG_PATH)
    return CollectionVisualizer(
        config_path, host=args.viser_host, port=args.viser_port, hz=args.viser_hz)


def _run_demo(demo_task, nn_controller):
    # Check DAGGER and set data_collector if necessary
    if nn_controller.task_config.data_config is not None:
        ####################
        ## DAGGER enabled ##
        ####################
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
    interactive_compliance = getattr(nn_controller, "interactive_compliance_enabled", False)
    p = Process(target=process_input, args=(event_queue, interactive_compliance))
    try:
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
                    nn_controller.exec_nn_control_stop()
                    data_collection_scheduler.exec_collection(mode="start")
            elif k == '4':
                if data_collection_scheduler is not None:
                    print("Executing current state dagger")
                    nn_controller.exec_nn_control_stop()
                    data_collection_scheduler.exec_collection(mode="current")
            elif k == '0':
                print("Stopping execution")
                nn_controller.exec_nn_control_stop()
            elif k == 'r' and interactive_compliance:
                nn_controller.toggle_rl_compliance_command()
            elif k == 'q':
                break
    finally:
        try:
            nn_controller.exec_nn_control_stop()
        finally:
            if p.pid is not None:
                p.join(timeout=1.)
                if p.is_alive():
                    p.terminate()
                    p.join(timeout=1.)
            event_queue.close()


def main():
    parser = argparse.ArgumentParser(description="Run a robot task policy.")
    parser.add_argument("demo_task")
    parser.add_argument(
        "--viser", action=argparse.BooleanOptionalAction, default=None,
        help="Override VISER_ENABLED in the task's config.py.")
    parser.add_argument("--viser-host", default="127.0.0.1")
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--viser-hz", type=float, default=15.)
    args = parser.parse_args()

    from helper.extra_utils import load_NN_controller

    controller_type = load_NN_controller(controller_type=args.demo_task)
    config = importlib.import_module(f"middle_level_controller.{args.demo_task}.config")
    visualizer = _start_visualizer(args, config)
    try:
        nn_controller = controller_type(visualizer=visualizer)
        nn_controller.load_policy()
        _run_demo(args.demo_task, nn_controller)
    finally:
        if visualizer is not None:
            visualizer.close()


if __name__ == "__main__":
    main()
