import argparse
from enum import Enum, auto


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


def main():
    parser = argparse.ArgumentParser(description="Collect robot demonstrations.")
    parser.add_argument("config_name")
    parser.add_argument(
        "--viser", action=argparse.BooleanOptionalAction, default=None,
        help="Show measured/commanded EIR overlay (default: enabled for lift_box).")
    parser.add_argument("--viser-host", default="127.0.0.1")
    parser.add_argument("--viser-port", type=int, default=8080)
    parser.add_argument("--viser-hz", type=float, default=15.)
    args = parser.parse_args()

    import getch
    from data_collector.config import CONFIGS

    config = CONFIGS[args.config_name]
    use_viser = args.viser if args.viser is not None else args.config_name == "lift_box"
    visualizer = None
    if use_viser:
        from communication.humanoid import HumanoidRobot
        from data_collector.visualizer import CollectionVisualizer

        robot_config = config.robot_config
        control_config = config.task_config.control_config
        if (len(robot_config.robot_ids) != 1
                or not issubclass(robot_config.robot_class, HumanoidRobot)):
            parser.error(
                "The EIR shadow viewer requires one HumanoidRobot. "
                "Use --no-viser for other robots.")
        visualizer = CollectionVisualizer(
            control_config.teleop_config.pink_config_path,
            host=args.viser_host, port=args.viser_port, hz=args.viser_hz)

    # set data collector and robot connection
    try:
        from data_collector.collector import DataCollectionScheduler
        data_collection_scheduler = DataCollectionScheduler(
            config_name=args.config_name, visualizer=visualizer)
        while True:
            char = getch.getch()
            if char == '1':
                print("Moving to task home")
                data_collection_scheduler.exec_home_movement(wait=False)
            elif char == '2':
                print("Data collecting from home pos")
                data_collection_scheduler.exec_collection(mode="start")
            elif char == '3':
                print("Data collecting from current pos")
                data_collection_scheduler.exec_collection(mode="current")
    finally:
        if visualizer is not None:
            visualizer.close()


if __name__ == "__main__":
    main()
