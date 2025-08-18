from enum import Enum, auto
from neuromeka import control_msgs


class ROBOT_STATE:
    IDLE = 5
    MOVE = 6
    DIRECT_TEACHING = 7
    TELE_OP = 17
    STOP = 0
    VIOLATE = 2
    COLLISION = 8
    
    @staticmethod
    def in_failure_state(robot_state):
        return robot_state in [ROBOT_STATE.STOP, ROBOT_STATE.VIOLATE, ROBOT_STATE.COLLISION]


class NN_CONTROL_STATE(Enum):
    TASK_IN_PROGRESS = auto()
    TASK_FINISH = auto()
    TASK_SUCCESS = auto()
    TASK_FAIL = auto()
    TASK_NOT_READY = auto()
    ROBOT_FAIL = auto()
    
    @staticmethod
    def get_out_control_loop(control_state):
        return control_state != NN_CONTROL_STATE.TASK_IN_PROGRESS


class ROBOT_CONTROL_MODE:
    TELE_TASK_ABSOLUTE = control_msgs.TELE_TASK_ABSOLUTE
    TELE_JOINT_ABSOLUTE = control_msgs.TELE_JOINT_ABSOLUTE


##########################################
from pynput import keyboard

class KeyboardListener:
    def __init__(self, key_targets):
        """
        params:
            key_targets: keyboard candidates to save states
        """
        self.key_targets = key_targets
        self.key_states = dict()
        for key_target in key_targets:
            self.key_states[key_target] = False
        self.updated = False

        self.listener = keyboard.Listener(on_press=self.on_press)
        self.listener.start()

    def on_press(self, key):
        key_str = str(key)
        key_str = key_str.strip("''")

        # update key states
        # self.updated = True
        for key_target in self.key_targets:
            if key == key_target or key_str == key_target:
                self.updated = True
                self.key_states[key_target] = True
            else:
                self.key_states[key_target] = False

    def get_key_states(self):
        data = dict()
        data["updated"] = self.updated
        data["value"] = self.key_states
        self.updated = False
        return data


##########################################
import importlib

def load_NN_controller(controller_type: str):
    module_path = f"middle_level_controller.{controller_type}.controller"
    try:
        module = importlib.import_module(module_path)
        return module.NN_controller
    except ModuleNotFoundError:
        raise ValueError(f"No module found for controller type '{controller_type}'") from None
    except AttributeError:
        raise ValueError(f"'NN_controller' not found in {module_path}") from None