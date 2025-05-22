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
