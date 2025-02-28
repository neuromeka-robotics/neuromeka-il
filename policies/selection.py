from enum import Enum
   

class RobotMode(Enum):
    SINGLE_ROBOT = 1
    SINGLE_ROBOT_GRIPPER = 2
    DUAL_ROBOT = 3
    DUAL_ROBOT_GRIPPER = 4
    
    @staticmethod
    def name_to_mode(name: str):
        if name == "single_robot":
            return RobotMode.SINGLE_ROBOT
        elif name == "single_robot_gripper":
            return RobotMode.SINGLE_ROBOT_GRIPPER
        elif name == "dual_robot":
            return RobotMode.DUAL_ROBOT
        elif name == "dual_robot_gripper":
            return RobotMode.DUAL_ROBOT_GRIPPER
        else:
            raise ValueError(f"Unavailable robot mode: {name}")
        
    @staticmethod
    def get_num_robots(mode):
        if mode in [RobotMode.SINGLE_ROBOT, RobotMode.SINGLE_ROBOT_GRIPPER]:
            num_robots = 1
        elif mode in [RobotMode.DUAL_ROBOT, RobotMode.DUAL_ROBOT_GRIPPER]:
            num_robots = 2
        else:
            raise ValueError(f"Unavailable robot mode: {mode}")
        return num_robots
        

class ControlMode(Enum):
    TASK_SPACE = 1
    RELATIVE_DELTA_TASK_SPACE = 2
    
    @staticmethod
    def name_to_mode(name: str):
        if name == "task_space":
            return ControlMode.TASK_SPACE
        elif name == "relative_delta_task_space":
            return ControlMode.RELATIVE_DELTA_TASK_SPACE
        else:
            raise ValueError(f"Unavailable control mode: {name}")
        
    @staticmethod
    def mode_to_action_name(mode):
        if mode == ControlMode.TASK_SPACE:
            return {"pos": "action.end_pos", "ori": "action.end_ori"}
        elif mode == ControlMode.RELATIVE_DELTA_TASK_SPACE:
            return {"pos": "action.relative_delta.end_pos", "ori": "action.relative_delta.end_ori"}
        else:
            raise ValueError(f"Unavailable control mode: {mode}")
    
    @staticmethod
    def get_candidate(name: str):
        if name == "task":
            return [ControlMode.TASK_SPACE, ControlMode.RELATIVE_DELTA_TASK_SPACE]
        else:
            raise ValueError(f"Unavailable property: {name}")
        
        