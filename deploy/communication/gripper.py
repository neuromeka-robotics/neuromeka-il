import time
from threading import Lock, Thread


class BaseGripperClient:
    def __init__(self, **kwargs):
        self.gripper_pos = 1
        self.is_grasping = False
        self._gripper_thread: Thread | None = None
        
    def open(self):
        self.gripper_pos = 1
        self.is_grasping = False
        raise NotImplementedError
    
    def close(self):
        self.gripper_pos = 0
        self.is_grasping = True
        raise NotImplementedError
    
    def MoveGripper(self, gripper_value):
        raise NotImplementedError
    
    def MoveGripperWThread(self, gripper_value):
        if self._gripper_thread is None or not self._gripper_thread.is_alive():
            self._gripper_thread = Thread(
                target=self.MoveGripper, args=(gripper_value,), daemon=True)
            self._gripper_thread.start()

    def MoveGripperWOThread(self, gripper_value):
        self.MoveGripper(gripper_value)
        

class RobotiqUSBClient(BaseGripperClient):
    # Variable shared across instance
    _initialized = False
    
    def __init__(self, port = '/dev/robotiq_2f85', slave_address = 9):
        super(RobotiqUSBClient, self).__init__()
        
        from pyrobotiqgripper import RobotiqGripper as RobotiqGripperClient
        self.gripper = RobotiqGripperClient(portname=port, slaveAddress=slave_address)
        if not self._initialized:
            self.initialize()
            self._initialized = True
        
    def open(self):
        try:
            self.MoveGripper(gripper_value=1)
        except:
            print("Gripper error")
        return self.gripper_pos, self.is_grasping 

    def close(self):
        try:
            self.MoveGripper(gripper_value=0)
        except:
            print("Gripper error")
        return self.gripper_pos, self.is_grasping
    
    def MoveGripper(self, gripper_value):
        raw_gripper_value = max(0, min(gripper_value, 1))
        raw_gripper_value = int((1 - raw_gripper_value) * 255)
        raw_gripper_pos, object_detected = self.gripper.goTo(position=raw_gripper_value, speed=255, force=10)
        
        self.gripper_pos = 1 - (float(raw_gripper_pos) / 255)
        self.is_grasping = object_detected
        
    def initialize(self):
        self.gripper.resetActivate()
        time.sleep(2.)
        
    def simple_test(self):
        print("Initializing gripper...")
        self.initialize()
        time.sleep(0.5)
        
        print("Closing gripper...")
        self.close()
        time.sleep(0.5)
        print(f"Position: {self.gripper_pos}, Object detected: {self.is_grasping}")
        
        print("Opening gripper...")
        self.open()
        time.sleep(0.5)
        print(f"Position: {self.gripper_pos}, Object detected: {self.is_grasping}")
    

class EndportDHGripperClient(BaseGripperClient):
    """DH gripper on an Indy endport; 1 is open and 0 is closed.

    Initializes on construction. Position feedback is cached and refreshed in
    the background (~500 ms per read). ``is_grasping`` is unsupported (False).
    """

    def __init__(self, robot_ip=None, tool_index=0, speed=100, force=100, robot=None):
        import inspect
        from neuromeka import IndyDCP3, device_msgs

        super().__init__()
        if not isinstance(tool_index, int) or tool_index < 0:
            raise ValueError("tool_index must be a non-negative integer")
        if robot is None and robot_ip is None:
            raise ValueError("Provide robot_ip or an existing IndyDCP3 client as robot")
        self.robot = robot if robot is not None else IndyDCP3(robot_ip=robot_ip)
        self.tool_index = tool_index
        self.speed = max(1, min(100, int(round(speed))))
        self.force = max(20, min(100, int(round(force))))
        self._commands = device_msgs.GripperCommand
        self._gripper_type = device_msgs.DH_GRIPPER
        self._supports_tool_index = "tool_index" in inspect.signature(
            self.robot.set_gripper_command).parameters
        self._indexed_getter = getattr(self.robot, "get_gripper_data_for", None)
        if tool_index != 0 and (not self._supports_tool_index or not callable(self._indexed_getter)):
            raise ValueError("This IndyDCP SDK only supports tool_index=0")
        self._feedback_lock = Lock()
        self._feedback_thread = None
        self._feedback_error = None
        self.initialize()
        # Obtain real feedback once at startup, before entering the control loop.
        self._gripper_pos = self._read_position()

    def _send(self, command, position=1000):
        params = dict(command=command, gripper_type=self._gripper_type,
                      pvt_data=[position, self.speed, self.force, 0])
        if self._supports_tool_index:
            params["tool_index"] = self.tool_index
        return self.robot.set_gripper_command(**params)

    def initialize(self):
        response = self.activate()
        time.sleep(1.)
        return response

    def activate(self):
        return self._send(self._commands.ACTIVATE)

    def deactivate(self):
        return self._send(self._commands.DEACTIVATE)

    def get_state(self):
        if callable(self._indexed_getter):
            return self._indexed_getter(self.tool_index)
        return self.robot.get_gripper_data()

    def _read_position(self):
        state = self.get_state()
        return max(0., min(1., float(state["gripper_position"]) / 1000.))

    def _refresh_position(self):
        try:
            position = self._read_position()
        except Exception as error:
            with self._feedback_lock:
                self._feedback_error = error
        else:
            with self._feedback_lock:
                self._gripper_pos = position
                self._feedback_error = None

    @property
    def gripper_pos(self):
        with self._feedback_lock:
            position = self._gripper_pos
            error = self._feedback_error
            # At most one read in flight; no queue of stale feedback requests.
            if self._feedback_thread is None or not self._feedback_thread.is_alive():
                self._feedback_thread = Thread(target=self._refresh_position, daemon=True)
                self._feedback_thread.start()
        if error is not None:
            raise RuntimeError("Endport gripper feedback read failed") from error
        return position

    @gripper_pos.setter
    def gripper_pos(self, value):
        self._gripper_pos = value

    def MoveGripper(self, gripper_value):
        import math

        value = float(gripper_value)
        if not math.isfinite(value):
            raise ValueError("gripper_value must be finite")
        position = int(round(max(0., min(1., value)) * 1000))
        return self._send(self._commands.SET_PVT, position)

    def open(self):
        return self.MoveGripper(1.)

    def close(self):
        return self.MoveGripper(0.)

    def simple_test(self):
        print("Closing gripper...")
        self.close()
        time.sleep(2.)
        print(f"Position: {self._read_position()}")

        print("Opening gripper...")
        self.open()
        time.sleep(2.)
        print(f"Position: {self._read_position()}")


if __name__ == "__main__":
    gripper = RobotiqUSBClient()
    # gripper = EndportDHGripperClient(robot_ip="192.168.0.95")
    gripper.simple_test()
