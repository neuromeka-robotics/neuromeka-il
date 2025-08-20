import time
from threading import Thread


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
    def __init__(self, port = '/dev/ttyUSB0', slave_address = 9):
        super(RobotiqUSBClient, self).__init__()
        
        from pyRobotiqGripper import RobotiqGripper as RobotiqGripperClient
        self.gripper = RobotiqGripperClient(portname=port, slaveAddress=slave_address)
        
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
    

if __name__ == "__main__":
    gripper = RobotiqUSBClient()
    gripper.simple_test()
