from typing import List, Dict, Union
from threading import Thread
import time

from imitation_learning.model import Empty_NN_policy, NN_policy
from config.robot import CONTROL


class NN_controller:
    def __init__(self, tasks: List[str]):
        self.nn_tasks = tasks
        
        self.nn_policy: Dict[str, Union[Empty_NN_policy, NN_policy]] = dict()
        for task in self.nn_tasks:
            self.nn_policy[task] = Empty_NN_policy(task)
        
        self._control_triggered = False
        self._control_thread = None
        
    def get_available_tasks(self):
        return self.nn_tasks
        
    def load_policy(self, task: str):
        assert task in self.nn_tasks, f"Unavailable task {task}"
        
        if isinstance(self.nn_policy[task], Empty_NN_policy):
            self.nn_policy[task] = NN_policy(task)
        
    def exec_nn_control(self, task: str, duration: float):
        assert task in self.nn_tasks, f"Unavailable task {task}"
        
        if not self._control_triggered:
            self._reset_control()
            
            self._control_triggered = True
            
            self._control_thread = Thread(target=self._nn_control_fn, args=(task, duration,), daemon=True)
            self._control_thread.start()
        else:
            print("Control loop is already triggered")
            
    def exec_nn_control_stop(self):
        self._control_triggered = False
        if self._control_thread is not None:
            self._control_thread.join()
            self._control_thread = None
            
    def _reset_control(self, task: str):
        self.nn_policy[task].reset()
        
    def _nn_control_fn(self, task: str, duration: float):
        control_loop_init = time.time()
        
        while self._control_triggered and (time.time() - control_loop_init < duration):
            control_start = time.time()
            
            #####################################################
            # 1. Get proprioception and exteroception            
            # 2. Forward pass neural network (i.e., NN_policy)  
            # 3. Execute to robot (i.e., MoveTele)              
            # 4. Sync control frequency                         
            
            control = self.nn_policy[task](q=None, qdot=None, rgb=None)["control"]
            #####################################################
            
            control_end = time.time()
            wait_time = CONTROL["period"] - (control_end - control_start)
            if wait_time > 0:
                time.sleep(wait_time)
                
        self._control_triggered = False
                
    
    
        
        