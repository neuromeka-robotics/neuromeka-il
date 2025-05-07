
class Empty_NN_policy:
    """
    Policy without loaded weights
    """
    def __init__(self, task):
        self.task = task
    
    def reset(self):
        pass

    def __call__(self, **kwargs):
        pass
    

class NN_policy:
    """
    Policy with loaded weights
    """
    def __init__(self, task):
        self.task = task
        
    def reset(self):
        pass
    
    def __call__(self, **kwargs):
        return {"control": None}