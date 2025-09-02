class BaseDevice:
    """
    Base class for every teleoperation device
    """
    CONTROL_MODE = None    
    
    def __init__(self):
        raise NotImplementedError
    
    def exit(self):
        pass
    
    def reset(self):
        pass
    
    def get_input(self, **kwargs):
        raise NotImplementedError