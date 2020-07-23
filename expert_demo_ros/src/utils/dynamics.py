import numpy as np


class DynamicsModel(object):
    def __init__(self):
        print("DynamicsModel class")

    def propagate(self, state, controls):
        raise NotImplementedError("Not implemented yet")
        
class SingleIntegrator(DynamicsModel):
    def __init__(self):
        '''
        x_t+1 = x_t + u dt
        '''
        super(SingleIntegrator, self).__init__()
    
    def propagate(self, state, controls, dt):
        return state + dt * controls

class SimpleCar(DynamicsModel):
    def __init__(self):
        '''
        d[x, y, th, v] = [Vcos(th), Vsin(th), om, a]
        controls are (om, a)
        '''
        super(SimpleCar, self).__init__()

    def propagate(self, state, controls):
        x, y, th, V = state
        om, a = controls
        return state + dt * np.array([V*np.cos(th), V*np.sin(th), om, a])
