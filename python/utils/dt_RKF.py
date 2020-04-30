from casadi import *
from horizon import *

class dt_RKF:
    def __init__(self, dict, F_integrator):
        self.dict = dict
        self.F_integrator = F_integrator

        self.keys = []
        for key in self.dict:
            self.keys.append(key)

    def compute(self, k):
        integrator_out = self.F_integrator(x0=self.dict['x0'][k], p=self.dict['p'][k], time=self.dict['time'][k])

        return integrator_out['dtf']

    def compute_nodes(self, from_node, to_node):
        dt_RKF = []

        for k in range(from_node, to_node):
            dt_RKF.append(self.compute(k))

        return vertcat(*dt_RKF)

